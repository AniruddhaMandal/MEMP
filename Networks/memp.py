import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.nn as pygn
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils.encoder import LinearEncoder

class SequenceGCN(pygn.MessagePassing):
    def __init__(self, in_dim, out_dim, bias=False, *args, **kwargs):
        super(SequenceGCN, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_u = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.weight_w = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight_u)
        nn.init.xavier_uniform_(self.weight_w)
        if bias:
            nn.init.zeros_(self.bias)
	
    def forward(self, initial_feature, temporal_feature, edge_index):
        transformed_initial_feature = torch.mm(initial_feature, self.weight_w)
        transformed_temporal_feature = torch.mm(temporal_feature, self.weight_u)
        neighbour_accumulated_temporal_feature = self.propagate(edge_index=edge_index, x=transformed_temporal_feature)
        output = neighbour_accumulated_temporal_feature + transformed_initial_feature
        return output


class MeMPConv(pygn.MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, type='GCN', *args, **kwargs):
        super(MeMPConv, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.type = type

        self.weight_w = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        nn.init.xavier_uniform_(self.weight_w)
        self.forget_conv = SequenceGCN(in_dim=self.in_dim,out_dim=self.out_dim)
        self.input_conv = SequenceGCN(in_dim=self.in_dim,out_dim=self.out_dim)

        if type == "GCN":
            self.temporal_conv = pygn.GCNConv(self.in_dim, self.out_dim)
        if type == "GIN":
            nn_obj_u = nn.Sequential(nn.Linear(self.in_dim,self.out_dim),
                                     nn.BatchNorm1d(self.out_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.out_dim,self.out_dim))
            self.temporal_conv = pygn.GINConv(nn_obj_u, eps=0.1)
        if type == "GAT":
            self.temporal_conv = pygn.GATConv(self.in_dim,self.out_dim)
        if type == "GSAGE":
            self.temporal_conv = pygn.GraphSAGE(self.in_dim,self.out_dim, num_layers=1)
        if type == "RGGC":
            self.temporal_conv = pygn.ResGatedGraphConv(self.in_dim,self.out_dim)
        if type == "GCNII":
            self.temporal_conv = pygn.GCN2Conv(self.in_dim,alpha=0.1,theta=0.5,layer=1)


    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        memory_feature = batch.memory_feature
        temporal_feature = batch.temporal_feature

        # Forget 
        theta = F.sigmoid(self.forget_conv(x, temporal_feature, edge_index))
        memory_feature = memory_feature*theta

        # Input
        input_weight = F.tanh(torch.mm(x, self.weight_w))
        input_feature = F.sigmoid(self.input_conv(x, temporal_feature, edge_index))
        input_feature = input_feature*input_weight
        memory_feature = F.sigmoid(memory_feature+input_feature)

        # Output 
        norm_edge_index, norm_edge_weight = gcn_norm(edge_index,edge_weight=None,num_nodes=batch.num_nodes,add_self_loops=True)
        memory_feature = self.propagate(norm_edge_index, x=memory_feature,edge_weight=norm_edge_weight) # Ablation with nomalized propagation
        if self.type == "GCNII":
            temporal_feature = self.temporal_conv(x=temporal_feature, x_0=x, edge_index=edge_index)
        else:
            temporal_feature = self.temporal_conv(x=temporal_feature, edge_index=edge_index)

        memory_feature = F.dropout(memory_feature, p=self.dropout, training=self.training)
        temporal_feature = F.dropout(temporal_feature, p=self.dropout, training=self.training)

        batch.memory_feature = memory_feature
        batch.temporal_feature = temporal_feature
        return batch


class MeMP(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim, hops, dropout, type='GCN', encoder=None,signal=False, *args, **kwargs):
        super(MeMP, self).__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hops = hops
        self.dropout = dropout
        self.signal = signal

        if encoder == 'Atom':
            self.input_layer = AtomEncoder(emb_dim=self.hidden_dim)
        else:
            self.input_layer = LinearEncoder(self.in_dim, self.hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(hops):
            self.layers.append(
                MeMPConv(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    dropout=self.dropout,
                    type=type)
                )
        self.cls_layer = nn.Linear(self.hidden_dim,self.out_dim)

    def forward(self, batch):
        batch = self.input_layer(batch)

        batch.temporal_feature = torch.zeros_like(batch.x)
        batch.memory_feature = torch.zeros_like(batch.x)

        for i in range(self.hops):
            batch = self.layers[i](batch)

        # Turn on signal for computing signal vs resistance graph.
        if self.signal == True:
            return F.sigmoid(batch.memory_feature) 

        y_pred = self.cls_layer(F.log_softmax(batch.memory_feature))
        y_pred = pygn.global_mean_pool(y_pred,batch.batch)
        
        return y_pred
