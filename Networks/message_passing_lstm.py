import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.nn as pygn
from torch_geometric.graphgym.models.encoder import AtomEncoder
from utils.encoder import LinearEncoder

class WGCN(nn.Module):
    """ Computes weighted graph convolution.
        :math:`H^t = \sigma(UX+\sum WAH^{(t-1)})` 
    """
    def __init__(self, hidden_dim, type='GCN', *args, **kwargs):
        """ 
            Input: `X` (original representation)
            Output: :math:`H^t = \sigma(UX+\sum WAH^{(t-1)})`"""
        super(WGCN, self).__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim

        if type == "GCN":
            self.u = pygn.GCNConv(self.hidden_dim,self.hidden_dim)
            self.w = pygn.GCNConv(self.hidden_dim,self.hidden_dim)
        if type == "GIN":
            nn_obj_u = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),
                                     nn.BatchNorm1d(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim,self.hidden_dim))
            
            nn_obj_w = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),
                                     nn.BatchNorm1d(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim,self.hidden_dim))

            self.u = pygn.GINConv(nn_obj_u, eps=0.1)
            self.w = pygn.GINConv(nn_obj_w, eps=0.1)
        if type == "GAT":
            self.u = pygn.GATConv(self.hidden_dim,self.hidden_dim)
            self.w = pygn.GATConv(self.hidden_dim,self.hidden_dim)
        if type == "GSAGE":
            self.u = pygn.GraphSAGE(self.hidden_dim,self.hidden_dim, num_layers=1)
            self.w = pygn.GraphSAGE(self.hidden_dim,self.hidden_dim, num_layers=1)
        if type == "RGGC":
            self.u = pygn.ResGatedGraphConv(self.hidden_dim,self.hidden_dim)
            self.w = pygn.ResGatedGraphConv(self.hidden_dim,self.hidden_dim)

    def forward(self, X: Tensor, H: Tensor, edge_index: Tensor) -> Tensor:
        """ Args: 
                `X`: Original representation.
                `H`: Temporal representation.
            
            Returns: 
                Tensor: :math:`H^t = \sigma(UX+\sum WAH^{(t-1)})`
                """
        H = self.u(X,edge_index) + self.w(H,edge_index)
        return H

class LSTMConv(nn.Module):
    def __init__(self, hidden_dim, dropout, type='GCN', *args, **kwargs):
        super(LSTMConv, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.forget_conv = WGCN(self.hidden_dim, type=type)
        self.input_conv = WGCN(self.hidden_dim, type=type)
        self.output_conv = WGCN(self.hidden_dim, type=type)

        self.U = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

    def forward(self, batch):
        x, h, c, edge_index = batch.x, batch.h, batch.c, batch.edge_index

        f = self.forget_conv(x, h, edge_index)
        f = F.sigmoid(f)
        c = c * f

        i = self.input_conv(x, h, edge_index) * F.tanh(x @ self.U)
        i = F.sigmoid(i)
        c = c + i
        c = F.dropout(c,p=self.dropout,training=self.training)

        o = self.output_conv(x, h, edge_index)
        o = F.sigmoid(o)
        h = o * F.tanh(c)
        h = F.dropout(h,p=self.dropout,training=self.training)

        batch.h, batch.c = h, c
        
        return batch


class MessagePassingLSTM(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim, hops, dropout, type='GCN', encoder=None, *args, **kwargs):
        super(MessagePassingLSTM, self).__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hops = hops
        self.dropout = dropout

        if encoder == 'Atom':
            self.input_layer = AtomEncoder(emb_dim=self.hidden_dim)
        else:
            self.input_layer = LinearEncoder(self.in_dim, self.hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(hops):
            self.layers.append(LSTMConv(self.hidden_dim,self.dropout,type=type))
        
        self.cls_layer = nn.Linear(self.hidden_dim,self.out_dim)

    def forward(self, batch):
        batch = self.input_layer(batch)

        batch.h = torch.zeros_like(batch.x)
        batch.c = torch.zeros_like(batch.x)

        for i in range(self.hops):
            batch = self.layers[i](batch)

        y_pred = self.cls_layer(batch.c)
        #y_pred = F.softmax(y_pred,dim=1)
        y_pred = pygn.global_mean_pool(y_pred,batch.batch)
        

        return y_pred