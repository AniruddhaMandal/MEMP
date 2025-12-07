import torch

def all_one_encoder(graphs):
    for g in graphs:
        g.x = torch.ones(g.num_nodes, 1)
    return graphs