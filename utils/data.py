import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from utils.preprocess import all_one_encoder

def get_dataset(cfg):
    """ Finds out the dataset from dataset name of `cfg`.
        Returns dataset in list format """
    rootdir = "Dataset"
    tudataset_names = ["MUTAG", "ENZYMES", "PROTEINS", "IMDB-BINARY", "REDDIT-BINARY", "COLLAB"]
    lrgb_names = ["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"]

    if(cfg.Data.name in tudataset_names):
        graphs = TUDataset(rootdir,cfg.Data.name)
        graphs = list(graphs)
    if(cfg.Data.name in lrgb_names):
        train_data = LRGBDataset(rootdir, cfg.Data.name,split="train")
        val_data = LRGBDataset(rootdir,cfg.Data.name, split="val")
        test_data = LRGBDataset(rootdir,cfg.Data.name, split="test")
        graphs = list(train_data)+list(val_data)+list(test_data)
    if graphs[0].x == None:
        if cfg.Data.node_feature_fill == 'all-one-feature':
            graphs = all_one_encoder(graphs)
    return graphs

def get_dataloaders(cfg):
    """ Generates train, val and test dataloaders."""
    train_frac = cfg.Data.train_frac
    val_frac = cfg.Data.val_frac
    batch_size = cfg.Data.batch_size

    dataset = get_dataset(cfg)
    train_data, val_data, test_data = random_split(dataset,[train_frac, val_frac,  1-(train_frac+val_frac)]) 

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

    return train_loader, val_loader, test_loader