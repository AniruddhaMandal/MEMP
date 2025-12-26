import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from utils.preprocess import all_one_encoder
from utils.rewiring import apply_rewiring

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

    # Apply graph rewiring if specified in config
    if hasattr(cfg.Data, 'rewiring') and cfg.Data.rewiring is not None:
        print(f"Applying {cfg.Data.rewiring.upper()} graph rewiring...")

        if cfg.Data.rewiring == 'sdrf':
            # SDRF parameters
            loops = getattr(cfg.Data, 'rewiring_loops', 10)
            tau = getattr(cfg.Data, 'rewiring_tau', 1.0)
            remove_edges = getattr(cfg.Data, 'rewiring_remove_edges', False)
            use_cuda = getattr(cfg.Data, 'rewiring_use_cuda', False)

            graphs = apply_rewiring(
                graphs,
                method='sdrf',
                loops=loops,
                tau=tau,
                remove_edges=remove_edges,
                is_undirected=True,
                use_cuda=use_cuda
            )
            print(f"  SDRF: loops={loops}, tau={tau}, remove_edges={remove_edges}, use_cuda={use_cuda}")

        elif cfg.Data.rewiring == 'fosr':
            # FoSR parameters
            num_iterations = getattr(cfg.Data, 'rewiring_num_iterations', 50)

            graphs = apply_rewiring(
                graphs,
                method='fosr',
                num_iterations=num_iterations
            )
            print(f"  FoSR: num_iterations={num_iterations}")

        else:
            raise ValueError(f"Unknown rewiring method: {cfg.Data.rewiring}. Choose 'sdrf' or 'fosr'.")

        print(f"  Rewiring complete! Total graphs: {len(graphs)}")

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