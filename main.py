import warnings
import torch
import argparse
import numpy as np
from utils.config import load_config, set_config, str2bool
from utils.experiment import Experiment

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=Warning)
def set_random_seed(seed,gpu=True):
    torch.random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Experiment configuration yaml file.")
    parser.add_argument("--log", type=str2bool, default=True, help="Logging On/Off")
    parser.add_argument("--hops", type=int, help="Number of Message Passing Layers")
    parser.add_argument("--hidden_dim", type=int, help="Dimension of the hidden space")
    parser.add_argument("--type", type=str, help="Model GNN type")
    parser.add_argument("--epoch", type=int, help="Number of epochs per experiment.")
    parser.add_argument("--dropout", type=float,help="Dropout Fraction")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--split", type=int, default=25, help="Number of experiment splits.")
    parser.add_argument("--framework", type=str)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = set_config(cfg, args)

    set_random_seed(seed=42)
    experiment = Experiment(cfg,num_split=args.split)
    experiment.run()