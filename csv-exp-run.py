#!/usr/bin/env python3
from types import SimpleNamespace
import csv
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
    parser.add_argument("csv_file",type=str,help="CSV file to load experiment from.")
    args = parser.parse_args()
    file_path = args.csv_file

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for exp in reader:
            custom_config = SimpleNamespace(
                hops = int(exp['Hops'].strip()),
                outdir = exp['Outdir'].strip(),
                log = str2bool(exp['Log'].strip()),
                hidden_dim = None,
                epoch = None,
                dropout = None,
                type = None,
                framework = None
                )
            data = exp['DatasetName'].strip()
            fw = exp['Framework'].strip()
            t = exp['Type'].strip()
            cfg_file_path = f'configs/{fw}/{data}/{data}-{t}.yaml'
            cfg = load_config(file_path=cfg_file_path)
            cfg = set_config(cfg, custom_config)
            set_random_seed(seed=42)
            experiment = Experiment(cfg,num_split=25)
            experiment.run()
        