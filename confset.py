#!/usr/bin/env python3

import argparse
from utils.config import save_config, load_config
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--add", type=str, help="Sets argument to the target config files.")
    parser.add_argument("--hops", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("-dt","--data", type=str)
    parser.add_argument("-t", "--type", type=str)
    parser.add_argument("-f","--framework", type=str, default=None, help="Name of framework to be configured.\
                        If not passed changes all framework configs.")
    parser.add_argument("-l", "--loss", type=str)
    parser.add_argument("-m", "--metric", type=str)
    parser.add_argument("-nff", "--node_feature_fill", type=str)
    parser.add_argument("--scheduler_name", type=str)
    parser.add_argument("--scheduler_mode", type=str)
    parser.add_argument("--scheduler_factor", type=float)
    parser.add_argument("--add_field", type=str)
    parser.add_argument("--del_field", type=str)
    args = parser.parse_args()

    root_dir = Path('./configs')
    yaml_files = list(root_dir.rglob("*.yaml"))
    for f in yaml_files:
        cfg = load_config(f)
        if(args.framework != None):
            f_change = (args.framework == cfg.Model.framework)
        else:
            f_change = True
        if(args.type != None):
            t_change = (args.type == cfg.Model.type)
        else:
            t_change = True
        if(args.data != None):
            d_change = (args.data == cfg.Data.name)
        else:
            d_change = True
        change = f_change*t_change*d_change
        
        if change:
            if(args.add != None):
                print("Enter Valule:")
                value = str(input())
                try:
                    setattr(cfg,args.add, value)
                except:
                    print("Parent field is not paresent. Add it with --add_field.")
            if(args.add_field != None):
                try:
                    setattr(cfg,args.add_field,dict({}))
                except:
                    print("Parent field is not paresent. Add it with --add_field.")
            if(args.del_field != None):
                delattr(cfg, args.del_field)
            if(args.hops != None):
                cfg.Model.hops = args.hops
            if(args.hidden_dim != None):
                cfg.Model.hidden_dim = args.hidden_dim
            if(args.dropout != None):
                cfg.Model.dropout_frac = args.dropout
            if(args.epoch != None):
                cfg.Train.max_epoches = args.epoch
            if(args.encoder != None):
                cfg.Model.encoder = args.encoder
            if(args.loss != None):
                cfg.Train.loss_fn = args.loss
            if(args.metric != None):
                cfg.Train.metric = args.metric
            if(args.node_feature_fill != None):
                cfg.Data.node_feature_fill = args.node_feature_fill
            if(args.scheduler_name != None):
                cfg.Scheduler.name = args.scheduler_name
            if(args.scheduler_mode != None):
                cfg.Scheduler.mode = args.scheduler_mode
            if(args.scheduler_factor != None):
                cfg.Scheduler.reduce_factor = args.scheduler_factor
            save_config(f,cfg)
