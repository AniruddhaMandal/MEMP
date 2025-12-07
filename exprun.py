#!/usr/bin/env python3

import os
import datetime
import subprocess
import argparse
from utils.config import load_config
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", type=str, default="True", help="Set the loging on/off. Default value is True.")
    parser.add_argument("-hp", "--hops", type=str, default=None, help="If not passed value will be taken from config file.")
    parser.add_argument("-e", "--epoch", type=str, default=None, help="If not passed value will be taken from config file.")
    parser.add_argument("-dp", "--dropout", type=str, default=None, help="If not passed value will be taken from config file.")
    parser.add_argument("-hd", "--hidden_dim", type=str, default=None, help="If not passed value will be taken from config file.")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="Directory where the results will be stored. If not passed files will be saved to `Results` directory.")
    parser.add_argument("-t", "--type", type=str, default=None, help="Type for which experiments will run. If not passed experiments run for all types.")
    parser.add_argument("-d", "--data", type=str, default=None, help="Dataset for which experiments will run. If not passed experiments run for all datasets.")
    parser.add_argument("-f","--framework", type=str, default=None, help="Framework for which experiments will run. If not passed experiments run for all frameworks.")
    parser.add_argument("-s", "--split", help="Number of experiment splits.")
    args = parser.parse_args()

    root_dir = Path('./configs')
    yaml_files = list(root_dir.rglob("*.yaml"))
    failed_exp = []

    now = datetime.datetime.now()
    dd = now.strftime("%d")
    mm = now.strftime("%m")
    s = now.strftime("%S")
    m = now.strftime("%M")
    h = now.strftime("%H")
    
    os.makedirs(".log", exist_ok=True)    
    log_file = f".log/Exp_log_{dd}.{mm}--{s}:{m}:{h}.log"
    for f in yaml_files:
        cfg = load_config(f)
        if(args.framework != None):
            f_run = (args.framework == cfg.Model.framework)
        else:
            f_run = True
        if(args.type != None):
            t_run = (args.type == cfg.Model.type)
        else:
            t_run = True
        if(args.data != None):
            d_run = (args.data == cfg.Data.name)
        else:
            d_run = True
        run = f_run*t_run*d_run

        if run:
            command = ["python", "main.py", "--cfg", f]
            if(args.hops != None):
                command += ["--hops", args.hops]
            if(args.hidden_dim != None):
                command += ["--hidden_dim", args.hidden_dim]
            if(args.dropout != None):
                command += ["--dropout", args.dropout]
            if(args.epoch != None):
                command += ["--epoch", args.epoch]
            if(args.outdir != None):
                command += ["--outdir", args.outdir]
            if(args.log != None):
                command += ["--log", args.log]
            if(args.split != None):
                command += ["--split", args.split]

            #print(command)
            try:
                result = subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True) 
            except subprocess.CalledProcessError as e:
                failed_exp.append(f)
                with open(log_file,"a") as l:
                    l.write(f"****************{f}**************\n")
                    l.write(e.stderr)
                    l.write("\n\n")

    if len(failed_exp)!=0:
        print("Failed Experiments for:")
        for f in failed_exp:
            print(f)