import yaml
import datetime
from types import SimpleNamespace


# possibly create a config data-type and add some printing method to it.
# return the config data-type 
def load_config(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = dict_to_class(config)
    return config

def dict_to_class(obj):
    if isinstance(obj,dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                obj[key] = dict_to_class(value)
            
        return SimpleNamespace(**obj)
    else: 
        return obj

def class_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: class_to_dict(val) if isinstance(val, SimpleNamespace) else val for key, val in vars(obj).items()}
    else:
        return obj

def save_config(file_path, cfg):
    cfg = class_to_dict(cfg)
    with open(file_path, 'w') as f:
        yaml.dump(cfg,f)

def set_config(cfg, args):
    cfg.log = args.log
    if(args.hops != None):
        cfg.Model.hops = args.hops
    if(args.hidden_dim != None):
        cfg.Model.hidden_dim = args.hidden_dim
    if(args.epoch != None):
        cfg.Train.max_epoches = args.epoch
    if(args.dropout != None):
        cfg.Model.dropout_frac = args.dropout
    if(args.type != None):
        cfg.Model.type = args.type
    if(args.framework != None):
        cfg.Model.framework = args.framework
    if(args.rewiring != None):
        cfg.Data.rewiring = args.rewiring

    # type casting 
    cfg.Optimizer.weight_decay = float(cfg.Optimizer.weight_decay)
    cfg.Optimizer.learning_rate = float(cfg.Optimizer.learning_rate)
    
    now = datetime.datetime.now()
    dd = now.strftime("%d")
    mm = now.strftime("%m")
    s = now.strftime("%S")
    m = now.strftime("%M")
    h = now.strftime("%H")
    uid = f"{dd}.{mm}-{s}:{m}:{h}" 
    if(cfg.Model.framework == "message-passing-lstm"):
        framework_name = "MPLSTM"
    elif(cfg.Model.framework == "vanilla"):
        framework_name = "VANILLA"
    elif(cfg.Model.framework == "MeMP"):
        framework_name = "MeMP"
    else: 
        framework_name = ""
    
    if(args.outdir != None):
        parent_dict = f"Results/{args.outdir}"
    else:
        parent_dict = "Results/Experiments"
    cfg.outdir_path = f"{parent_dict}/{cfg.Data.name}/Exp-{framework_name}-{cfg.Model.type}-{uid}"

    return cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
