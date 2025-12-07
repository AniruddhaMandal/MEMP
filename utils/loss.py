import torch.nn as nn
def get_loss_fn(cfg):
    if(cfg.Train.loss_fn == 'cross-entropy'):
        return nn.CrossEntropyLoss()
    if(cfg.Train.loss_fn == 'multilable-cross-entropy'):
        return nn.BCEWithLogitsLoss()
    if(cfg.Train.loss_fn == "l1"):
        return nn.L1Loss()