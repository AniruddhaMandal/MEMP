import torch.optim as optim

def get_scheduler(optimizer, cfg):
    if cfg.Scheduler.name == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode = cfg.Scheduler.mode,
                    factor = cfg.Scheduler.reduce_factor
                )
        return scheduler
    else:
         return None