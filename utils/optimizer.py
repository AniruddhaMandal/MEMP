import torch

def build_optimizer(cfg, model):
    if(cfg.Optimizer.type == 'Adam'):
        optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=cfg.Optimizer.learning_rate,
                        weight_decay=cfg.Optimizer.weight_decay
                        )
        return optimizer
    if(cfg.Optimizer.type == 'AdamW'):
        optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=cfg.Optimizer.learning_rate,
                        weight_decay=cfg.Optimizer.weight_decay
                        )
        return optimizer