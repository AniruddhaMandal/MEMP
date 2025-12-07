import os
import shutil
import time
import uniplot
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import numpy as np
import torch

from utils.evaluate import train_epoch, test_epoch
from utils.data import get_dataloaders
from utils.model import build_model
from utils.loss import get_loss_fn
from utils.metric import get_metric_fn
from utils.optimizer import build_optimizer
from utils.config import save_config
from utils.logging import config_logger, LoggPerformance
from utils.scheduler import get_scheduler

class Experiment():
    """
    Experiment object configured through config file.
    
    Extract information about experiment from config file.
    Runs experiment for `num_split` times. Saves mean and 
    std of the experiment accuracies. For each split loads 
    model, loss function, metric, data loader for different
    splits of the data each time(unless specified otherwise) 

    From configuration file detects the model, loss function
    , metric, data loader, execution device (i.e. `cuda` or 
    `cpu`). Builds the model, loss function,metric, 
    optimizer, shceduler, data loaders.

    Attributes
    ----------
    cfg : dict
        Configuration for experiment. Loaded form `.yaml` 
        file to a dictionary.
    num_split : int
        Number of times experiments will run.  
    """
    def __init__(self, cfg, num_split):
        """
        Intializes `Experiment` object

        Parameters
        ----------
        cfg : dict
        num_split : int
        """
        self.num_split = num_split
        self.cfg = cfg

        # output path handling
        self.path_temp = cfg.outdir_path
        cfg.outdir_path = f".temp/{cfg.outdir_path}"
        os.makedirs(cfg.outdir_path,exist_ok=True)
    
    def run(self):
        """
        Runs experiment loop `num_split` times. In each 
        experiment model, loss function,metric, optimizer, 
        shceduler, data loaders. Each experiment contains a 
        training loop. In each training loop it trains model 
        and calulates performance for validation data. 
        Training loop runs for `cfg.Train.max_epoch` times. 
        End of each experiment loggs performance on the 
        validation data and experiment configuration.
        Also stores performance in an array. Finally, `mean` 
        and `std` of this array is logged and saved in 
        `result.txt` file in output directory.
        """
        cfg = self.cfg
        experiment_acc_history = []

        for exp_i in range(self.num_split):
            perf_logger = LoggPerformance(self.cfg)
            train_loader, val_loader, test_loader = get_dataloaders(cfg)
            model = build_model(cfg)
            loss_fn = get_loss_fn(cfg)
            metric_fn = get_metric_fn(cfg)
            optimizer = build_optimizer(cfg, model)
            scheduler = get_scheduler(optimizer,cfg)
            device = cfg.Device

            if(cfg.log and (exp_i == 0)):
                config_logger(cfg,model,loss_fn,metric_fn,optimizer,scheduler,device)
            train_loss_history = []
            optimal_model_acc = 0

            pbar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style= 'yellow',finished_style='green'),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                TextColumn("[red]train_loss:{task.fields[train_loss]}"),
                TextColumn("[blue]val_acc:{task.fields[val_acc]}")
            )
            task = pbar.add_task(f"[yellow]EXP {exp_i}:",
                                 total=cfg.Train.max_epoches,
                                 train_loss=1,
                                 val_acc=0)
            pbar.start()
            model.to(device)
            start_time = time.perf_counter()
            for i in range(cfg.Train.max_epoches):    
                train_loss, train_acc = train_epoch(model,train_loader,loss_fn,metric_fn,optimizer,scheduler,device)
                train_loss_history.append(train_loss.item())
                val_loss, val_acc = test_epoch(model, val_loader, loss_fn, metric_fn, device)

                if(optimal_model_acc<val_acc):
                    optimal_model_acc = val_acc
                    torch.save(model,f"{cfg.outdir_path}/optimal_model.pt")
                pbar.update(task,advance=1,
                            train_loss=f"{train_loss.item():.6f}",
                            val_acc=f"{val_acc.item()*100: .2f}%")
            pbar.stop()
            del model 

            optimal_model = torch.load(f"{cfg.outdir_path}/optimal_model.pt",weights_only=False)
            opt_loss, opt_acc = test_epoch(optimal_model,test_loader,loss_fn,metric_fn,device)
            del optimal_model

            experiment_acc_history.append(opt_acc.item())
            end_time = time.perf_counter()
            if(cfg.log):
                uniplot.plot(train_loss_history, 
                                lines=True,
                                title=f"Train Loss History {cfg.Data.name} {cfg.Model.type}",
                                character_set='braille')
                perf_logger.add_perf(exp_i, opt_loss.item(), opt_acc.item(),time=(end_time-start_time)) 
                perf_logger.print()
        
        return_str = f"{np.mean(experiment_acc_history)*100: 0.2f}\u00B1{(np.std(experiment_acc_history)/np.sqrt(self.num_split))*100: 0.2f}" 
        print("Experiment Acc: ",return_str)
        with open(f"{cfg.outdir_path}/result.txt",'w') as f:
            f.write(return_str)

        save_config(f"{cfg.outdir_path}/config.yaml",cfg)
        shutil.move(self.cfg.outdir_path,self.path_temp)
        return (np.mean(experiment_acc_history)*100, (np.std(experiment_acc_history)/np.sqrt(self.num_split))*100)
