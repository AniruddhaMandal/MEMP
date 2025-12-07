from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

def model_table(cfg):
    table = Table(border_style='dim')
    table.add_column("Parameter", style='blue')
    table.add_column("Value", style='magenta')

    table.add_row("Framework", cfg.Model.framework)
    table.add_row("Type",cfg.Model.type)
    table.add_row("Encoder",cfg.Model.encoder)
    table.add_row("Hidden Dim", str(cfg.Model.hidden_dim))
    table.add_row("Hops", str(cfg.Model.hops))
    table.add_row("Dropout", str(cfg.Model.dropout_frac))

    return table

def data_table(cfg):
    table = Table(border_style='dim')
    table.add_column("Parameter", style='green')
    table.add_column("Value", style='yellow')

    table.add_row("Name", cfg.Data.name)
    table.add_row("Batch Size", str(cfg.Data.batch_size))
    table.add_row("Train Frac", str(cfg.Data.train_frac))
    table.add_row("Val Frac", str(cfg.Data.val_frac))
    return table

    
def config_logger(cfg,model,loss_fn,metric_fn,optimizer,scheduler,device):
    model_panel = Panel(
        str(model),
        title=f"{model.__class__.__name__} Architecture",
        border_style="dim",
        expand=True
    )
    training_text = Text()
    training_text.append("Optimizer:\n",style="cyan underline")
    training_text.append(f" learning rate->{cfg.Optimizer.learning_rate}\n", style="cyan")
    training_text.append(f" weight decay->{cfg.Optimizer.weight_decay}\n", style="cyan")
    training_text.append("Loss:\n",style="blue underline")
    training_text.append(f" {loss_fn}\n",style="blue")
    training_text.append("Metric:\n",style="red underline")
    training_text.append(f" {metric_fn}\n", style="red")
    training_text.append(f"Scheduler:\n",style="underline")
    training_text.append(f" {scheduler.__class__.__name__}\n")
    training_text.append(f" mode->{cfg.Scheduler.mode}\n")
    training_text.append(f" reduce factor->{cfg.Scheduler.reduce_factor}\n")
    training_text.append(f"Device:\n",style="yellow underline")
    training_text.append(f" {device}", style="yellow")
    training_panel = Panel(
        training_text,
        title=f"Training Configuration",
        border_style="dim",
        expand=True
    )
    model_cfg_panel = Panel(model_table(cfg),title="Model Parameters",border_style="dim")
    data_cfg_panel = Panel(data_table(cfg),title="Data Parameters",border_style="dim")
    console = Console()
    console.print(Columns([model_cfg_panel,model_panel]))
    console.print(Columns([data_cfg_panel, training_panel]))

def _config_logger(cfg):
    console = Console()
    console.print(Columns([model_table(cfg),data_table(cfg)]))

class LoggPerformance():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_panel = Panel(model_table(self.cfg), border_style='dim')
        self.data_panel = Panel(data_table(self.cfg), border_style='dim')

    def add_perf(self, exp_i, loss, acc, time):
        perf_table = Text()
        perf_table.append(f"Model {self.cfg.Model.framework}\n",style="bold underline")
        perf_table.append(f"🔬 Experiment: ")
        perf_table.append(f"\t{exp_i}\n", style='cyan')
        perf_table.append(f"🔄 Epoch: ")
        perf_table.append(f"      {self.cfg.Train.max_epoches}\n", style='blue')
        perf_table.append(f"🎯 Accuracy: ")
        perf_table.append(f"  {acc*100: .2f}%\n", style='green')
        perf_table.append(f"📉 Loss: " )
        perf_table.append(f"      {loss: .4f}\n", style='red')
        perf_table.append(f"⏱  Time Taken: ")
        perf_table.append(f"{time: .2f}s\n",style='yellow')
        self.perf_panel = Panel(perf_table,border_style='dim')

    def print(self):
        self.console = Console()
        self.console.print(Columns([self.perf_panel,self.model_panel,self.data_panel]))