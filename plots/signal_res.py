import os
import sys
import argparse
import copy
import numpy as np
import random
import torch 
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch
import pandas as pd
from rich.progress import Progress, TaskProgressColumn, TextColumn, BarColumn,TimeRemainingColumn
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
import matplotlib. pyplot as plt

from utils.model import build_model
from utils.data import get_dataloaders, get_dataset
from utils.config import load_config
from Networks.memp import MeMP

DEVICE = "cpu"

def smooth_plot(x, y=None, ax=None, label='', halflife=10,color=None):
    if y is None:
      y_int = x
    else:
      y_int = y
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)

    if color == None:
        colors = list(mpl.rcParams['axes.prop_cycle'])
        color = random.choice(colors)['color']
    if y is None:
      line = ax.plot(x_ewm.mean(), label=label, color=color)
      ax.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.15)
    else:
      line = ax.plot(x, x_ewm.mean(), label=label, color=color)
      ax.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.15)
    return line


def get_resistances(graph, reference_node):
    resistances = {}
    for node in graph.nodes:
        if node != reference_node:
            resistances[node] = round(nx.resistance_distance(graph, reference_node, node),3)
        else:
            resistances[node] = 0
    return resistances

def get_pairs(framework, dataset_name, model_type):
    os.makedirs("plots/output", exist_ok=True)
    os.makedirs("plots/output/results", exist_ok=True)
    config_file_path = f"configs/{framework}/{dataset_name}/{dataset_name}-{model_type}.yaml"
    config = load_config(config_file_path)
    config.Device = DEVICE
    config.Data.batch_size = 1
    config.Data.train_frac = 0.98
    config.Data.val_frac = 0.01
    config.Model.hops = 10

    model = build_model(config)
    dataset, _, _ = get_dataloaders(config)
    #dataset = get_dataset(config)

    pb = Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("• [{task.completed} / {task.total}]"),
                TimeRemainingColumn())
    task = pb.add_task(f"[green]{model_type}",total=len(dataset))

    sample_size=10
    diameters = []
    pairs = []

    pb.start()
    for d in dataset:
        try:
            pairs_runaway = []
            G = to_networkx(d, to_undirected=True)
            diameters.append(nx.diameter(G))
            distances = nx.floyd_warshall_numpy(G)
            for i in range(sample_size):
                source = np.random.randint(0, len(G)) #pick a source node
                max_t_A_st = distances[source].max()
                x = torch.zeros_like(d.x) #put zero almost everywhere
                x[source] = torch.randn_like(d.x[source]) #set random mass
                x[source] = x[source].softmax(dim=-1) #normalize to unitary positive mass on the source node
                d.x.data = x #set that feature matrix
                input_graph = copy.deepcopy(d)
                model = build_model(config,signal=True)
                output_graph = model(input_graph)
                acc = 0.0
                for j in range(len(output_graph)):
                    acc += output_graph * distances[j, source]
                propagation = ((1/max_t_A_st) * acc).mean().detach()

                total_effective_resistance = sum(get_resistances(G, source).values())
                x_y = (total_effective_resistance, propagation)
                pairs_runaway.append(x_y)
        except Exception as e:
            print("Graph Discconnected, Cannot compute effectve resistance.")
            continue
        pb.update(task,advance=1)
        pairs.append(tuple(np.array(pairs_runaway).mean(axis=0).tolist()))
    pb.stop()
    np.save(f"plots/output/results/{framework}-{dataset_name}-{model_type}.npy",np.array(pairs))
    return pairs


def gen_plot_from_saved_data(dataset_name,model_type,colors=(None,None)):
    v_path_string = f"plots/output/results/VANILLA-{dataset_name}-{model_type}.npy"
    v_pairs = np.load(v_path_string)
    v_pairs = v_pairs[v_pairs[:,0].argsort()]

    m_path_string = f"plots/output/results/MeMP-{dataset_name}-{model_type}.npy"
    m_pairs = np.load(m_path_string)
    m_pairs = m_pairs[m_pairs[:,0].argsort()]

    fig,ax = plt.subplots(1,1,figsize=(18,15))
    vanilla_line, = smooth_plot(
        x=MinMaxScaler().fit_transform(v_pairs[:,0].reshape(-1,1)).flatten(), 
        y=MinMaxScaler().fit_transform(v_pairs[:,1].reshape(-1,1)).flatten(), 
        ax=ax, 
        halflife=2,
        color=colors[0]
        )
    memp_line, = smooth_plot(
        x=MinMaxScaler().fit_transform(m_pairs[:,0].reshape(-1,1)).flatten(), 
        y=MinMaxScaler().fit_transform(m_pairs[:,1].reshape(-1,1)).flatten(), 
        ax=ax, 
        halflife=2,
        color=colors[1]
        )
    vanilla_line.set_label("VANILLA")
    memp_line.set_label("MeMP")
    ax.legend(fontsize=24)
    fig.text(0.5, 0.04,  'Normalized Total Effective Resistance', size=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Signal Propagation',  size=20, ha='center', va='center', rotation='vertical')
    plt.savefig(f"plots/output/Resistance_MeMP_vs_VANILLA_{dataset_name}_{model_type}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model_type")
    args = parser.parse_args()

    v_path_string = f"plots/output/results/VANILLA-{args.dataset}-{args.model_type}.npy"
    m_path_string = f"plots/output/results/MeMP-{args.dataset}-{args.model_type}.npy"

    if not os.path.exists(v_path_string):
        get_pairs("VANILLA",args.dataset,args.model_type)
    if not os.path.exists(m_path_string):
        get_pairs("MeMP",args.dataset,args.model_type)
        
    color_v = None # Change for specific color for Vanilla line 
    color_m = None # Change for specific color for MeMP line
    colors= (color_v,color_m)

    gen_plot_from_saved_data(args.dataset,args.model_type,colors=colors)
