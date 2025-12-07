import os
from types import SimpleNamespace
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment import Experiment
from utils.config import load_config, set_config
from pathlib import Path

def create_fillin_plot(dataset, plot_type, save_path=None, to_percent=False):
    v_path = Path(f"plots/output/results/Fillin-Plot-VANILLA-{dataset}-{plot_type}.npz")
    m_path = Path(f"plots/output/results/Fillin-Plot-MeMP-{dataset}-{plot_type}.npz")

    vanilla = np.load(v_path)
    memp    = np.load(m_path)

    x = np.arange(2, 14, 2)  # 2..12

    v_mean, v_std = vanilla["mean"], vanilla["std"]
    m_mean, m_std = memp["mean"],    memp["std"]

    if to_percent:
        v_mean, v_std = v_mean * 100, v_std * 100
        m_mean, m_std = m_mean * 100, m_std * 100

    v_low, v_up = v_mean - v_std, v_mean + v_std
    m_low, m_up = m_mean - m_std, m_mean + m_std

    assert len(x) == len(v_mean) == len(m_mean), \
        f"Length mismatch: x={len(x)}, vanilla={len(v_mean)}, memp={len(m_mean)}"

    plt.figure(figsize=(10,6))
    plt.fill_between(x, v_low, v_up, alpha=0.25, label='Vanilla ±1σ', zorder=1)
    plt.fill_between(x, m_low, m_up, alpha=0.25, label='MeMP ±1σ', zorder=1)

    plt.plot(x, v_mean, linewidth=2.5, label='Vanilla Mean', zorder=3, marker='o')
    plt.plot(x, m_mean, linewidth=2.5, label='MeMP Mean', linestyle='--', zorder=3, marker='s')

    plt.xlabel('Hops')
    plt.ylabel('Accuracy (%)' if to_percent else 'Accuracy')
    plt.title(f'{plot_type}: Vanilla vs MeMP on {dataset}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', format='svg')
    else:
        plt.show()

def gen_results(dataset, type):
    vanilla_mean, vanilla_std = [], []
    memp_mean, memp_std = [], []

    os.makedirs("plots/output/results", exist_ok=True)

    v_cfg_path = f"configs/VANILLA/{dataset}/{dataset}-{type}.yaml"
    m_cfg_path = f"configs/MeMP/{dataset}/{dataset}-{type}.yaml"

    for i in range(6):
        hop = 2 + (i * 2)
        custom_config = SimpleNamespace(
            hops=hop,
            outdir=None,
            log=False,
            hidden_dim=None,
            epoch=None,
            dropout=None,
            type=None,
            framework=None
        )

        vanilla_cfg = set_config(load_config(v_cfg_path), custom_config)
        v_exp = Experiment(vanilla_cfg, 5)
        mean, std = v_exp.run()
        vanilla_mean.append(mean); vanilla_std.append(std)

        memp_cfg = set_config(load_config(m_cfg_path), custom_config)
        m_exp = Experiment(memp_cfg, 5)
        mean, std = m_exp.run()
        memp_mean.append(mean); memp_std.append(std)

    vanilla_mean = np.array(vanilla_mean); vanilla_std = np.array(vanilla_std)
    np.savez(f"plots/output/results/Fillin-Plot-VANILLA-{dataset}-{type}.npz",
             mean=vanilla_mean, std=vanilla_std)

    memp_mean = np.array(memp_mean); memp_std = np.array(memp_std)
    np.savez(f"plots/output/results/Fillin-Plot-MeMP-{dataset}-{type}.npz",
             mean=memp_mean, std=memp_std)

    return ((vanilla_mean, vanilla_std), (memp_mean, memp_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("type")
    parser.add_argument("--log", action="store_true")  # make it a real flag
    args = parser.parse_args()

    v_path_string = f"plots/output/results/Fillin-Plot-VANILLA-{args.dataset}-{args.type}.npz"
    m_path_string = f"plots/output/results/Fillin-Plot-MeMP-{args.dataset}-{args.type}.npz"

    need_v = not os.path.exists(v_path_string)
    need_m = not os.path.exists(m_path_string)
    if need_v or need_m:
        print("Generating results...")
        gen_results(args.dataset, args.type)

    save_path = f"plots/output/fillin-plots/vanilla_vs_memp-{args.dataset}-{args.type}.svg"
    create_fillin_plot(args.dataset, args.type, save_path)
