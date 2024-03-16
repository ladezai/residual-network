import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional

from argparse import ArgumentParser
from math import log, floor, sqrt
from pathlib import Path
import re
import os
import sys

def plot_weights(weights : list[np.ndarray], 
                 weights_name : list[str],
                 ylabel : str,
                 title : str = ""):
    fig, ax = plt.subplots() 

    for name, w in zip(weights_name, weights): 
        points = len(w)
        x = np.linspace(0,1, points)
        ax.plot(x, w, linewidth=2, label=name)

    plt.xlabel(r'$k/L$')
    plt.ylabel(ylabel)
    #plt.title()
    plt.legend()
    print(title)
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()
    

def load_weights_in_dir(models_dir : str,
                        ext : str = "pth", 
                        beta : float = 0.5,
                        pos_x : list[int] = [1], 
                        pos_y : list[int] = [0],
                        verbose : bool = False):
    p = Path(models_dir)
    LS = []
    for path in p.iterdir():
        if not path.is_file():
            continue 
        
        if not str(path).endswith(ext):
            continue

        nbrs = re.findall(r'\d+', str(path.stem))
        if verbose:
            print(f"{path=}, {nbrs=}")
        LS.append(int(nbrs[0])) 
    
    LS = sorted(LS)
    if verbose:
        print(f"{LS=}")

    weights = { (px,py) : [] for px, py in zip(pos_x, pos_y)}
    for L in LS:
        # pre-process the weights
        weight = torch.load(f"{models_dir}/resnet_{L}.{ext}")
        weight = [L**beta * layer.detach().numpy() for layer in weight]
        # selects the position required
        for px, py in zip(pos_x, pos_y):
            selected_weights = [layer[px,py] for layer in weight]
            weights[(px,py)].append(np.array(selected_weights)) #.reshape(-1))
    
    return weights, LS

if __name__ == "__main__":
    parser = ArgumentParser(prog="Visualization of a specific weights as trajectories")
    parser.add_argument("--ext", dest='file_ext', type=str, default='pth')
    parser.add_argument("--dir_path", dest='dir_path', type=str)
    parser.add_argument("--beta", dest='beta', type=float, default=0.5)
    args = parser.parse_args()

    weights, def_l = load_weights_in_dir(models_dir=args.dir_path.rstrip("/"),
                        ext=args.file_ext.lstrip("."),
                        pos_x = [0,1, 2, 7, 8, 9],
                        pos_y = [0,1, 2, 3, 5, 6],
                        beta=args.beta)

    for k, v in weights.items():
        ylab="W_{" + str(k[0])+  str(k[1]) + "}"
        plot_weights(v, weights_name=[f"L = {def_l[i]}" for i,l in enumerate(v)],
                        ylabel=rf"${ylab}$",
                         title=f"{args.dir_path.rstrip('/').split('/')[-1]}{k[0]}{k[1]}")

