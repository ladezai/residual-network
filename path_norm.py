import torch
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from typing import Optional

from argparse import ArgumentParser
from math import log, floor, sqrt
from pathlib import Path
import re
import os


from pvar import p_var

def load_weights_in_dirs(models_dirs : [str],
                        ext : str = "pth", 
                        beta : float = 0.5,
                        norm : bool = True,
                        verbose : bool = False):
    # Theoretically, we should have the same model in each of the 
    # models_dirs stored in the argument.
    LS = []
    p = Path(models_dirs[0])
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
    
    weights = {L : [] for L in LS}
    for L in LS: 
        for models_dir in models_dirs:
            weight = torch.load(f"{models_dir}/resnet_{L}.{ext}")

            # pre-process the weights
            weight = [L**beta * layer.detach().numpy().reshape(-1) for layer in weight]
            if norm:
                weight = [np.linalg.norm(w,keepdims=True) for w in weight]
            # selects the position required
            weights[L].append(weight)
    # Returns a dictionary whose key are the number of layers, 
    # but the value are the models_dir's respective models.
    # e.g. 2000 : [ [weights for the first model], 
    #               [weights of the second model], ...]
    return weights


def load_weights_in_dir(models_dir : str,
                        ext : str = "pth", 
                        beta : float = 0.5,
                        norm : bool = True,
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
    
    weights = {L : [] for L in LS}
    for L in LS:
        # pre-process the weights
        weight = torch.load(f"{models_dir}/resnet_{L}.{ext}")

        weight = [L**beta * layer.detach().numpy() for layer in weight]
        if norm:
            weight = [np.linalg.norm(w, ord='fro') for w in weight]
        # selects the position required
        weights[L] = weight
    
    return weights

if __name__ == "__main__":
    parser = ArgumentParser(prog="Visualize the weights' norm as a trajectory")
    parser.add_argument("--ext",       dest='file_ext',  type=str,   default='pth')
    parser.add_argument("--dir_path",  dest='dir_path',  type=str)
    parser.add_argument("--beta",      dest='beta',      type=float, default=0.5)
    parser.add_argument("--visualize_CI", dest='vci',    type=bool,  default=False)
    parser.add_argument("--pvar_only", dest='pvar_only', type=bool,  default=False)
    parser.add_argument("--pvar",      dest='pvar',      type=float, default=1.0)
    parser.add_argument("--verbose",   dest='verbose',   type=bool,  default=False)
    
    args = parser.parse_args()

    if args.verbose:
        print(args.dir_path)

    weights = load_weights_in_dirs(models_dirs=[dir.rstrip("/") for dir in args.dir_path.split()],
                        ext=args.file_ext.lstrip("."),
                        norm=not args.pvar_only, # NOTE! Here changes how to load
                                  # the weights, depending on the choice of
                                  #  computation of the pvar only, i.e. only the
                                  #  weights' norm pvar or the pvar of the weights.
                        beta=args.beta,
                        verbose=args.verbose)
    print(f"Computes pvar only: {args.pvar_only}.")
    if args.pvar_only:
        for k, v in weights.items():
            pvar_values = []
            for model in v:
                weights    = np.array(model) #[vv.reshape(-1) for vv in model])
                pvar_value = p_var(weights, args.pvar, 
                                   'euclidean') ** (1/args.pvar)
                pvar_values.append(pvar_value)
            max_pvar = np.array(pvar_values).max()
            blank_spaces = " " * (10 - len(str(k)))
            print(f"Max 1-variation for $W_{k}$:{blank_spaces}${max_pvar:4.4g}$")
    else:
        # USE LATEX RENDERING
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots() 
        plt.xlabel(r'$k/L$')
        # TEST THIS LABEL CONSTRUCTION!
        b = f"{args.beta}"
        ylabel = "$\|L^{"+ f"{b}"+"} W_\cdot^{(L)}\|$"
        plt.ylabel(rf"{ylabel}")
        ax.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    
        # DON'T compute the pvar as it is bounded by the previous 
        # computation in a straightforward manner. Hence, reduce the number of 
        # tables.
        for k, v in weights.items():
            x         = np.linspace(0, 1, k)
            data      = np.zeros( (len(v), k) )
            #pvar_values    = []
            for i, model in enumerate(v):
                weights    = np.array(model)
                #pvar_val = p_var(weights, args.pvar, 'euclidean') ** (1/args.pvar)
                #pvar_values.append(pvar_val)
                data[i, :] = weights[:,0]
            # Visualization
            avg_plot = data.mean(axis=0) 
            ax.plot(x, avg_plot, linewidth=2, label=rf"$L =$ {k}")
            if k >=2 and args.vci:
                std_plot = data.std(axis=0) 
                lower_bounds = avg_plot - 1.96 * std_plot / sqrt(3)
                upper_bounds = avg_plot + 1.96 * std_plot / sqrt(3)
                ax.fill_between(x, lower_bounds, upper_bounds, alpha=.05)
            
            #pvar_max = np.array(pvar_values).max()
            #pvar_std = np.array(pvar_values).std()
            #blank_spaces = " " * (10 - len(str(k)))
            #print(f"Worst 1-variation for $W_{k}$:{blank_spaces}${pvar_max:4.2g}$")

        plt.legend()
        fullpath = args.dir_path.split()[0]
        interesting_bit = "".join(fullpath.split("/")[1:])
        plt.savefig(f"beta={args.beta}-{interesting_bit}.png", dpi=300)
        plt.show()


