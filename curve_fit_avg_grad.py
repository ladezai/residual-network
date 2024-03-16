import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import tensorboard as tb 
import tensorboard.backend.event_processing.event_accumulator as ea

from cycler import cycler

from typing import Optional

from argparse import ArgumentParser
from math import log, floor, sqrt
from pathlib import Path
import re
import os
import sys

from pvar import p_var

# Load event from tensor board
def load_events(paths : list[str], 
                tag : str,  
                keys : Optional[list[str]] = None) -> dict[str, list[float]]:
    """
        Given paths, returns a tensor of values in which each column
        represents the tag and the row the event path.
    """
    if keys is None or len(keys) != len(paths):
        keys = [path for path in paths]

    experiments = [ea.EventAccumulator(path) for path in paths] 
    results     = {key : [] for key in keys}
    for i, experiment in enumerate(experiments):
        experiment.Reload()
        for j, scalar in enumerate(experiment.Scalars(tag)):
            results[keys[i]].append(scalar.value)

    return results

def data_cleaning(data : np.array, 
                  filter_period : int = 0,
                  window_size   : int = 1,
                  windows_map_func : "np.array -> bool_array" = None ):
    if windows_map_func is None:
        windows_map_func = lambda x: x
   
    # Augment data with steps
    new_data = np.stack([data, np.arange(1, data.shape[0]+1)])

    # Remove an element every filter_period data points
    if filter_period > 0:
        filtered_data = new_data[:, new_data[1,:] % filter_period != 0]
    else:
        filtered_data = new_data.copy()    

    # Split in all the windows, than apply accum_func 
    if window_size >= 1:
        windows  = np.array_split(filtered_data, 
                                  filtered_data.shape[1] // window_size, 
                                  axis=1)
        windows  = [window[:, windows_map_func(window[0,:])] for window in windows]
        result   = np.column_stack(windows)
        return result
    else:
        return filtered_data
    

# Function we want to regress to
def funcinv_shift(x, C, rho, shift):
    return C/(x+shift)**rho

def funcinv(x, C, rho):
    return C/(x+1)**rho

# Compute MSE loss optimal value
def regress(xs, ys, shift=False):
    if shift:
        popt, pcov = curve_fit(funcinv_shift, xs,
                               ys, bounds=([0,0,1], [1,3,500]),
                               method='trf')
    else:
        popt, pcov = curve_fit(funcinv, xs,
                               ys, bounds=(0, 3),
                               method='trf')
    return popt, pcov

if __name__ == "__main__":
    parser = ArgumentParser(prog="""Curve fitting procedure given a tensorboard
                            event having a parameter called `avg_grad_norm`""")
    parser.add_argument("--event_path",   type=str)
    parser.add_argument("--filter_period", type=int,   default=0)
    parser.add_argument("--infer_shift",   type=bool,  default=False)
    parser.add_argument("--window_size",   type=int,   default=5)
    parser.add_argument("--scaling_x",     type=float, default=1.0)
    args = parser.parse_args()

    event_path = args.event_path.rstrip("/") 
    res = load_events([event_path], 'avg_grad_norm',
                      keys=["example"]) 
    ys = np.array(res['example'])
    clean_ys = data_cleaning(ys, filter_period=args.filter_period, 
                           window_size=args.window_size, 
                           # Remove the maximum and min point
                           windows_map_func = lambda x : (x!=np.max(x)) & (x!=np.min(x)))
    _, N = clean_ys.shape
    data_points = clean_ys[0,:]
    xs = clean_ys[1,:] * args.scaling_x
    
    params, cov = regress(xs, data_points, args.infer_shift)
    variables_std_error = np.sqrt(np.diag(cov))/np.sqrt(N)
    print(fr"Estimated parameter values ($C$, $\rho$): {params}")
    print(f"SD of the parameters: {np.sqrt(np.diag(cov))}")
    print(f"Estimated parameters std error: {variables_std_error}")

    plt.rc('text', usetex=True)
    fig, ax = plt.subplots() 

    params0_val = r"\cdot 10^{".join(f"{params[0]:4.2g}".split("e"))
    params1_val = r"\cdot 10^{".join(f"{params[1]:4.2g}".split("e")) 
    if args.infer_shift:
        params2_val = r"\cdot 10^{".join(f"{params[2]:4.2g}".split("e")) 
        if "{" in params2_val:
            params2_val = params2_val + "}"

    if "{" in params0_val:
        params0_val = params0_val + "}"
    if "{" in params1_val:
        params1_val = params1_val + "}"
    
    shift = params2_val if args.infer_shift else 1
    label_function = r"$g(t) = \frac{" + params0_val +"}{(t+" + f"{shift})"+ "^{"+ params1_val + "}}$"
    if args.infer_shift:
        # In order to have a more dense representation we recreate a linear 
        # vector ranging from 0 to the highest data point index
        xx = np.arange(int(np.max(xs/args.scaling_x))+1) * args.scaling_x 
        ax.plot(xs, funcinv_shift(xx, params[0], params[1], params[2]),
                c='royalblue',
                linewidth=3,
                label=label_function)
    else:
        xx = np.arange(int(np.max(xs/args.scaling_x))+1) * args.scaling_x 
        ax.plot(xx, funcinv(xx, params[0], params[1]), c='royalblue',
                linewidth=3, label=label_function) 
        
        # PREDICTION code
        #params_pred, cov_pred = regress(xs[10:100], data_points[10:100], args.infer_shift)
        #print(f"{params_pred=}")
        #print(f"SD of the parameters: {np.sqrt(np.diag(cov_pred))}")
        #xx = np.arange(int(np.max(xs/args.scaling_x))+1) * args.scaling_x 
        #ax.plot(xx, funcinv(xx, params_pred[0], params_pred[1]), c='purple',
        #        linewidth=3, label="prediction")
    ax.scatter(xs, data_points, marker='x', c="orange",label="data")
    #lower_bounds = funcinv(xs, params[0]-1.96*variables_std_error[0],
                           #params[1]+1.96*variables_std_error[1])
    #upper_bounds = funcinv(xs, params[0]+1.96*variables_std_error[0],
                           #prams[1]-1.96*variables_std_error[1])
    #ax.fill_between(XS, lower_bounds, upper_bounds, alpha=.25)
    plt.xlabel('epoch (t)')
    plt.ylabel(r"$\frac{1}{N}\sum_{i=1}^N \| \nabla \ell_i\|^2$")
    plt.title("")
    plt.legend()
    path = "-".join(event_path.split("/")[:2])
    plt.savefig(f"avg_grad_norm-{path}.png", dpi=300)
    plt.show()


