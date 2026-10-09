#!/usr/bin/env python3

"""
Script to plot mummer and lastz theoretical predictions vs simulations and alignment.

Potentially summarize the data in an other way than graphical representation
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from mosaic_method.fitting import theoretical_mld

ALIGNER_DELTA = {
    "lastz": 0.82,
    "mummer": 0.25,
    "lastz_corrected": 0.82,
    "mummer_corrected": 0.25,
    "lastz_uncorrected": 0.5,
    "mummer_uncorrected":0.2,
    "no_aligner": 3
}

def filter_many_and(list_of_masks):
    # from stackoverflow
    aggregate_mask = list_of_masks[0]

    for mask in list_of_masks[1:]:
        aggregate_mask = aggregate_mask & mask

    return aggregate_mask
def plot_pred(ax, binned_mld, aligner, param_tau, muc, mus, L0):
    "plots the predicted MLD for a given aligner"
    th_match_length = np.logspace(0, np.log10(max(binned_mld["match_length"])), 1000)
    th_res = theoretical_mld(param_tau, th_match_length, muc, mus, L0, ALIGNER_DELTA[aligner], integ=integrand)
    ax.plot(th_match_length, th_res, label=aligner)

def plot_sim(ax, binned_mld, aligner):
    "plots the simulated binned mld"
    ax.scatter(binned_mld["match_length"], binned_mld["freq"], label=f"simulated - {aligner}")


def plot_pred_sim(ax, binned_mld, aligner, param_tau, muc, mus, L0):
    "plots both the simulated and predicted mld for a given aligner"
    plot_pred(ax, binned_mld, aligner, param_tau, muc, mus, L0)
    plot_pred(ax, binned_mld, f"{aligner}_uncorrected", param_tau, muc, mus, L0)
    plot_sim(ax, binned_mld, aligner)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"aligner: {aligner}, tau: {param_tau:.1e}")


def plot_all_params(binned_mlds, params, muc, mus, L0):
    "plots a grid for every instance of parameter of the simulation and inference"
    # comment écrire une fonction hyper générale pour un problème spécifique 👍 
    if not "aligner" in params:
        print("no different aligners for plotting theoretical vs simulated")
        return
    params = {f"{param}": binned_mlds[param].drop_duplicates().to_list() for param in params}
    num_params = [len(x) for _, x in params.items()]
    # 🤯 grocervo 
    params_settings_tuples = [settings for settings in itertools.product(*params.values())]
    params_settings_dics = [dict(zip(names, values)) for names, values in \
                            zip(itertools.repeat(params.keys()), params_settings_tuples)]

    if len(num_params) == 1:
        num_params.append(1)
    fig, axs = plt.subplots(num_params[0], num_params[1], layout="constrained", figsize=(num_params[1]*5, num_params[0]*5))
    for ax, param_dic  in zip(axs.flat, params_settings_dics):
        aligner = param_dic["aligner"]
        param_tau = 2*param_dic["tree_height"]
        masks = [binned_mlds[param] == value for param, value in param_dic.items()]
        aggr_mask = filter_many_and(masks)
        plot_pred_sim(ax, binned_mlds[aggr_mask], aligner, param_tau, muc, mus, L0)
    fig.savefig("pred_vs_sim.png", dpi=300)

        

if __name__ == "__main__":
    binned_mlds = pd.read_csv("jc_repr/all_binned_mlds.csv")
    plot_all_params(binned_mlds, ["aligner", "tree_height"], 6e-11, 5e-9, 5e6)
