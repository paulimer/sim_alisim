#!/usr/bin/env python3

"""Functions to measure and plot the performance of the inference relative to the simulation parameters."""

import argparse
from io import StringIO
import os
import shlex
import subprocess as sp
import tempfile as tmp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from skbio import DistanceMatrix, TreeNode
from skbio.tree import upgma

from sim_alisim.simulate_infer import fit_mld

def nb_steps(row):
    return int(((row["sim_tau"]/2)/row["tree_height"]) * row["rw_step"])
def row_re(row):
    return np.abs((row["sim_tau"] - row["fit_tau"])/row["sim_tau"])

def collect_metrics(all_res_csv):
    """Reads the results csv and print basic metrics."""
    all_res_df = pd.read_csv(all_res_csv, dtype={"rw_step_fraction": str})
    all_res_df["fit_tau"] = all_res_df["fit_tau"].map(lambda x: 10**x)
    all_res_df["sim_tau"] = all_res_df["sim_tau"].map(lambda x: 10**x)
    all_res_df["rw_step"] = all_res_df["rw_step_fraction"].map(lambda x :float(x))
    all_res_df["nb_steps"] = all_res_df.apply(nb_steps, axis=1)
    all_res_df["relative_error"] = all_res_df.apply(row_re, axis=1)
    r2_general = sk.metrics.r2_score(all_res_df["sim_tau"], all_res_df["fit_tau"])
    num_step_group = all_res_df.groupby("rw_step_fraction")
    r2_step = []
    for gk in num_step_group.groups.keys():
        r2_step += [{"rw_step": gk, "r2": sk.metrics.r2_score(num_step_group.get_group(gk)["sim_tau"], num_step_group.get_group(gk)["fit_tau"])}]
    mae_df = pd.DataFrame(r2_step)
    print(f"General r2: {r2_general}")
    print("r2 by number of step:")
    print(mae_df)
    return all_res_df, mae_df


def dot_plot_steps(all_res_df):
    """Plots the infered distance against the true distance"""
    min_steps = all_res_df["nb_steps"].min()
    max_steps = all_res_df["nb_steps"].max()
    min_tau = min([all_res_df["sim_tau"].min(), all_res_df["fit_tau"].min()])
    max_tau = max([all_res_df["sim_tau"].max(), all_res_df["fit_tau"].max()])
    g = sns.FacetGrid(data=all_res_df, col="rw_step_fraction", col_wrap=4)#, hue="empirical")
    g.map_dataframe(sns.scatterplot, x="sim_tau", y="fit_tau")#, c="nb_steps")#, cmap="viridis", vmin=min_steps, vmax=max_steps, alpha=0.5)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    g.add_legend()
    g.savefig("dotplot_by_steps_mean.png", dpi=300)


def distr_exp_mu(all_res_df, fixed_muc, fixed_mus):
    """Plots the distribution of empirical mus and also muc and mus."""
    empirical_mus_df = all_res_df[all_res_df["empirical"] == True][["muc", "mus", "rw_step_fraction"]]
    exp_df = pd.melt(empirical_mus_df, id_vars=["rw_step_fraction"], value_vars=["muc", "mus"], var_name="mu", value_name="mutation_rate")
    g = sns.FacetGrid(exp_df, col="rw_step_fraction", col_wrap=4)
    g.map(sns.boxplot, "mu", "mutation_rate")
    g.set(yscale="log")
    g.refline(y=fixed_muc)
    g.refline(y=fixed_mus)
    g.savefig("boxplots_mus.png", dpi=300)




def violin_steps(all_res_df):
    """Plots MAE for each step number."""
    g = sns.FacetGrid(all_res_df, col="rw_step_fraction", col_wrap=4)
    g.map_dataframe(sns.boxplot, y="relative_error")
    g.set(ylabel="Relative error", xlabel="Steps from root to leaf", ylim=0)
    g.savefig("relative_error_boxplot_mean.png")



def plot_r2(r2_df):
    """Plots the tree-wise r squared for each tree step setting."""
    r2_df["rw_step"] = r2_df["rw_step"].astype(float).astype(int)
    fig, ax = plt.subplots()
    ax.scatter(r2_df["rw_step"], r2_df["r2"])
    ax.set_xlabel("Steps amount to cover tree height")
    ax.set_ylabel("R2")
    fig.savefig("r2_vs_steps_mean.png", dpi=300)


def add_mash(all_res_df, data_path):
    """Adds mash as a distance measure for comparisons."""
    all_dirs_data = [os.path.join(data_path, d) for d in os.listdir(data_path) if d.startswith("rw")]
    res_list_w_mash = []
    for d in all_dirs_data:
        rw_step_fraction = float(os.path.split(d)[1][-5:])
        d_df = all_res_df[all_res_df["rw_step_fraction"] == rw_step_fraction]
        genome_files = [os.path.join(d, g) for g in os.listdir(d) if g.endswith("fasta")]
        mash_cmd_str = f"mash triangle -p 10 -s 100000 -E {' '.join(genome_files)}"
        mash_cmd = shlex.split(mash_cmd_str)
        mash_res = sp.run(mash_cmd, capture_output=True, check=True, encoding="utf-8")

        mash_df = pd.read_csv(StringIO(mash_res.stdout), sep="\t", header=None, names=["genome_1", "genome_2", "mash_dist", "pval", "shared_hashes"])
        mash_df["genome_1"] = mash_df["genome_1"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        mash_df["genome_2"] = mash_df["genome_2"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        mash_df[["genome_1", "genome_2"]] = mash_df.apply(lambda x: (x.genome_1, x.genome_2) if x.genome_1 < x.genome_2 else (x.genome_2, x.genome_1), axis=1, result_type="expand")
        res_list_w_mash.append(pd.merge(d_df, mash_df, "inner", on=["genome_1", "genome_2"]))

    res = pd.concat(res_list_w_mash)
    return res


# ok mash is better
def mash_vs_mosaic(res_df, also_fixed=False):
    """Plots the mash distance against the mosaic distance."""
    if also_fixed:
        plot_df = res_df[["mash_dist", "rw_step_fraction", "sim_tau", "fit_tau"]]
    else:
        plot_df = res_df[res_df["empirical"] == True][["mash_dist", "rw_step_fraction", "sim_tau", "fit_tau"]].drop_duplicates()
    scaler = sk.preprocessing.MinMaxScaler()
    scaled = scaler.fit_transform(plot_df[["mash_dist", "fit_tau", "sim_tau"]])
    plot_df[["mash_dist", "fit_tau", "sim_tau"]] = scaled
    plot_df = plot_df.melt(id_vars=["rw_step_fraction", "sim_tau"], value_vars = ["fit_tau", "mash_dist"], value_name="normalized estimated distance", var_name="method")
    g = sns.FacetGrid(data=plot_df, col="rw_step_fraction", col_wrap=4)#, hue="empirical")
    g.map_dataframe(sns.scatterplot, x="sim_tau", y="normalized estimated distance", hue="method")#, c="nb_steps")#, cmap="viridis", vmin=min_steps, vmax=max_steps, alpha=0.5)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    g.add_legend()
    plt.show()





def refit_fixed_mus(all_res_df, res_path, muc, mus, delta, genome_length):
    """Refits the MLDs but with a fixed, assumed mu distribution"""
    # /home/paulimer/Data/simulated_datasets/sim_alisim_data/entero_sim_find_steps_cor_5k
    res_dirs = [os.path.join(res_path, dirp) for dirp in os.listdir(res_path) if dirp.startswith("res")]
    refit_list = []
    for res_dir in res_dirs:
        rw_step_fraction = res_dir[-5:]
        binned_mlds = os.path.join(res_dir, "binned_mlds")
        mlds_csv = [csv for csv in os.listdir(binned_mlds) if csv.endswith(".csv")]
        for mld_csv in mlds_csv:
            mld_df = pd.read_csv(os.path.join(binned_mlds, mld_csv))
            cur_comp = os.path.splitext(mld_csv)[0].split("_")
            _, res_opt, cur_comp = fit_mld(mld_df, muc, mus, delta, genome_length, cur_comp, True)
            fit_res = {"genome_1": cur_comp[0], "genome_2": cur_comp[1], "fit_tau": 10**res_opt.x[0], "muc":muc, "mus":mus, "rw_step_fraction": float(rw_step_fraction)}
            refit_list.append(fit_res)

    refit_df = pd.DataFrame(refit_list)
    refit_df["rw_step_fraction"] = refit_df["rw_step_fraction"].astype(str)
    all_res_df["rw_step_fraction"] = all_res_df["rw_step_fraction"].astype(float).astype(str)
    tmp_df = pd.merge(all_res_df.drop(['fit_tau', 'empirical_muc', 'empirical_mus'], axis=1), refit_df, on=("genome_1", "genome_2", "rw_step_fraction"))
    all_res_df.rename({"empirical_muc": "muc", "empirical_mus": "mus"}, inplace=True, axis=1)
    all_res_df["rw_step_fraction"] = all_res_df["rw_step_fraction"].astype(float)
    tmp_df["rw_step_fraction"] = tmp_df["rw_step_fraction"].astype(float)
    return pd.concat({True: all_res_df, False: tmp_df}).reset_index().drop("level_1", axis=1).rename({"level_0": "empirical"}, axis=1)


def fit_refit_distance(all_res_df):
    """Compares fits with correct and incorrect mus across the range of distances."""
    all_res_df["rw_step_fraction"] = all_res_df["rw_step_fraction"].astype(float).astype(int)
    all_res_df["relative_error"] = all_res_df.apply(lambda x: np.abs((x["sim_tau"] - x["fit_tau"])/x["sim_tau"]), axis=1)
    num_step_group = all_res_df.groupby("rw_step_fraction")
    # fig, ax = plt.subplot
    # for i, gk in enumerate(num_step_group.groups.keys()):
    #     rw_step_df = num_step_group.get_group(gk).copy()
    #     ax.scatter(rw_step_df["sim_tau"], rw_step_df["relative_error"]
    #     ax.scatter(rw_step_df["sim_tau"], rw_step_df["refit_relative_error"]
    g = sns.FacetGrid(all_res_df, col="rw_step_fraction", col_wrap=4)
    g.map(sns.boxplot, "empirical", "relative_error")
    g.savefig("fit_refit_distance_mean.png", dpi=300)


def make_upgma(res_df):
    tau_df = res_df[["genome_1", "genome_2", "fit_tau"]]
    tau_matrix = pd.DataFrame(np.concatenate([tau_df.values, tau_df.values[:, [1,0,2]]])).pivot(columns=0,index=1,values=2).fillna(0)
    dist_matrix = DistanceMatrix(tau_matrix, ids=tau_matrix.columns)
    tree_upgma = upgma(dist_matrix)
    return tree_upgma


# always 0 anyways
def RF_stepsize(all_res_df, ori_tree_str):
    """Computes the RF distance as a function of the parameter size"""
    ori_tree = TreeNode.read([ori_tree_str])
    empirical_df = all_res_df[all_res_df["empirical"] == True]
    fixed_df = all_res_df[all_res_df["empirical"] == False]
    dfs = {True: empirical_df, False: fixed_df}
    res_list = []
    for exp in dfs:
        num_step_group = dfs[exp].groupby("rw_step_fraction")
        for i, gk in enumerate(num_step_group.groups.keys()):
           rw_step_df = num_step_group.get_group(gk).copy()
           print(gk)
           inf_tree = make_upgma(rw_step_df)
           rfd = inf_tree.compare_rfd(ori_tree, rooted=True)
           res_list.append({"rw_step_fraction": gk, "RF distance": rfd, "empirical": exp})
    res_df = pd.DataFrame(res_list)
    return pd.merge(all_res_df, res_df, on = ["rw_step_fraction", "empirical"])


def plot_RF(all_res_df, also_fixed=False):
    """Plots the RF distance for each value of the step size parameter."""
    if also_fixed:
        plot_df = all_res_df[["RF distance", "empirical", "rw_step_fraction"]].drop_duplicates()
    else:
        plot_df = all_res_df[all_res_df["empirical"] == True][["RF distance", "empirical", "rw_step_fraction"]].drop_duplicates()
    fig, ax = plt.subplots()
    exp_group = plot_df.groupby("empirical")
    for exp, group_df in exp_group:
        ax.plot(group_df["rw_step_fraction"], group_df["RF distance"], label=exp)
    ax.legend()
    fig.tight_layout()
    plt.show()


# pas super utile
def re_vs_mu_range(all_res_df, also_fixed=False):
    """Plots the performance (relative error) against the range of the distribution of mutation rates."""
    if also_fixed:
        plot_df = all_res_df[["muc", "mus", "rw_step_fraction", "nb_steps", "relative_error"]]
    else:
        plot_df = all_res_df[all_res_df["empirical"] == True][["muc", "mus", "rw_step_fraction", "nb_steps", "relative_error"]]

    plot_df["range_mu"] = plot_df["mus"] / plot_df["muc"]
    plot_df["rw_step_fraction"] = plot_df["rw_step_fraction"].astype(float).astype(int)
    min_steps = plot_df["nb_steps"].min()
    max_steps = plot_df["nb_steps"].max()
    num_step_group = plot_df.groupby("rw_step_fraction")
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, step_df in num_step_group:
        sc = ax.scatter(step_df["range_mu"],
                   step_df["relative_error"],
                   c=step_df["nb_steps"],
                   cmap="viridis",
                   vmin=min_steps,
                   vmax=max_steps
                   )
    ax.legend()
    ax.set_xlabel("Mu Distribution Range")
    ax.set_ylabel("Relative Error")
    ax.set_xscale("log")
    fig.colorbar(sc, ax=ax, label='nb_steps')
    fig.tight_layout()
    fig.savefig("error_vs_range_mean.png", dpi=300)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get inference performance for simulation data")
    parser.add_argument("res_csv", help="Path to the csv of all the results")
    args = parser.parse_args()
    res_df, mae_df = collect_metrics(args.res_csv)
    dot_plot_steps(res_df)
