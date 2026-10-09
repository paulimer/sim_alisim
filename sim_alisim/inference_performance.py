#!/usr/bin/env python3

"""Functions to measure and plot the performance of the inference relative to the simulation parameters."""

import argparse
import itertools
from io import StringIO
import os
import shlex
import subprocess as sp
import tempfile as tmp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import sklearn as sk
from skbio import DistanceMatrix, TreeNode
from skbio.tree import upgma
from scipy.stats import pearsonr, spearmanr
import yaml
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import FuncFormatter
import math

from sim_alisim.simulate_infer import fit_mld, load_tips_mut_rate
from sim_alisim.gene_trees import get_time_tree

from mosaic_method.global_fitting_reparam_kappa_inv import (
    precompute_pairs as precompute_pairs_global,
    global_loss,
)


def nb_steps(row):
    return int(((row["sim_tau"] / 2) / row["tree_height"]) * row["rw_step"])


def row_re(row):
    return np.abs((row["sim_tau"] - row["fit_tau"]) / row["sim_tau"])


def collect_metrics(all_res_csv):
    """Reads the results csv and print basic metrics."""
    all_res_df = pd.read_csv(all_res_csv, dtype={"n_steps": str})
    all_res_df["fit_tau"] = all_res_df["fit_tau"].map(lambda x: 10**x)
    all_res_df["sim_tau"] = all_res_df["sim_tau"].map(lambda x: 10**x)
    all_res_df["rw_step"] = all_res_df["n_steps"].map(lambda x: float(x))
    all_res_df["nb_steps"] = all_res_df.apply(nb_steps, axis=1)
    all_res_df["relative_error"] = all_res_df.apply(row_re, axis=1)
    r2_general = sk.metrics.r2_score(all_res_df["sim_tau"], all_res_df["fit_tau"])
    num_step_group = all_res_df.groupby("n_steps")
    r2_step = []
    for gk in num_step_group.groups.keys():
        r2_step += [
            {
                "rw_step": gk,
                "r2": sk.metrics.r2_score(
                    num_step_group.get_group(gk)["sim_tau"],
                    num_step_group.get_group(gk)["fit_tau"],
                ),
            }
        ]
    mae_df = pd.DataFrame(r2_step)
    print(f"General r2: {r2_general}")
    print("r2 by number of step:")
    print(mae_df)
    return all_res_df, mae_df


def compute_fit_tree(res_df, fct_var="n_steps", out_path=None):
    fct_values = list(res_df[fct_var].drop_duplicates())
    fit_trees = {}
    for k, gdf in res_df.groupby(fct_var):
        to_pivot_df = gdf[["genome_1", "genome_2", "fit_tau"]].copy()
        to_pivot_df["fit_tau"] = 10 ** to_pivot_df["fit_tau"]
        fit_tree = make_upgma(to_pivot_df)
        fit_trees[f"{k}"] = fit_tree
        with open(os.path.join(out_path, f"{k}.nwk"), "a") as fo:
            fo.write(str(fit_tree))
    return fit_trees


def conc_exp(
    all_res_df_paths: dict[str, str], keyname: str = "simulation"
) -> pd.DataFrame:
    """
    example: {
            "random_walk": "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/random_tree_kappa_fixedbin/all_res_fit.csv",
            "linear": "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/random_tree_kappa_linear_repro_fig2_fixedbin/all_res_fit.csv"
    }
    """
    all_res_dfs = {k: pd.read_csv(path) for k, path in all_res_df_paths.items()}
    return pd.concat(all_res_dfs, names=[keyname]).reset_index()


def dot_plot_distance(res_df, dist_var="tau", fct_var="n_steps", out_path=None):
    """
    Plots the infered distance against the true distance,
    facets along fct_var.
    """
    fct_values = list(res_df[fct_var].drop_duplicates())
    n = len(fct_values)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        x = 10 ** res_df[f"sim_{dist_var}"].values
        y = 10 ** res_df[f"fit_{dist_var}"].values

        pear = pearsonr(x, y).statistic
        spear = spearmanr(x, y).statistic

        ax.scatter(x, y)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.text(
            0.1,
            0.6,
            f"Pearson r={pear:.3f}\nSpearman ρ={spear:.3f}",
            transform=ax.transAxes,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Simulated distance")
        ax.set_ylabel(f"Inferred distance")
    elif n == 2:

        # def _minor_fmt(x, pos):
        #     for val, label in [(5e7, r"$5\cdot10^7$"), (5e8, r"$5\cdot10^8$")]:
        #         if math.isclose(x, val, rel_tol=1e-9):
        #             return label
        #     return ""

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        custom_colors = ["#6497B1", "#679C35"]
        for color, (k, gdf) in zip(custom_colors, res_df.groupby(fct_var)):
            x = 10 ** gdf[f"sim_{dist_var}"].values
            y = 10 ** gdf[f"fit_{dist_var}"].values

            pear = pearsonr(x, y).statistic
            spear = spearmanr(x, y).statistic

            ax.scatter(x, y, label=f"{k}", color=color)
            # lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            lims = [1e7, 1e9]
            ax.plot(lims, lims, "k--", linewidth=0.8)
            # ax.text(
            #     0.1,
            #     0.6,
            #     f"Pearson r={pear:.3f}\nSpearman ρ={spear:.3f}",
            #     transform=ax.transAxes,
            # )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Simulated distance")
            ax.set_ylabel(f"Inferred distance")
            # for axis in [ax.xaxis, ax.yaxis]:
            # axis.set_minor_formatter(FuncFormatter(_minor_fmt))
            ax.tick_params(axis="both", which="minor", labelsize=7)

    else:
        fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n), squeeze=False)

        for ax, (k, gdf) in zip(axes.flatten(), res_df.groupby(fct_var)):
            x = gdf[f"sim_{dist_var}"].values
            y = gdf[f"fit_{dist_var}"].values

            pear = pearsonr(x, y).statistic
            spear = spearmanr(x, y).statistic

            ax.scatter(x, y)
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, "k--", linewidth=0.8)
            ax.text(
                0.1,
                0.6,
                f"Pearson r={pear:.3f}\nSpearman ρ={spear:.3f}",
                transform=ax.transAxes,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Simulated distance")
            ax.set_ylabel(f"Inferred distance")
            ax.set_title(f"{fct_var} = {k:.1e}")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300)
    else:
        plt.show()
    return fig


def dot_plot_steps(all_res_df):
    """Plots the infered distance against the true distance"""
    min_steps = all_res_df["nb_steps"].min()
    max_steps = all_res_df["nb_steps"].max()
    min_tau = min([all_res_df["sim_tau"].min(), all_res_df["fit_tau"].min()])
    max_tau = max([all_res_df["sim_tau"].max(), all_res_df["fit_tau"].max()])
    g = sns.FacetGrid(data=all_res_df, col="n_steps", col_wrap=4)  # , hue="empirical")
    g.map_dataframe(
        sns.scatterplot, x="sim_tau", y="fit_tau"
    )  # , c="nb_steps")#, cmap="viridis", vmin=min_steps, vmax=max_steps, alpha=0.5)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    g.add_legend()
    g.savefig("dotplot_by_steps_mean.png", dpi=300)


def plot_one_exp_mus(ax: plt.Axes, rates, pair, fit_muc, **kwargs):
    """Plots the distribution of simulated mutation for a given pair"""
    bins = np.geomspace(min(rates), max(rates), 30)
    ax.hist(rates, bins, label="experimental rates", color="tab:blue")
    # ax.axvline(x=min(rates), label="Minimum rate", linestyle="--")
    # ax.axvline(x=max(rates), label="Maximum rate", linestyle="--")
    ax.axvline(
        x=fit_muc, label="Fitted minimum rate", linestyle="-", color="tab:orange"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"pair: {pair}, conf: {kwargs}")
    ax.legend()


def distr_exp_mu(
    res_df: pd.DataFrame, sim_dir, evolution_param="n_steps", out_path="."
):
    """
    Plots:
    - the distributions of simulated substitution rates,
    - the inferred range and,
    - the empirical range.

    Limits: supposes 1 aligner.
    """
    with open(os.path.join(sim_dir, "sim_config.yaml")) as f:
        sim_conf = yaml.safe_load(f)
    input_tree = TreeNode.read([sim_conf["species_tree"]])
    if sim_conf["rate_evolution"] == "none":
        # 4.00e+08___none_none_genomes
        evolution_param_value = {"none": "none"}
        nconf = len(sim_conf["tree_height"])
    else:
        evolution_param_value = {float(s): s for s in sim_conf[evolution_param]}
        nconf = len(sim_conf[evolution_param]) * len(sim_conf["tree_height"])
    ntips = len(list(input_tree.tips()))
    npairs = ntips * (ntips - 1) // 2
    fig, axes2d = plt.subplots(
        nconf, npairs, squeeze=False, figsize=(5 * npairs, 4 * nconf)
    )
    for ((th, rss, aligner), treedf), axes1d in zip(
        res_df.groupby(["tree_height", evolution_param, "aligner"]), axes2d
    ):
        time_tree = get_time_tree(input_tree, th)
        genomes_path = os.path.join(
            sim_dir,
            "genomes",
            f"{th:.2e}___{evolution_param}_{evolution_param_value[rss]}_genomes",
        )
        tips_mut_rate = load_tips_mut_rate(genomes_path, time_tree)
        tips_avg_rates = {
            key: [(a + b) / 2 for a, b in value] for key, value in tips_mut_rate.items()
        }
        for (pair, row), ax in zip(treedf.groupby(["genome_1", "genome_2"]), axes1d):
            subs_rates = tips_avg_rates[pair]
            plot_one_exp_mus(
                ax,
                subs_rates,
                pair,
                row["fitted_muc"].iloc[0],
                rss=rss,
                th=th,
            )
    fig.tight_layout()
    fig.supxlabel("Mutation rate")
    fig.supylabel("Density")
    fig.savefig(out_path)


def fitted_vs_exp_muc(res_df: pd.DataFrame):
    "plots the prediction error of fitted muc by the inference"
    res_df["muc relative error"] = (
        res_df["fitted_muc"] - res_df["empirical_muc"]
    ) / res_df["empirical_muc"]
    g = sns.catplot(
        data=res_df,
        x="n_steps",
        y="muc relative error",
        col="tree_height",
        col_wrap=4,
        kind="bar",
        native_scale=True,
        log_scale=(True, False),
    )
    g.savefig(
        "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/rw_lin_pair_stsi/fitted_muc_error.png",
        dpi=300,
    )


def exp_kappa(res_df: pd.DataFrame):
    "plots the prediction error of fitted muc by the inference"
    res_df["exp_kappa"] = res_df["empirical_muc"] / res_df["empirical_mus"]
    lims = (
        (res_df["exp_kappa"].min(), res_df["kappa"].min()),
        (res_df["exp_kappa"].max(), res_df["kappa"].max()),
    )

    def _xyline(data, **kws):
        ax = plt.gca()
        ax.plot(lims, lims, "k--", linewidth=0.8)

    g = sns.relplot(
        data=res_df,
        x="exp_kappa",
        y="kappa",
        col="tree_height",
        hue="n_steps",
        kind="scatter",
        col_wrap=4,
    )
    g.map_dataframe(_xyline)
    g.savefig(
        "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/rw_lin_pair_stsi/kappa_exp_kappa.png",
        dpi=300,
    )


def violin_steps(all_res_df):
    """Plots MAE for each step number."""
    g = sns.FacetGrid(all_res_df, col="n_steps", col_wrap=4)
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
    all_dirs_data = [
        os.path.join(data_path, d) for d in os.listdir(data_path) if d.startswith("rw")
    ]
    res_list_w_mash = []
    for d in all_dirs_data:
        n_steps = float(os.path.split(d)[1][-5:])
        d_df = all_res_df[all_res_df["n_steps"] == n_steps]
        genome_files = [
            os.path.join(d, g) for g in os.listdir(d) if g.endswith("fasta")
        ]
        mash_cmd_str = f"mash triangle -p 10 -s 100000 -E {' '.join(genome_files)}"
        mash_cmd = shlex.split(mash_cmd_str)
        mash_res = sp.run(mash_cmd, capture_output=True, check=True, encoding="utf-8")

        mash_df = pd.read_csv(
            StringIO(mash_res.stdout),
            sep="\t",
            header=None,
            names=["genome_1", "genome_2", "mash_dist", "pval", "shared_hashes"],
        )
        mash_df["genome_1"] = mash_df["genome_1"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        mash_df["genome_2"] = mash_df["genome_2"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        mash_df[["genome_1", "genome_2"]] = mash_df.apply(
            lambda x: (
                (x.genome_1, x.genome_2)
                if x.genome_1 < x.genome_2
                else (x.genome_2, x.genome_1)
            ),
            axis=1,
            result_type="expand",
        )
        res_list_w_mash.append(
            pd.merge(d_df, mash_df, "inner", on=["genome_1", "genome_2"])
        )

    res = pd.concat(res_list_w_mash)
    return res


# ok mash is better
def mash_vs_mosaic(res_df, also_fixed=False):
    """Plots the mash distance against the mosaic distance."""
    if also_fixed:
        plot_df = res_df[["mash_dist", "n_steps", "sim_tau", "fit_tau"]]
    else:
        plot_df = res_df[res_df["empirical"] == True][
            ["mash_dist", "n_steps", "sim_tau", "fit_tau"]
        ].drop_duplicates()
    scaler = sk.preprocessing.MinMaxScaler()
    scaled = scaler.fit_transform(plot_df[["mash_dist", "fit_tau", "sim_tau"]])
    plot_df[["mash_dist", "fit_tau", "sim_tau"]] = scaled
    plot_df = plot_df.melt(
        id_vars=["n_steps", "sim_tau"],
        value_vars=["fit_tau", "mash_dist"],
        value_name="normalized estimated distance",
        var_name="method",
    )
    g = sns.FacetGrid(data=plot_df, col="n_steps", col_wrap=4)  # , hue="empirical")
    g.map_dataframe(
        sns.scatterplot, x="sim_tau", y="normalized estimated distance", hue="method"
    )  # , c="nb_steps")#, cmap="viridis", vmin=min_steps, vmax=max_steps, alpha=0.5)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    g.add_legend()
    plt.show()


def refit_fixed_mus(all_res_df, res_path, muc, mus, delta, genome_length):
    """Refits the MLDs but with a fixed, assumed mu distribution"""
    # /home/paulimer/Data/simulated_datasets/sim_alisim_data/entero_sim_find_steps_cor_5k
    res_dirs = [
        os.path.join(res_path, dirp)
        for dirp in os.listdir(res_path)
        if dirp.startswith("res")
    ]
    refit_list = []
    for res_dir in res_dirs:
        n_steps = res_dir[-5:]
        binned_mlds = os.path.join(res_dir, "binned_mlds")
        mlds_csv = [csv for csv in os.listdir(binned_mlds) if csv.endswith(".csv")]
        for mld_csv in mlds_csv:
            mld_df = pd.read_csv(os.path.join(binned_mlds, mld_csv))
            cur_comp = os.path.splitext(mld_csv)[0].split("_")
            _, res_opt, cur_comp = fit_mld(
                mld_df, muc, mus, delta, genome_length, cur_comp, True
            )
            fit_res = {
                "genome_1": cur_comp[0],
                "genome_2": cur_comp[1],
                "fit_tau": 10 ** res_opt.x[0],
                "muc": muc,
                "mus": mus,
                "n_steps": float(n_steps),
            }
            refit_list.append(fit_res)

    refit_df = pd.DataFrame(refit_list)
    refit_df["n_steps"] = refit_df["n_steps"].astype(str)
    all_res_df["n_steps"] = all_res_df["n_steps"].astype(float).astype(str)
    tmp_df = pd.merge(
        all_res_df.drop(["fit_tau", "empirical_muc", "empirical_mus"], axis=1),
        refit_df,
        on=("genome_1", "genome_2", "n_steps"),
    )
    all_res_df.rename(
        {"empirical_muc": "muc", "empirical_mus": "mus"}, inplace=True, axis=1
    )
    all_res_df["n_steps"] = all_res_df["n_steps"].astype(float)
    tmp_df["n_steps"] = tmp_df["n_steps"].astype(float)
    return (
        pd.concat({True: all_res_df, False: tmp_df})
        .reset_index()
        .drop("level_1", axis=1)
        .rename({"level_0": "empirical"}, axis=1)
    )


def fit_refit_distance(all_res_df):
    """Compares fits with correct and incorrect mus across the range of distances."""
    all_res_df["n_steps"] = all_res_df["n_steps"].astype(float).astype(int)
    all_res_df["relative_error"] = all_res_df.apply(
        lambda x: np.abs((x["sim_tau"] - x["fit_tau"]) / x["sim_tau"]), axis=1
    )
    num_step_group = all_res_df.groupby("n_steps")
    # fig, ax = plt.subplot
    # for i, gk in enumerate(num_step_group.groups.keys()):
    #     rw_step_df = num_step_group.get_group(gk).copy()
    #     ax.scatter(rw_step_df["sim_tau"], rw_step_df["relative_error"]
    #     ax.scatter(rw_step_df["sim_tau"], rw_step_df["refit_relative_error"]
    g = sns.FacetGrid(all_res_df, col="n_steps", col_wrap=4)
    g.map(sns.boxplot, "empirical", "relative_error")
    g.savefig("fit_refit_distance_mean.png", dpi=300)


def make_upgma(res_df):
    tau_df = res_df[["genome_1", "genome_2", "fit_tau"]]
    tau_matrix = (
        pd.DataFrame(np.concatenate([tau_df.values, tau_df.values[:, [1, 0, 2]]]))
        .pivot(columns=0, index=1, values=2)
        .fillna(0)
    )
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
        num_step_group = dfs[exp].groupby("n_steps")
        for i, gk in enumerate(num_step_group.groups.keys()):
            rw_step_df = num_step_group.get_group(gk).copy()
            print(gk)
            inf_tree = make_upgma(rw_step_df)
            rfd = inf_tree.compare_rfd(ori_tree, rooted=True)
            res_list.append({"n_steps": gk, "RF distance": rfd, "empirical": exp})
    res_df = pd.DataFrame(res_list)
    return pd.merge(all_res_df, res_df, on=["n_steps", "empirical"])


def plot_RF(all_res_df, also_fixed=False):
    """Plots the RF distance for each value of the step size parameter."""
    if also_fixed:
        plot_df = all_res_df[["RF distance", "empirical", "n_steps"]].drop_duplicates()
    else:
        plot_df = all_res_df[all_res_df["empirical"] == True][
            ["RF distance", "empirical", "n_steps"]
        ].drop_duplicates()
    fig, ax = plt.subplots()
    exp_group = plot_df.groupby("empirical")
    for exp, group_df in exp_group:
        ax.plot(group_df["n_steps"], group_df["RF distance"], label=exp)
    ax.legend()
    fig.tight_layout()
    plt.show()


# pas super utile
def re_vs_mu_range(all_res_df, also_fixed=False):
    """Plots the performance (relative error) against the range of the distribution of mutation rates."""
    if also_fixed:
        plot_df = all_res_df[["muc", "mus", "n_steps", "nb_steps", "relative_error"]]
    else:
        plot_df = all_res_df[all_res_df["empirical"] == True][
            ["muc", "mus", "n_steps", "nb_steps", "relative_error"]
        ]

    plot_df["range_mu"] = plot_df["mus"] / plot_df["muc"]
    plot_df["n_steps"] = plot_df["n_steps"].astype(float).astype(int)
    min_steps = plot_df["nb_steps"].min()
    max_steps = plot_df["nb_steps"].max()
    num_step_group = plot_df.groupby("n_steps")
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, step_df in num_step_group:
        sc = ax.scatter(
            step_df["range_mu"],
            step_df["relative_error"],
            c=step_df["nb_steps"],
            cmap="viridis",
            vmin=min_steps,
            vmax=max_steps,
        )
    ax.legend()
    ax.set_xlabel("Mu Distribution Range")
    ax.set_ylabel("Relative Error")
    ax.set_xscale("log")
    fig.colorbar(sc, ax=ax, label="nb_steps")
    fig.tight_layout()
    fig.savefig("error_vs_range_mean.png", dpi=300)


# global
def global_obj_perf(
    all_res_df: pd.DataFrame,
    all_binned_mlds: pd.DataFrame,
    gene_length: int,
    out_path: str,
):
    """Evaluates whether the objective function can discriminate parameters of the simulation from the result of inference"""
    row_key = "tree_height"
    col_key = "n_gene_trees"
    conf_keys = [row_key] + [col_key]
    all_res_df["sim_theta"] = (10 ** all_res_df["sim_tau"]) * all_res_df[
        "empirical_mus"
    ]
    binned_mld_dic_of_dic = {}
    for conf, conf_mlds in all_binned_mlds.groupby(conf_keys):
        binned_mld_conf = {}
        for pair, pair_mld in conf_mlds.groupby(["species_1", "species_2"]):
            binned_mld_conf[pair] = pair_mld
        binned_mld_dic_of_dic[conf] = binned_mld_conf

    pair_list = list(next(iter(binned_mld_dic_of_dic.values())).keys())

    ids_exp = list(binned_mld_dic_of_dic.keys())
    row_vals = sorted(set([rid for rid, _ in ids_exp]))
    col_vals = sorted(set([cid for _, cid in ids_exp]))

    fig = plt.Figure((5 * len(col_vals), len(row_vals) * 5))
    axes = fig.subplots(len(row_vals), len(col_vals))
    pad = 5

    if len(ids_exp) == 1:
        axes = [axes]
    for i, rid in enumerate(row_vals):
        for j, cid in enumerate(col_vals):
            # conf_dic = {conf_param: value for conf_param, value in zip(conf_keys, id_exp)}
            if i == 0:
                # annotate columns
                axes[i, j].annotate(
                    f"{cid} trees",
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    size="large",
                    ha="center",
                    va="baseline",
                )
            if j == 0:
                # annotate rows
                axes[i, j].annotate(
                    f"Distance: {rid:.2e}",
                    xy=(0, 0.5),
                    xytext=(-axes[i, j].yaxis.labelpad - pad, 0),
                    xycoords=axes[i, j].yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                )

            conf_dic = {row_key: rid, col_key: cid}
            L0_df_sim = pd.DataFrame(
                {
                    "bac1": [p[0] for p in pair_list],
                    "bac2": [p[1] for p in pair_list],
                    "L0": float(conf_dic["n_gene_trees"]) * gene_length,
                }
            )
            query = " & ".join([f"({k} == {v})" for k, v in conf_dic.items()])
            exp_res_df = all_res_df.query(query)

            precomputed = precompute_pairs_global(
                binned_mld_dic_of_dic[(rid, cid)], L0_df_sim
            )

            # global parameters
            kappa_sim = (
                exp_res_df["empirical_mus"].values[0]
                / exp_res_df["empirical_muc"].values[0]
            )
            kappa_fit = exp_res_df["kappa"].values[0]
            print("========================================")
            print(f"Sim conf: {conf_dic}")
            print(f"Kappa sim: {kappa_sim}    ; Kappa fit: {kappa_fit}")
            print("========================================")

            # pairwise parameters
            theta_logxi_sim = [[ts, -300] for ts in exp_res_df["sim_theta"].values]
            theta_xi_fit = exp_res_df[["theta", "xi"]].values.tolist()
            theta_logxi_fit = [(theta, np.log10(xi)) for theta, xi in theta_xi_fit]

            # for j in range(len(precomputed[0])):
            #     print(
            #         f"{precomputed[0][j][0]} {precomputed[0][j][1]}: theta_sim: {theta_logxi_sim[j][0]} theta_fit {theta_logxi_fit[j][0]}"
            #     )

            kappa_vals = np.linspace(1.1, 30, 100)

            params_fit = [
                [kp] + list(itertools.chain(*theta_logxi_fit)) for kp in kappa_vals
            ]
            params_sim = [
                [kp] + list(itertools.chain(*theta_logxi_sim)) for kp in kappa_vals
            ]
            kappa_obj_fit = []
            kappa_obj_param = []
            for fit_opt, param_opt in zip(params_fit, params_sim):
                kappa_obj_fit.append(
                    global_loss(np.array(fit_opt), precomputed, delta=3)
                )
                kappa_obj_param.append(
                    global_loss(np.array(param_opt), precomputed, delta=3)
                )
            axes[i, j].plot(kappa_vals, kappa_obj_fit, label="Estimate")
            axes[i, j].plot(kappa_vals, kappa_obj_param, label="True")
            axes[i, j].axvline(
                kappa_fit,
                label=r"$\hat{\kappa}$",
                ls="--",
            )
            axes[i, j].axvline(
                kappa_sim,
                label=r"$\kappa_{sim}$",
            )
            axes[i, j].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)


# poissonnoseq plots
def kappa_perf(all_res_df, abciss_var, facet_var, out_path):
    all_res_df["kappa_sim"] = all_res_df["empirical_mus"] / all_res_df["empirical_muc"]
    all_res_df["kappa_relerr"] = (
        np.absolute(all_res_df["kappa_sim"] - all_res_df["kappa"])
        / all_res_df["kappa_sim"]
    )
    plot = (
        so.Plot(all_res_df, x=abciss_var, y="kappa_relerr")
        .add(so.Dot())
        .facet(col=facet_var, wrap=3)
        .scale(x="log")
    )
    plot.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get inference performance for simulation data"
    )
    parser.add_argument("res_csv", help="Path to the csv of all the results")
    args = parser.parse_args()
    res_df, mae_df = collect_metrics(args.res_csv)
    dot_plot_steps(res_df)

    if False:
        for di in [
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/random_tree_linear_mu_kappa_20k",
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/random_tree_linear_mu_kappa_10k",
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/random_tree_linear_mu_kappa",
        ]:
            res_df = pd.read_csv(os.path.join(di, "all_res_fit.csv"))
            distr_exp_mu(res_df, di, "none", os.path.join(di, "distr_exp_mus.png"))
        all_res_df = pd.read_csv(
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/rw_lin_pair_stsi/all_res_fit.csv"
        )
        null_res_df = pd.read_csv(
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/null_lin_pair_stsi/all_res_fit.csv"
        )
        res_df = pd.concat([all_res_df, null_res_df], keys=("rw", "null"))
        res_df.loc[res_df["n_steps"].isna(), "n_steps"] = 0

        # theta (same thing as tau)
        res_df["fit_theta"] = res_df["theta"]
        res_df["sim_theta"] = 10 ** res_df["sim_tau"] * res_df["empirical_mus"]
        f = dot_plot_distance(
            res_df, "theta", "n_steps", "/home/paulimer/Downloads/test.png"
        )
