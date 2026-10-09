#!/usr/bin/env python3

import itertools
from collections import Counter, defaultdict
import os
import sys
import yaml

from mosaic_method.residuals_plot import calc_residuals, plot_resid, chi_square
from sim_alisim.simulate_infer import to_list, RATE_EVOLUTION_DIC
from sim_alisim.theoretical_vs_simulated import ALIGNER_DELTA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
import numpy as np


def get_residuals_setting(res_dir, mld_path, genomes_dir, filters, L0, inf_config):
    """Returns the bdl_mld with the residuals for a given setting of a given simulation."""
    all_res_df = pd.read_csv(os.path.join(res_dir, "all_res_fit.csv"))
    mask = pd.Series(True, index=all_res_df.index)
    for col, val in filters.items():
        try:
            val = float(val)
        except (ValueError, TypeError):
            pass
        mask &= all_res_df[col] == val
    subres_df = all_res_df[mask].copy()
    comparisons = subres_df[["genome_1", "genome_2"]]
    L0_df = comparisons[["genome_1", "genome_2"]].copy()
    subres_df.rename(
        {"genome_1": "species_1", "genome_2": "species_2"}, inplace=True, axis=1
    )
    L0_df["L0"] = L0
    L0_df.rename({"genome_1": "bac1", "genome_2": "bac2"}, inplace=True, axis=1)
    # TODO careful, to iterate maybe
    inf_config["delta"] = ALIGNER_DELTA[inf_config["aligner"][0]]
    with open("tmp.yaml", "w") as of:
        yaml.dump(inf_config, of)

    bdl_mld = calc_residuals(
        mld_path,
        pd.read_csv(os.path.join(genomes_dir, "taxon.csv")),
        subres_df,
        "tmp.yaml",
        L0_df,
    )
    os.remove("tmp.yaml")
    return bdl_mld


def get_residuals(sim_cfg, inf_cfg):
    """Returns the residuals_mld for each configuration of a simulation experiment."""
    tree_heights = to_list(sim_cfg["tree_height"])
    tree_heights = [float(th) for th in tree_heights]
    rate_evolution_parameter = RATE_EVOLUTION_DIC[sim_cfg["rate_evolution"]]
    aligners = to_list(inf_cfg["aligner"])
    if rate_evolution_parameter == "none" or rate_evolution_parameter == "null":
        rate_params = ["none"]
    else:
        rate_params = sim_cfg[rate_evolution_parameter].copy()

    configs = [
        {"tree_height": th, "aligner": al, rate_evolution_parameter: rep}
        for th, al, rep in itertools.product(tree_heights, aligners, rate_params)
    ]
    file_names = [
        {
            "db_name": f"{sim_cfg['outdir']}/dbs/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}.db",
            "fit_expected_name": f"{sim_cfg['outdir']}/fit_expected_fig/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_fit_expected.png",
            "genomes_dir": f"{sim_cfg['outdir']}/genomes/{conf['tree_height']:.2e}___{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_genomes/",
        }
        for conf in configs
    ]
    residuals_mlds = {}
    L0 = sim_cfg["n_gene_trees"] * sim_cfg["length_gene"]
    for conf, names in zip(configs, file_names):
        key = tuple([(key, item) for key, item in conf.items()])
        # technically I should not rebin here
        residuals_mlds[key] = get_residuals_setting(
            sim_cfg["outdir"],
            names["db_name"],
            names["genomes_dir"],
            {
                "tree_height": conf["tree_height"],
                rate_evolution_parameter: conf[rate_evolution_parameter],
            },
            L0,
            inf_cfg,
        )
    return residuals_mlds


def get_exps_residuals(exp_dics):
    all_exp_df_dic = {}
    for exp, (sim_cfg_f, inf_cfg_f) in exp_dics.items():
        with open(sim_cfg_f, "r") as f:
            sim_cfg = yaml.safe_load(f)
        with open(inf_cfg_f, "r") as f:
            inf_cfg = yaml.safe_load(f)
        res_mld_dic = get_residuals(sim_cfg, inf_cfg)
        res_mld_df = (
            pd.concat(
                list(res_mld_dic.values()), keys=[str(k) for k in res_mld_dic.keys()]
            )
            .reset_index(names=["exp", "drop"])
            .drop("drop", axis=1)
        )
        all_exp_df_dic[exp] = res_mld_df
    return all_exp_df_dic


def parameter_exp(exp_dics):
    res = {}
    for exp, (sim_cfg_f, _) in exp_dics.items():
        with open(sim_cfg_f, "r") as f:
            sim_cfg = yaml.safe_load(f)
        res[exp] = RATE_EVOLUTION_DIC[sim_cfg["rate_evolution"]]
    return res


def plot_residuals(exp_dics, outdir="."):
    all_exp_df_dic = get_exps_residuals(exp_dics)
    # actually I should first iterate on tree_height then on the experiment
    exp_tree_heights = []
    for exp in all_exp_df_dic:
        sim_cfg_f, _ = exp_dics[exp]
        with open(sim_cfg_f, "r") as f:
            sim_cfg = yaml.safe_load(f)
        exp_tree_heights.append(tuple(sim_cfg["tree_height"]))
    if not len(set(exp_tree_heights)) == 1:
        print("Not all experiments have the same tree height values")
        print(exp_tree_heights)
        raise ValueError

    all_exp_df = (
        pd.concat(all_exp_df_dic)
        .reset_index(names=["general_exp", "drop"])
        .drop("drop", axis=1)
    )
    param_exp = parameter_exp(exp_dics)
    all_exp_df["Evolution parameter"] = all_exp_df.apply(
        lambda x: x[param_exp[x["general_exp"]]], axis=1
    )
    all_exp_df["tree_height_fmt"] = all_exp_df["tree_height"].apply(
        lambda x: f"{x:.2e}"
    )
    wrap = 7
    os.makedirs(outdir, exist_ok=True)
    ordered_heights = [f"{h:.2e}" for h in sorted(all_exp_df["tree_height"].unique())]
    resid_plot = (
        so.Plot(
            all_exp_df, x="match_length", y="residuals", color="Evolution parameter"
        )
        .add(so.Dot())
        .facet(col="tree_height_fmt", order=ordered_heights, wrap=wrap)
        .scale(color=so.Nominal(), x=so.Continuous(trans="log"))
        .layout(size=(20, 7))
    )
    resid_plot.save(
        os.path.join(outdir, "residuals_all_heights_12.png"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_mlds(exp_dics, outdir="."):
    all_exp_df_dic = get_exps_residuals(exp_dics)
    # actually I should first iterate on tree_height then on the experiment
    exp_tree_heights = []
    for exp in all_exp_df_dic:
        sim_cfg_f, _ = exp_dics[exp]
        with open(sim_cfg_f, "r") as f:
            sim_cfg = yaml.safe_load(f)
        exp_tree_heights.append(tuple(sim_cfg["tree_height"]))
    if not len(set(exp_tree_heights)) == 1:
        print("Not all experiments have the same tree height values")
        print(exp_tree_heights)
        raise ValueError

    all_exp_df = (
        pd.concat(all_exp_df_dic)
        .reset_index(names=["general_exp", "drop"])
        .drop("drop", axis=1)
    )
    param_exp = parameter_exp(exp_dics)
    all_exp_df["Evolution parameter"] = all_exp_df.apply(
        lambda x: x[param_exp[x["general_exp"]]], axis=1
    )
    all_exp_df["tree_height_fmt"] = all_exp_df["tree_height"].apply(
        lambda x: f"{x:.2e}"
    )
    wrap = 4
    os.makedirs(outdir, exist_ok=True)
    ordered_heights = [f"{h:.2e}" for h in sorted(all_exp_df["tree_height"].unique())]
    th_df = all_exp_df[["match_length", "th_freq", "tree_height_fmt"]].drop_duplicates(
        subset=["match_length", "tree_height_fmt"]
    )
    mld_plot = (
        so.Plot(all_exp_df, x="match_length", y="freq", color="Evolution parameter")
        .add(mark=so.Dot())
        .add(so.Line(color="black"), data=th_df, y="th_freq")
        .facet(col="tree_height_fmt", order=ordered_heights, wrap=wrap)
        .scale(
            color=so.Nominal(),
            x=so.Continuous(trans="log"),
            y=so.Continuous(trans="log"),
        )
        .layout(size=(14, 7 * wrap))
    )
    mld_plot.save(
        os.path.join(outdir, "mlds_all_heights.png"), dpi=300, bbox_inches="tight"
    )


def get_chis(exp_dics):
    """Returns the summed residuals for each configuration of each simulation experiment."""
    all_chi_dic = {}
    all_exp_df_dic = get_exps_residuals(exp_dics)
    for exp, exp_df in all_exp_df_dic.items():
        all_chi_dic[exp] = chi_square(exp_df)
    all_chi_df = pd.concat(all_chi_dic)
    return all_chi_df


def plot_chis(all_chi_df):
    all_chis = all_chi_df.reset_index(names=["global_exp", "drop"]).drop(
        ["drop"], axis=1
    )
    all_chis[["tree_h", "al", "exp_setting"]] = all_chis.exp.str.split(
        "\),", expand=True
    )
    all_chis["tree_h"] = all_chis.tree_h.str.split(",", expand=True).iloc[:, 1]
    all_chis["tree_h"] = all_chis["tree_h"].astype(float)
    all_chis = all_chis.drop("al", axis=1)
    all_chis[["evo_param", "value"]] = all_chis.exp_setting.str.split(",", expand=True)
    all_chis["evo_param"] = all_chis.evo_param.str.extract(r"\('(.*)'")
    all_chis["value"] = all_chis.value.str.extract(r"'(.*)'")
    all_chis["chi2_norm"] = all_chis["chi2"] / all_chis["nbins"]

    theme_dict = {**axes_style("whitegrid"), "grid.linestyle": ":"}
    g = (
        so.Plot(all_chis, x="tree_h", y="chi2_norm", color="value")
        .add(so.Dot(pointsize=10))
        .scale(x=so.Continuous(trans="log"))
        .theme(theme_dict)
    )

    return g


if __name__ == "__main__":
    exp_dic = {
        "null": [
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_null/entero_sim_config_null.yaml",
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_null/entero_inf_config.yaml",
        ],
        "random_walk": [
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_rw/entero_sim_config.yaml",
            "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_rw/entero_inf_config.yaml",
        ],
    }
    all_chis = get_chis(exp_dic)
    all_chis = all_chis.reset_index(names=["global_exp", "drop"]).drop(["drop"], axis=1)
    all_chis["chi2_norm"] = all_chis["chi2"] / all_chis["nbins"]
    sns.violinplot(all_chis, x="exp", y="chi2_norm")
    plt.show()
