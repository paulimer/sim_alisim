#!/usr/bin/env python3

import itertools
import os
import sys
import yaml

from mosaic_method.residuals_plot import calc_residuals, plot_resid, chi_square
from sim_alisim.simulate_infer import to_list, RATE_EVOLUTION_DIC

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def get_residuals_setting(res_dir, mld_path, genomes_dir, setting, L0, inf_config):
    """Returns the bdl_mld with the residuals for a given setting of a given simulation."""
    all_res_df = pd.read_csv(os.path.join(res_dir, "all_res_fit.csv"))
    try:
        setting = (setting[0], float(setting[1]))
    except ValueError:
        pass
    subres_df = all_res_df[all_res_df[setting[0]] == setting[1]].copy()
    comparisons = subres_df[["genome_1", "genome_2"]]
    L0_df = comparisons[["genome_1", "genome_2"]].copy()
    subres_df.rename({"genome_1": "species_1", "genome_2": "species_2"}, inplace=True, axis=1)
    L0_df["L0"] = L0
    L0_df.rename({"genome_1": "bac1", "genome_2": "bac2"}, inplace=True, axis=1)
    with open("tmp.yaml", "w") as of:
        yaml.dump(inf_config, of)

    bdl_mld = calc_residuals(
        mld_path,
        os.path.join(genomes_dir, "taxon.csv"),
        subres_df,
        L0_df,
        "tmp.yaml" 
    )
    os.remove("tmp.yaml")
    # not interested in binning by 3 here
    return bdl_mld[bdl_mld["type"] == "summed"]


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

    configs = [{"tree_height": th, "aligner": al, rate_evolution_parameter: rep} \
               for th, al, rep in itertools.product(tree_heights, aligners, rate_params)]
    file_names = [{"db_name":f"{sim_cfg['outdir']}/dbs/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}.db",\
                   "binned_mld_name":f"{sim_cfg['outdir']}/binned_mlds/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_binned_mld/",\
                   "fit_expected_name":f"{sim_cfg['outdir']}/fit_expected_fig/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_fit_expected.png",\
                   "genomes_dir":f"{sim_cfg['outdir']}/genomes/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_genomes/"} \
                for conf in configs]
    residuals_mlds = {}
    L0 = sim_cfg["n_gene_trees"] * sim_cfg["length_gene"]
    for conf, names in zip(configs, file_names):
        key = tuple([(key, item) for key, item in conf.items()])
        # technically I should not rebin here
        residuals_mlds[key] = get_residuals_setting(
            sim_cfg["outdir"],
            names["db_name"],
            names["genomes_dir"],
            (rate_evolution_parameter, conf[rate_evolution_parameter]),
            L0,
            inf_cfg
        )
    return residuals_mlds


def get_chis(exp_dic):
    """Returns the summed residuals for each configuration of each simulation experiment."""
    all_chi_dic = {}
    for exp, (sim_cfg_f, inf_cfg_f) in exp_dic.items():
        with open(sim_cfg_f, "r") as f:
            sim_cfg = yaml.safe_load(f)
        with open(inf_cfg_f, "r") as f:
            inf_cfg = yaml.safe_load(f)
        res_mld_dic = get_residuals(sim_cfg, inf_cfg)
        all_chi_dic[exp] = chi_square(res_mld_dic)
    all_chi_df = pd.concat(all_chi_dic)
    return all_chi_df


if __name__ == "__main__":
    exp_dic = {
        "null" : ["/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_null/entero_sim_config_null.yaml", "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_null/entero_inf_config.yaml"],
        "random_walk" : ["/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_rw/entero_sim_config.yaml", "/home/paulimer/Documents/results_bacteria_mlds/simulated_results/entero_sim_rw/entero_inf_config.yaml"]
    }
    all_chis = get_chis(exp_dic)
    all_chis = all_chis.reset_index(names=["global_exp", "drop"]).drop(["drop"], axis=1)
    all_chis["chi2_norm"] = all_chis["chi2"] / all_chis["nbins"]
    sns.violinplot(all_chis, x="exp", y="chi2_norm")
    plt.show()

