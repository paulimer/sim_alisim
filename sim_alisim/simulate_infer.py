#!/usr/bin/env python3
import shutil

import argparse
import concurrent.futures
import cProfile
import itertools
import os
from pathlib import Path
import pstats
import sys
import time
import tracemalloc

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import TreeNode
import yaml

RATE_EVOLUTION_DIC = {
    "random_walk": "rw_step_fraction",
    "kishino": "nu",
    "none": "none",
    "null": "none"
}

from sim_alisim.gene_trees import get_time_tree, run_simulation, get_all_pair_mutation_rate, to_list, generate_gene_trees, plot_distance_distribution, plot_pseudo_empirical
from sim_alisim.theoretical_vs_simulated import ALIGNER_DELTA
from mosaic_method.fitting import theoretical_mld, fit_params
from mosaic_method.parsing import get_genome_comp, get_all_mlds, sum_mlds, bin_mld
from mosaic_method.aligning import align_exec, create_lastz_db


# TODO: concatenate all binned mld in a single df/csv, all the db in a single db, etc
#
def inf_add_suffix(inference_cfg, suffix):
    for key in inference_cfg:
        if key.endswith("dir"):
            inference_cfg[key] = inference_cfg[key] + suffix


def extract_pairwise_cherries(simulation_cfg):
    """Extracts the different pairwise "cherries" of a given tree, to simulate and infer a baseline."""
    time_tree = TreeNode.read([simulation_cfg["species_tree"]])
    pairs = itertools.combinations([tip.name for tip in time_tree.tips()], 2)
    cherries = []
    for pair in pairs:
        cherries.append(time_tree.shear(pair))
    return cherries


def make_cherry_conf(simulation_cfg, inference_cfg, cherry, names):
    """Create a conf dic for running "control" cherries to compare to the baseline."""
    cherry_sim_conf = simulation_cfg.copy()
    cherry_sim_conf["species_tree"] = str(cherry)
    cherry_sim_conf["outdir"] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    cherry_sim_conf["rate_evolution"] = "random_walk"
    cherry_sim_conf["tree_height"] = cherry.height()[0] / TreeNode.read([simulation_cfg["species_tree"]]).height()[0] * simulation_cfg["tree_height"]
    cherry_inf_conf = inference_cfg.copy()
    cherry_inf_conf["genomes_dir"] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    cherry_inf_conf["results_dir"] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    # print(cherry_sim_conf)
    return cherry_sim_conf, cherry_inf_conf


def cherries_inference(cfg, names):
    """Just get me the binned mld bro."""
    align_res = align_exec([cfg["genomes_dir"] + f"/{names[0]}.fasta", cfg["genomes_dir"] + f"/{names[1]}.fasta"], "lastz", "bla")
    unbinned_mld = align_res[2]["result"][0]
    summed_mld = pd.DataFrame({"freq":unbinned_mld}, index=range(1, len(unbinned_mld) + 1)).reset_index(names="match_length")
    binned_mld = bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=1
        )
    return binned_mld



def simplified_parsing(cfg, cherry_inf=False):
    """
    Aligns, gathers MLDs and bins them.
    """

    if not os.path.exists(cfg["taxon_csv"]):
        cfg["taxon_csv"] = os.path.join(cfg["genomes_dir"], cfg["taxon_csv"])
        if not os.path.exists(cfg["taxon_csv"]):
            print(f"Taxon csv not found at {cfg['taxon_csv']}")
            sys.exit(1)

    masked_genomes_dir = cfg["genomes_dir"]
    # alignment
    print("Aligning genomes")
    database_path = cfg["database_name"]
    if os.path.exists(database_path):
        update_db = True
    else:
        update_db = False
    con = create_lastz_db(
        cfg["taxon_csv"],
        masked_genomes_dir,
        cfg["cluster_name"],
        database_path,
        cfg["max_threads"],
        update_db,
        cfg["aligner"]
    )
    con.close()

    # mlds
    print("Computing MLDs")
    taxon_df = pd.read_csv(cfg["taxon_csv"])
    level_list = sorted(taxon_df[cfg["cluster_name"]].unique())
    levels = list(itertools.combinations(level_list, 2))
    binned_mlds = {}
    for level in levels:
        genome_comps = get_genome_comp(level, taxon_df, cfg["cluster_name"])
        full_mld = get_all_mlds(genome_comps, database_path)
        summed_mld = sum_mlds(full_mld)
        binned_mld = bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=full_mld.shape[0]
        )
        binned_mlds[level] = binned_mld

    if cherry_inf:
        # just return the mld
        return binned_mlds[0]
    return binned_mlds
    os.makedirs(cfg["binned_mld"], exist_ok=True)
    for level in levels:
        binned_mlds[level].to_csv(
            os.path.join(cfg["binned_mld"], f"{level[0]}_{level[1]}.csv"),
            index=False
        )


def fit_mld(binned_mld, empirical_muc, empirical_mus, delta, genome_length, level, optim_method="dual_annealing"):
    "Fits the mosaic model, returns the results of the optimisation."
    print("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS"))
    t0 = time.perf_counter()
    # mus and muc are different for each comp, need to refit here
    _, _, res_opt_4 = fit_params(
        optim_method,
        np.array([9, -8]),
        binned_mld["freq"],
        0.1,
        binned_mld["match_length"],
        empirical_mus,
        empirical_muc,
        delta,
        genome_length,
        only_minus4=True
        )
    print(f"task done in{time.perf_counter() -t0}")
    return binned_mld, res_opt_4, level


def plot_mld_fit_and_expected(fitted_params, binned_mld, ax, empirical_muc, empirical_mus, delta, simulated_params, genome_length, level, save_res=None, outfile=None, plotminus4=True):
    """Plots the fit of the mosaic model to the MLDs and the expected MLD given the simulation parameters."""
    if save_res:
        with open(save_res, "a") as f:
            f.write(f"{level[0]},{level[1]},{fitted_params[0]},{fitted_params[1]}\n")


    max_x = binned_mld["match_length"].max()
    min_y = binned_mld[binned_mld["freq"] > 0]["freq"].min()
    if binned_mld["freq"].sum() == 0:
        # mld is empty, do not plot
        return
    r = np.logspace(0, np.log10(max_x), 1000)
    if plotminus4:
        _, mc = theoretical_mld(fitted_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)
    else:
        mh, mc = theoretical_mld(fitted_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)
    _, mc_simulated = theoretical_mld(simulated_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)

    # get synthetic MLD
    # synthetic_mld = smld.synthetic_mld(summed_mu_array, 10**simulated_params[0], 1000)
    # synthetic_df = pd.DataFrame({"match_length": np.arange(1, 1001), "freq": synthetic_mld})
    # binned_synthetic_df = bin_mld(synthetic_df, 3, 30.5, 0.1, 1)

    # synthetic_mld_delta = smld.synthetic_mld(summed_mu_array, 10**simulated_params[0], 1000, 100, delta)
    # synthetic_df_delta = pd.DataFrame({"match_length": np.arange(1, 1001), "freq": synthetic_mld_delta})
    # binned_synthetic_df_delta = bin_mld(synthetic_df_delta, 3, 30.5, 0.1, 1)


    ax.plot(binned_mld["match_length"], binned_mld["freq"], 'o', label="Observed", color="black")
    # ax.plot(binned_synthetic_df_delta["match_length"], binned_synthetic_df_delta["freq"], 'o', label="Synthetic - delta", color="purple", alpha=0.5)
    # ax.plot(binned_synthetic_df["match_length"], binned_synthetic_df["freq"], 'o', label="Synthetic", color="grey", alpha=0.5)
    if not plotminus4:
        ax.plot(r, mh, label="mh - fit", color="red")
        ax.plot(r, mc, label="mc - fit", color="blue")
    else:
        ax.plot(r, mc, label="fit", color="blue")
    ax.plot(r, mc_simulated, label="mc - expected", color="green")
    ax.set_ylim(ymin=min_y/10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.1, 0.1, f"mus = {empirical_mus:.2e}\nmuc = {empirical_muc:.2e}", transform=ax.transAxes)
    ax.legend()
    ax.set_title(f"MLD fit for {level[0]} vs {level[1]}")
    ax.set_xlabel("Match length")
    ax.set_ylabel("Normalized count")
    if outfile:
        plt.savefig(outfile)
    return {"genome_1": level[0], "genome_2": level[1], "fit_tau": fitted_params[0], "sim_tau": simulated_params[0], "empirical_muc": empirical_muc, "empirical_mus": empirical_mus}


def simulate_infer(simulation_cfg, inference_cfg):
    rate_evolution_parameter = RATE_EVOLUTION_DIC[simulation_cfg["rate_evolution"]]
    tree_heights = to_list(simulation_cfg["tree_height"])
    tree_heights = [float(th) for th in tree_heights]
    if rate_evolution_parameter == "none":
        rate_params = ["none"]
    else:
        rate_params = simulation_cfg[rate_evolution_parameter].copy()
    all_res_df_list = []
    aligners = to_list(inference_cfg["aligner"])
    tmpdir = os.getenv("TMPDIR", default=simulation_cfg["outdir"])
    configs = [{"tree_height": th, "aligner": al, rate_evolution_parameter: rep} \
               for th, al, rep in itertools.product(tree_heights, aligners, rate_params)]
    file_names = [{"db_name":f"{tmpdir}/dbs/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}.db",\
                   "binned_mld_name":f"{tmpdir}/binned_mlds/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_binned_mld/",\
                   "fit_expected_name":f"{tmpdir}/fit_expected_fig/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_fit_expected.png",\
                   "genomes_dir":f"{tmpdir}/genomes/{conf['tree_height']:.2e}_{conf['aligner']}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_genomes/"} \
                  for conf in configs]
    os.makedirs(os.path.join(tmpdir, "dbs"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "binned_mlds"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "fit_expected_fig"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "genomes"), exist_ok=True)
    binned_mld_dic_of_dics = {}

    species_tree = TreeNode.read([simulation_cfg["species_tree"]])
    n_combinations = len(list(itertools.combinations(species_tree.tips(), 2)))
    for conf, names in zip(configs, file_names):
        print("################################################")
        print(f"Currently working on {conf}")
        print("################################################")
        time_tree = get_time_tree(species_tree, conf["tree_height"])
        fig_fe, ax_fe = plt.subplots(n_combinations, 2, figsize=(10, 5*n_combinations))
        if inference_cfg["also_cherries"] == "yes":
            res_cherries = []
            cherries = extract_pairwise_cherries(simulation_cfg_copy)
            for cherry in cherries:
                cherry_names = sorted([tip.name for tip in cherry.tips()])
                cherry_sim_conf, cherry_inf_conf = make_cherry_conf(simulation_cfg_copy, inference_cfg_copy, cherry, cherry_names)
                # print(cherry_sim_conf)
                tips_mut_rate_cherry = run_simulation(cherry_sim_conf)
                # urgh c'est laid
                summed_mus_cherry = [(a + b)/2 for a, b in tips_mut_rate_cherry[next(iter(tips_mut_rate_cherry))]]
                binned_mld = cherries_inference(cherry_inf_conf, cherry_names)
                _, res, _ = fit_mld(binned_mld,
                                    np.min(summed_mus_cherry),
                                    np.max(summed_mus_cherry),
                                    cherry_inf_conf["delta"],
                                    cherry_sim_conf["n_gene_trees"] * cherry_sim_conf["length_gene"],
                                    " ".join(cherry_names)
                                    )
                cherry_time_tree = get_time_tree(cherry, cherry_sim_conf["tree_height"])
                cherry_1 = cherry_time_tree.find(cherry_names[0])
                cherry_2 = cherry_time_tree.find(cherry_names[1])
                sim_tau_cherry = cherry_1.distance(cherry_2)
                res_cherries.append({"genome_1": cherry_names[0],
                                     "genome_2": cherry_names[1],
                                     "fit_tau_cherry": res.x[0],
                                     "sim_tau_cherry": np.log10(sim_tau_cherry)})
            cherries_df = pd.DataFrame(res_cherries)

        #simulations
        simulation_cfg_copy = simulation_cfg.copy()
        simulation_cfg_copy["tree_height"] = conf["tree_height"]
        simulation_cfg_copy[rate_evolution_parameter] = conf[rate_evolution_parameter]
        simulation_cfg_copy["genomes_dir"] = names["genomes_dir"]
        tips_mut_rate = run_simulation(simulation_cfg_copy, ax_fe)

        # parsing
        inference_cfg_copy = inference_cfg.copy()
        inference_cfg_copy["database_name"] = names["db_name"]
        inference_cfg_copy["cluster_name"] = "clade"
        inference_cfg_copy["genomes_dir"] = names["genomes_dir"]
        inference_cfg_copy["binned_mld"] = names["binned_mld_name"]
        inference_cfg_copy["aligner"] = conf["aligner"]
        inference_cfg_copy["delta"] = ALIGNER_DELTA[conf["aligner"]]
        key = tuple([(key, item) for key, item in conf.items()])
        binned_mld_dic_of_dics[key] = simplified_parsing(inference_cfg_copy)

        # fitting
        # prepare lists of arguments for parallel execution
        all_summed_mus = {pair: [(a + b)/2 for a, b in mut_rate] for pair, mut_rate in tips_mut_rate.items()}
        sim_tau_list = {pair: np.log10(time_tree.find(pair[0]).distance(time_tree.find(pair[1]))) for pair in all_summed_mus.keys()}
        if inference_cfg_copy["exp_mus"]:
            muc_list = {pair: np.min(all_summed_mus[pair]) for pair in all_summed_mus.keys()}
            mus_list = {pair: np.max(all_summed_mus[pair]) for pair in all_summed_mus.keys()}
        else:
            muc_list = {pair: float(simulation_cfg_copy["muc"]) for pair in all_summed_mus.keys()}
            mus_list = {pair: float(simulation_cfg_copy["mus"]) for pair in all_summed_mus.keys()}

        if n_combinations == 1:
            ax2_list = {list(all_summed_mus.keys())[0]: ax_fe[1]}
        else:
            ax2_list = {pair: ax_fe[i, 1] for i, pair in enumerate(all_summed_mus.keys())}

        res_fit = {}
        overall_args = [
            (
                binned_mld_dic_of_dics[key][pair],
                muc_list[pair],
                mus_list[pair],
                inference_cfg_copy["delta"],
                simulation_cfg_copy["length_gene"]*simulation_cfg_copy["n_gene_trees"],
                pair,
                inference_cfg_copy["optim"]
            )
            for pair in all_summed_mus.keys()
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=inference_cfg_copy["max_threads"]) as executor:
            futures = [executor.submit(fit_mld, *args) for args in overall_args]
            for future in concurrent.futures.as_completed(futures):
                res_fit[future.result()[2]] = future.result()

        # plotting
        plot_res = []
        for pair, (binned_mld, res_opt, pair) in res_fit.items():
            res_dic = plot_mld_fit_and_expected(
                res_opt.x,
                binned_mld,
                ax2_list[pair],
                muc_list[pair],
                mus_list[pair],
                inference_cfg_copy["delta"],
                (sim_tau_list[pair], -20),
                simulation_cfg_copy["length_gene"]*simulation_cfg_copy["n_gene_trees"],
                pair,
                None,
                None,
                True
            )
            opt_value_dic = {"minimum": res_opt.fun}
            plot_res.append(res_dic | opt_value_dic)
        fig_fe.tight_layout()
        fig_fe.savefig(names["fit_expected_name"], dpi=300)
        plt.close()

        # saving
        res_fit_df = pd.DataFrame(plot_res)
        res_fit_df[rate_evolution_parameter] = conf[rate_evolution_parameter] 
        res_fit_df["tree_height"] = conf["tree_height"]
        all_res_df_list.append(res_fit_df)
        if inference_cfg_copy["also_cherries"] == "yes":
            res_fit_df = pd.merge(res_fit_df, cherries_df, on=["genome_1", "genome_2"], how="left")
            print(res_fit_df)

    all_res_fit_df = pd.concat(all_res_df_list).reset_index(drop=True)
    all_res_fit_df.to_csv(os.path.join(tmpdir, "all_res_fit.csv"), index=False)
    all_binned_mlds = []
    for conf, binned_mld_dic in binned_mld_dic_of_dics.items():
        conf_binned_mld = []
        for level, binned_mld in binned_mld_dic.items():
            binned_mld["species_1"] = level[0]
            binned_mld["species_2"] = level[1]
            conf_binned_mld.append(binned_mld)
        conf_binned_mld_df = pd.concat(conf_binned_mld)
        for param, value in conf:
            conf_binned_mld_df[param] = value
        all_binned_mlds.append(conf_binned_mld_df)
    all_binned_mlds_df = pd.concat(all_binned_mlds).reset_index(drop=True)
    all_binned_mlds_df.to_csv(os.path.join(tmpdir, "all_binned_mlds.csv"), index=False)

    if tmpdir != simulation_cfg["outdir"]:
        shutil.copytree(tmpdir, simulation_cfg["outdir"], dirs_exist_ok=True)


def simulate_trees(simulation_cfg):
    """
    Simulate only trees, plot the distribution of mutation rates, computes pseudo-empirical mlds.
    """
    rate_evolution_parameter = RATE_EVOLUTION_DIC[simulation_cfg["rate_evolution"]]
    tree_heights = to_list(simulation_cfg["tree_height"])
    tree_heights = [float(th) for th in tree_heights]
    if rate_evolution_parameter == "none":
        rate_params = ["none"]
    else:
        rate_params = simulation_cfg[rate_evolution_parameter].copy()
    configs = [{"tree_height": th, rate_evolution_parameter: rep} for th, rep in itertools.product(tree_heights, rate_params)]
    plot_names = [f"{conf['tree_height']:.2e}_{rate_evolution_parameter}_{conf[rate_evolution_parameter]}_mudistr.png" for conf in configs]
    species_tree = TreeNode.read([simulation_cfg["species_tree"]])
    n_combinations = len(list(itertools.combinations(species_tree.tips(), 2)))

    for conf, plot_n in zip(configs, plot_names):
        time_tree = get_time_tree(species_tree, conf["tree_height"])
        fig, axs = plt.subplots(n_combinations, 2, figsize=(10, 5*n_combinations), layout="constrained")
        rate_evolution_value = None if rate_evolution_parameter == "none" else float(conf[rate_evolution_parameter])
        mutation_rate_trees = generate_gene_trees(
            time_tree=time_tree,
            n=simulation_cfg["n_gene_trees"],
            muc=float(simulation_cfg["muc"]),
            mus=float(simulation_cfg["mus"]),
            threads=simulation_cfg["threads"],
            rate_evolution=simulation_cfg["rate_evolution"],
            rate_evolution_value=rate_evolution_value,
            mean_steps=bool(simulation_cfg.get("mean_steps", False)),
        )
        tips_mut_rate = get_all_pair_mutation_rate(mutation_rate_trees, time_tree)
        mean_mus = {}
        for pair, dists in tips_mut_rate.items():
            mean_mus[pair] = [a + b for a, b in dists]
        muc_mus = {pair: [min(muss), max(muss)] for pair, muss in mean_mus.items()}
        plot_distance_distribution(axs[:, 0], mutation_rate_trees=mutation_rate_trees, muc_mus=muc_mus, time_tree=time_tree)
        plot_pseudo_empirical(axs[:, 1], tips_mut_rate, time_tree, float(simulation_cfg["length_gene"]))
        fig.savefig(Path(simulation_cfg["outdir"]) / plot_n, dpi=300)





if False:
    # debug region
    # simulation data
    gene_trees = []
    gene_tree_dir = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw/rw_v_rw_1.00e+04"
    res_dir = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw/res_rw_v_rw_1.00e+04"
    for _ in range(5000):
        gene_trees.append(TreeNode.read(f"{gene_tree_dir}/gene_tree_{_}.newick"))
    species_tree = TreeNode.read([f"(A:3,(B:2,(C:1,D:1):1):1);"])
    level = ("A", "B")
    tree_height = 2e8
    time_tree = gt.get_time_tree(species_tree, tree_height)
    sim_tau = time_tree.find(level[0]).distance(time_tree.find(level[1]))

    all_paired_mut_rates_dic = get_all_pair_mutation_rate(time_tree, gene_trees)
    all_summed_mus = [(a + b)/2 for a, b in all_paired_mut_rates_dic[level]]
    muc = np.min(all_summed_mus)
    mus = np.max(all_summed_mus)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_mld_fit_and_expected(ax, res_dir, muc, mus, 0.82, (np.log10(sim_tau), -20), all_summed_mus, all_paired_mut_rates_dic[level], 5e6, level)
    fig.show()



    # plot fit results vs input of simulation
    res_csv = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw_height/all_res_fit.csv"
    res_df = pd.read_csv(res_csv)

    res_df["fit_tau"] = res_df["fit_tau"].apply(lambda x: 10**x)
    res_df["sim_tau"] = res_df["sim_tau"].apply(lambda x: 10**x)
    min_tau = res_df["fit_tau"].min()
    max_tau = res_df["fit_tau"].max()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for rw_step_fraction, df in res_df.groupby("rw_step_fraction"):
        ax.plot(df["sim_tau"], df["fit_tau"], "o", label=rw_step_fraction)
    ax.plot([min_tau, max_tau], [min_tau, max_tau], "k--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fitted tau")
    ax.set_ylabel("Simulated tau")
    ax.legend()
    ax.set_title("Fitted tau vs simulated tau")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(res_csv), "fit_vs_sim.png"), dpi=300)
