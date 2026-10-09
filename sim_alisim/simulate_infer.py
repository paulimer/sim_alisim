#!/usr/bin/env python3
import shutil

import argparse
import concurrent.futures
import cProfile
import itertools
import multiprocessing
import os
from pathlib import Path
import pstats
import sys
import time
import tracemalloc

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import TreeNode
import yaml

RATE_EVOLUTION_DIC = {
    "random_walk": "n_steps",
    "kishino": "nu",
    "none": "none",
    "null": "none",
}

from sim_alisim.gene_trees import (
    get_time_tree,
    run_simulation,
    get_all_pair_mutation_rate,
    to_list,
    generate_gene_trees,
    plot_distance_distribution,
    plot_pseudo_empirical,
    get_pseudo_empirical,
)
from sim_alisim.theoretical_vs_simulated import ALIGNER_DELTA
from mosaic_method.fitting import theoretical_mld, fit_params
from mosaic_method.parsing import get_genome_comp, sum_mlds, bin_mld
from mosaic_method.aligning import align_exec, create_lastz_db


def _conf_tag(*args) -> str:
    def fmt(arg):
        if isinstance(arg, float):
            return f"{arg:.2e}"
        return str(arg)

    return "_".join(fmt(arg) for arg in args)


def _tips_ready(genomes_dir, tip_names):
    """True if genomes_dir contains and mutation-rate trees.

    For skipping simulations for the sequences=False case
    tips_mut_rate can be reconstructed from the saved trees via _load_tips_mut_rate().
    """
    if not os.path.isdir(genomes_dir):
        return False
    return os.path.exists(os.path.join(genomes_dir, "mutation_rate_trees.nwk"))


def _genomes_ready(genomes_dir, tip_names):
    """True if genomes_dir contains the expected FASTAs, taxon.csv, and mutation-rate trees.

    When all three artefacts are present the simulation stage can be skipped and
    tips_mut_rate can be reconstructed from the saved trees via _load_tips_mut_rate().
    """
    if not os.path.isdir(genomes_dir):
        return False
    expected = {f"{name}.fasta" for name in tip_names} | {
        "taxon.csv",
        "mutation_rate_trees.nwk",
    }
    return expected.issubset(set(os.listdir(genomes_dir)))


def _db_ready(db_path):
    """True when the alignment database from a prior run already exists."""
    return os.path.exists(db_path)


def load_tips_mut_rate(genomes_dir, time_tree):
    """Reconstruct tips_mut_rate from saved mutation-rate trees without re-running simulation."""
    nwk_path = os.path.join(genomes_dir, "mutation_rate_trees.nwk")
    with open(nwk_path) as fh:
        mutation_rate_trees = [
            TreeNode.read([line.strip()]) for line in fh if line.strip()
        ]
    return get_all_pair_mutation_rate(mutation_rate_trees, time_tree)


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
    cherry_sim_conf[
        "outdir"
    ] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    cherry_sim_conf["rate_evolution"] = "random_walk"
    cherry_sim_conf["tree_height"] = (
        cherry.height()[0]
        / TreeNode.read([simulation_cfg["species_tree"]]).height()[0]
        * simulation_cfg["tree_height"]
    )
    cherry_inf_conf = inference_cfg.copy()
    cherry_inf_conf[
        "genomes_dir"
    ] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    cherry_inf_conf[
        "results_dir"
    ] += f"_cherry_{'_vs_'.join(names)}_{simulation_cfg['tree_height']}"
    # print(cherry_sim_conf)
    return cherry_sim_conf, cherry_inf_conf


def cherries_inference(cfg, names):
    """Just get me the binned mld bro."""
    align_res = align_exec(
        [
            cfg["genomes_dir"] + f"/{names[0]}.fasta",
            cfg["genomes_dir"] + f"/{names[1]}.fasta",
        ],
        "lastz",
        "bla",
    )
    unbinned_mld = align_res[2]["result"][0]
    summed_mld = pd.DataFrame(
        {"freq": unbinned_mld}, index=range(1, len(unbinned_mld) + 1)
    ).reset_index(names="match_length")
    binned_mld = bin_mld(
        summed_mld, linear_bin_width=3, limit_size=30.5, power_increment=0.1, ncomp=1
    )
    return binned_mld


def simplified_parsing(cfg, cherry_inf=False, skip_alignment=False):
    """
    Aligns, gathers MLDs and bins them.

    Parameters
    ----------
    skip_alignment : bool
        If True, assume the database at cfg["database_name"] already exists and skip the
        alignment step. Used when restarting a run where the DB is already present.
    """

    if not os.path.exists(cfg["taxon_csv"]):
        cfg["taxon_csv"] = os.path.join(cfg["genomes_dir"], cfg["taxon_csv"])
        if not os.path.exists(cfg["taxon_csv"]):
            print(f"Taxon csv not found at {cfg['taxon_csv']}")
            sys.exit(1)

    masked_genomes_dir = cfg["genomes_dir"]
    database_path = cfg["database_name"]
    if not skip_alignment:
        # alignment
        print("Aligning genomes")
        if cfg["aligner"].startswith("lastz"):
            aligner = "lastz"
        elif cfg["aligner"].startswith("mummer"):
            aligner = "mummer"
        else:
            aligner = "no_aligner"
        update_db = os.path.exists(database_path)
        con = create_lastz_db(
            cfg["taxon_csv"],
            masked_genomes_dir,
            cfg["cluster_name"],
            database_path,
            cfg["max_threads"],
            update_db,
            aligner,
        )
        con.close()
    else:
        print(
            f"Skipping alignment, reading MLDs from existing database {database_path}."
        )

    # mlds
    print("Computing MLDs")
    taxon_df = pd.read_csv(cfg["taxon_csv"])
    level_list = sorted(taxon_df[cfg["cluster_name"]].unique())
    levels = list(itertools.combinations(level_list, 2))
    binned_mlds = {}
    for level in levels:
        genome_comps = get_genome_comp(level, taxon_df, cfg["cluster_name"])
        summed_mld = sum_mlds(genome_comps, database_path)
        binned_mld = bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=len(genome_comps),
        )
        binned_mlds[level] = binned_mld

    if cherry_inf:
        # just return the mld
        return next(iter(binned_mlds.values()))
    return binned_mlds


def fit_mld(
    binned_mld,
    empirical_muc,
    empirical_mus,
    delta,
    genome_length,
    level,
    optim_method="dual_annealing",
):
    "Fits the mosaic model, returns the results of the optimisation."
    print("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS"))
    t0 = time.perf_counter()
    print(f"fitting with {optim_method} using aligner delta {delta}")
    # mus and muc are different for each comp, need to refit here
    _, _, res_opt_4 = fit_params(
        optim_method,
        np.array([9, -20]),
        binned_mld["freq"],
        0.1,
        binned_mld["match_length"],
        empirical_mus,
        empirical_muc,
        delta,
        genome_length,
        only_minus4=True,
    )
    print(f"task done in{time.perf_counter() -t0}")
    return binned_mld, res_opt_4, level


def plot_mld_fit_and_expected(
    fitted_params,
    binned_mld,
    ax,
    empirical_muc,
    empirical_mus,
    delta,
    simulated_params,
    genome_length,
    gene_length,
    level,
    save_res=None,
    outfile=None,
    plotminus4=True,
    fitted_muc=None,
):
    """Plots the fit of the mosaic model to the MLDs and the expected MLD given the simulation parameters."""
    if fitted_muc is None:
        fitted_muc = empirical_muc
    if save_res:
        with open(save_res, "a") as f:
            f.write(f"{level[0]},{level[1]},{fitted_params[0]},{fitted_params[1]}\n")

    max_x = binned_mld["match_length"].max()
    min_y = binned_mld[binned_mld["freq"] > 0]["freq"].min()
    if binned_mld["freq"].sum() == 0:
        # mld is empty, do not plot
        return {
            "genome_1": level[0],
            "genome_2": level[1],
            "fit_tau": None,
            "sim_tau": simulated_params[0],
            "empirical_muc": empirical_muc,
            "empirical_mus": empirical_mus,
        }
    r = np.logspace(0, np.log10(max_x), 1000)
    if plotminus4:
        _, mc = theoretical_mld(
            fitted_params,
            0.1,
            r,
            empirical_mus,
            fitted_muc,
            delta,
            genome_length,
            False,
        )
    else:
        mh, mc = theoretical_mld(
            fitted_params,
            0.1,
            r,
            empirical_mus,
            fitted_muc,
            delta,
            genome_length,
            False,
        )
    _, mc_simulated = theoretical_mld(
        simulated_params,
        0.1,
        r,
        empirical_mus,
        empirical_muc,
        delta,
        genome_length,
        False,
    )

    ax.plot(
        binned_mld["match_length"],
        binned_mld["freq"],
        "o",
        label="Observed",
        color="black",
    )
    if not plotminus4:
        ax.plot(r, mh, label="mh - fit", color="tab:red")
        ax.plot(r, mc, label="mc - fit", color="tab:blue")
    else:
        ax.plot(r, mc, label="fit", color="tab:blue")
    ax.plot(r, mc_simulated, label="mc - expected", color="tab:green")
    ax.set_ylim(min_y / 10, None)
    ax.set_xlim(1, gene_length * 10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(
        0.1,
        0.1,
        f"mus = {empirical_mus:.2e}\nmuc = {empirical_muc:.2e}\ndelta/tau = {delta/10**fitted_params[0]:.2e}",
        transform=ax.transAxes,
    )
    ax.legend()
    ax.set_title(f"MLD fit for {level[0]} vs {level[1]}")
    ax.set_xlabel("Match length")
    ax.set_ylabel("Normalized count")
    if outfile:
        plt.savefig(outfile)
    return {
        "genome_1": level[0],
        "genome_2": level[1],
        "fit_tau": fitted_params[0],
        "sim_tau": simulated_params[0],
        "empirical_muc": empirical_muc,
        "empirical_mus": empirical_mus,
    }


def get_sim_configs(simulation_cfg: dict, tmpdir: str) -> tuple[list[dict], list]:
    """
    Returns a list of confs for each combination (product) of the lists items of simulation_cfg.

    This list of confs should also contain the non-listed items of simulation_cfg
    """
    listed = {k: v for k, v in simulation_cfg.items() if isinstance(v, list)}
    keys = list(listed.keys())
    confs = []
    for combo in itertools.product(*listed.values()):
        conf = {**simulation_cfg, **dict(zip(keys, combo))}
        conf["genomes_dir"] = f"{tmpdir}/genomes/{_conf_tag(*combo)}"
        conf["tree_height"] = float(conf["tree_height"])
        confs.append(conf)
    return confs, keys


def get_inf_configs(
    inference_cfg: dict, tmpdir: str, sim_conf: dict, sim_keys: list
) -> tuple[list[dict], list]:
    """
    Returns a list of confs for each combination (product) of the lists items of inference_cfg.

    This list of confs should also contain the non-listed items of inference_cfg.
    Additionally, it takes the current sim_conf list keys, not to overwrite inference results of other sim runs.
    """
    listed = {k: v for k, v in inference_cfg.items() if isinstance(v, list)}
    keys = list(listed.keys())
    confs = []
    for combo in itertools.product(*listed.values()):
        conf = {**inference_cfg, **dict(zip(keys, combo))}
        sim_combo = [v for k, v in sim_conf.items() if k in sim_keys]
        ident_values = list(combo) + sim_combo
        conf["database_name"] = f"{tmpdir}/dbs/{_conf_tag(*ident_values)}.db"
        conf["fit_expected_name"] = (
            f"{tmpdir}/fit_expected_fig/{_conf_tag(*ident_values)}_fit_expected.png"
        )
        conf["delta"] = ALIGNER_DELTA[conf["aligner"]]
        confs.append(conf)
    return confs, keys


def simulate_infer(simulation_cfg, inference_cfg):
    rate_evolution_parameter = RATE_EVOLUTION_DIC[simulation_cfg["rate_evolution"]]
    if rate_evolution_parameter == "none":
        rate_params = ["none"]
    else:
        rate_params = simulation_cfg[rate_evolution_parameter].copy()

    all_res_df_list = []
    tmpdir = os.getenv("MXQ_JOB_TMPDIR", default=simulation_cfg["outdir"])

    os.makedirs(os.path.join(tmpdir, "dbs"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "fit_expected_fig"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "genomes"), exist_ok=True)
    binned_mld_dic_of_dics = {}

    species_tree = TreeNode.read([simulation_cfg["species_tree"]])
    n_combinations = len(list(itertools.combinations(species_tree.tips(), 2)))
    sim_configs, sim_keys = get_sim_configs(simulation_cfg, tmpdir)
    for sim_conf in sim_configs:
        print("################################################")
        print(f"Currently working on:")
        for k in sim_keys:
            print(f"{k}: {sim_conf[k]}")
        print("################################################")
        time_tree = get_time_tree(species_tree, sim_conf["tree_height"])
        fig_fe = plt.Figure(figsize=(10, 5 * n_combinations))
        ax_fe = fig_fe.subplots(n_combinations, 2)

        # if inference_cfg["also_cherries"]:
        #     res_cherries = []
        #     cherries = extract_pairwise_cherries(sim_conf)
        #     for cherry in cherries:
        #         cherry_names = sorted([tip.name for tip in cherry.tips()])
        #         cherry_sim_conf, cherry_inf_conf = make_cherry_conf(
        #             sim_conf, inference_cfg, cherry, cherry_names
        #         )
        #         # print(cherry_sim_conf)
        #         tips_mut_rate_cherry = run_simulation(cherry_sim_conf)
        #         # urgh c'est laid
        #         summed_mus_cherry = [
        #             (a + b)
        #             for a, b in tips_mut_rate_cherry[next(iter(tips_mut_rate_cherry))]
        #         ]
        #         binned_mld = cherries_inference(cherry_inf_conf, cherry_names)
        #         _, res, _ = fit_mld(
        #             binned_mld,
        #             np.min(summed_mus_cherry),
        #             np.max(summed_mus_cherry),
        #             cherry_inf_conf["delta"],
        #             cherry_sim_conf["n_gene_trees"] * cherry_sim_conf["length_gene"],
        #             " ".join(cherry_names),
        #         )
        #         cherry_time_tree = get_time_tree(cherry, cherry_sim_conf["tree_height"])
        #         cherry_1 = cherry_time_tree.find(cherry_names[0])
        #         cherry_2 = cherry_time_tree.find(cherry_names[1])
        #         sim_tau_cherry = cherry_1.distance(cherry_2)
        #         res_cherries.append(
        #             {
        #                 "genome_1": cherry_names[0],
        #                 "genome_2": cherry_names[1],
        #                 "fit_tau_cherry": res.x[0],
        #                 "sim_tau_cherry": np.log10(sim_tau_cherry),
        #             }
        #         )
        #     cherries_df = pd.DataFrame(res_cherries)

        tip_names = [tip.name for tip in time_tree.tips()]
        if _genomes_ready(sim_conf["genomes_dir"], tip_names) & sim_conf["sequences"]:
            print(f"Genomes found in {sim_conf['genomes_dir']}, skipping simulation.")
            tips_mut_rate = load_tips_mut_rate(sim_conf["genomes_dir"], time_tree)
        elif _tips_ready(sim_conf["genomes_dir"], tip_names) & (
            not sim_conf["sequences"]
        ):
            tips_mut_rate = load_tips_mut_rate(sim_conf["genomes_dir"], time_tree)
        elif not sim_conf["sequences"]:
            rate_evolution_value = (
                None
                if rate_evolution_parameter == "none"
                else float(sim_conf[rate_evolution_parameter])
            )
            mutation_rate_trees = generate_gene_trees(
                time_tree=time_tree,
                n=sim_conf["n_gene_trees"],
                mumin=float(sim_conf["mumin"]),
                mumax=float(sim_conf["mumax"]),
                threads=sim_conf["threads"],
                rate_evolution=sim_conf["rate_evolution"],
                rate_evolution_value=rate_evolution_value,
                mean_steps=bool(sim_conf.get("mean_steps", False)),
                linear_mu=bool(sim_conf.get("linear_mu", False)),
            )
            tips_mut_rate = get_all_pair_mutation_rate(mutation_rate_trees, time_tree)

            avg_mus = {}
            for pair, rates in tips_mut_rate.items():
                avg_mus[pair] = [(a + b) / 2 for a, b in rates]
            muc_mus = {
                pair: [min(avg_mu), max(avg_mu)] for pair, avg_mu in avg_mus.items()
            }
            plot_distance_distribution(ax_fe, mutation_rate_trees, muc_mus, time_tree)

        else:
            tips_mut_rate = run_simulation(sim_conf, ax_fe)

        inf_configs, inf_keys = get_inf_configs(
            inference_cfg, tmpdir, sim_conf, sim_keys
        )
        for inf_conf in inf_configs:
            # parsing/aligning
            inf_conf["genomes_dir"] = sim_conf["genomes_dir"]
            conf = sim_conf | inf_conf
            conf_keys = sim_keys + inf_keys
            id_exp = tuple((k, conf[k]) for k in conf_keys)
            skip_alignment = _db_ready(inf_conf["database_name"])

            if skip_alignment:
                print(
                    f"Database {inf_conf['database_name']} found, skipping alignment."
                )
            if sim_conf["sequences"]:
                binned_mld_dic_of_dics[id_exp] = simplified_parsing(
                    inf_conf, skip_alignment=skip_alignment
                )
            else:
                binned_mld_dic_of_dics[id_exp] = get_pseudo_empirical(
                    tips_mut_rate, time_tree, sim_conf["length_gene"]
                )

            # fitting
            # prepare lists of arguments for parallel execution
            all_summed_mus = {
                pair: [(a + b) / 2 for a, b in mut_rate]
                for pair, mut_rate in tips_mut_rate.items()
            }
            sim_tau_list = {
                pair: np.log10(
                    time_tree.find(pair[0]).distance(time_tree.find(pair[1]))
                )
                for pair in all_summed_mus.keys()
            }
            # whether to use experimental mus for fitting
            muc_list = {
                pair: np.min(all_summed_mus[pair]) for pair in all_summed_mus.keys()
            }
            mus_list = {
                pair: np.max(all_summed_mus[pair]) for pair in all_summed_mus.keys()
            }
            if n_combinations == 1:
                ax2_list = {list(all_summed_mus.keys())[0]: ax_fe[1]}
            else:
                ax2_list = {
                    pair: ax_fe[i, 1] for i, pair in enumerate(all_summed_mus.keys())
                }

            # fitting
            plot_res = []
            genome_length = sim_conf["length_gene"] * sim_conf["n_gene_trees"]
            if inf_conf["optim"] == "global":
                from mosaic_method.global_fitting_reparam_kappa_inv import (
                    run_global_fitting,
                    precompute_pairs as precompute_pairs_global,
                )

                pair_list = list(all_summed_mus.keys())
                L0_df_sim = pd.DataFrame(
                    {
                        "bac1": [p[0] for p in pair_list],
                        "bac2": [p[1] for p in pair_list],
                        "L0": float(genome_length),
                    }
                )
                precomputed = precompute_pairs_global(
                    binned_mld_dic_of_dics[id_exp], L0_df_sim
                )
                print(f"Delta = {inf_conf['delta']}")
                rows = run_global_fitting(
                    precomputed, kappa_grid_n=20, top_k=4, delta=inf_conf["delta"]
                )
                # crucial bit. If there are > 2 tips, we only use the max
                mus = max(mus_list.values())
                muc = min(muc_list.values())
                kappa = rows[0]["kappa"]
                for row in rows:
                    pair = (row["species_1"], row["species_2"])
                    log10_tau = np.log10(row["theta"] / mus)
                    log10_rho = np.log10(row["xi"] * mus)
                    res_dic = plot_mld_fit_and_expected(
                        [log10_tau, log10_rho],
                        binned_mld_dic_of_dics[id_exp][pair],
                        ax2_list[pair],
                        muc,
                        mus,
                        inf_conf["delta"],
                        (sim_tau_list[pair], -20),
                        genome_length,
                        sim_conf["n_gene_trees"],
                        pair,
                        None,
                        None,
                        True,
                        fitted_muc=1 / kappa * mus,
                    )
                    plot_res.append(
                        res_dic
                        | {
                            "minimum": row["minimum"],
                            "theta": row["theta"],
                            "xi": row["xi"],
                            "kappa": row["kappa"],
                            "fitted_muc": 1 / kappa * mus,
                        }
                    )
            else:
                res_fit = {}
                overall_args = [
                    (
                        binned_mld_dic_of_dics[id_exp][pair],
                        muc_list[pair],
                        mus_list[pair],
                        inf_conf["delta"],
                        genome_length,
                        pair,
                        inf_conf["optim"],
                    )
                    for pair in all_summed_mus.keys()
                ]
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=inf_conf["max_threads"],
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    futures = [executor.submit(fit_mld, *args) for args in overall_args]
                    for future in concurrent.futures.as_completed(futures):
                        res_fit[future.result()[2]] = future.result()

                for binned_mld, res_opt, pair in res_fit.values():
                    res_dic = plot_mld_fit_and_expected(
                        res_opt.x,
                        binned_mld,
                        ax2_list[pair],
                        muc_list[pair],
                        mus_list[pair],
                        inf_conf["delta"],
                        (sim_tau_list[pair], -20),
                        genome_length,
                        sim_conf["n_gene_trees"],
                        pair,
                        None,
                        None,
                        True,
                    )
                    plot_res.append(res_dic | {"minimum": res_opt.fun})

            fig_fe.tight_layout()
            fig_fe.savefig(inf_conf["fit_expected_name"], dpi=300)
            plt.close()
            for ax in ax2_list.values():
                ax.clear()

            # saving
            res_fit_df = pd.DataFrame(plot_res)
            try:
                res_fit_df[rate_evolution_parameter] = sim_conf[
                    rate_evolution_parameter
                ]
            except KeyError:
                res_fit_df[rate_evolution_parameter] = "none"
            for k in sim_keys:
                res_fit_df[k] = sim_conf[k]
            res_fit_df["aligner"] = inf_conf["aligner"]
            all_res_df_list.append(res_fit_df)
            # if inf_conf["also_cherries"]:
            #     res_fit_df = pd.merge(
            #         res_fit_df, cherries_df, on=["genome_1", "genome_2"], how="left"
            #     )
            #     print(res_fit_df)

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
        all_binned_mlds_df.to_csv(
            os.path.join(tmpdir, "all_binned_mlds.csv"), index=False
        )

    if tmpdir != simulation_cfg["outdir"]:
        shutil.copytree(tmpdir, simulation_cfg["outdir"], dirs_exist_ok=True)


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

    all_paired_mut_rates_dic = get_all_pair_mutation_rate(gene_trees, time_tree)
    all_summed_mus = [a + b for a, b in all_paired_mut_rates_dic[level]]
    muc = np.min(all_summed_mus)
    mus = np.max(all_summed_mus)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_mld_fit_and_expected(
        ax,
        res_dir,
        muc,
        mus,
        0.82,
        (np.log10(sim_tau), -20),
        all_summed_mus,
        all_paired_mut_rates_dic[level],
        5e6,
        level,
    )
    fig.show()

    # plot fit results vs input of simulation
    res_csv = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw_height/all_res_fit.csv"
    res_df = pd.read_csv(res_csv)

    res_df["fit_tau"] = res_df["fit_tau"].apply(lambda x: 10**x)
    res_df["sim_tau"] = res_df["sim_tau"].apply(lambda x: 10**x)
    min_tau = res_df["fit_tau"].min()
    max_tau = res_df["fit_tau"].max()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for n_steps, df in res_df.groupby("n_steps"):
        ax.plot(df["sim_tau"], df["fit_tau"], "o", label=n_steps)
    ax.plot([min_tau, max_tau], [min_tau, max_tau], "k--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fitted tau")
    ax.set_ylabel("Simulated tau")
    ax.legend()
    ax.set_title("Fitted tau vs simulated tau")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(res_csv), "fit_vs_sim.png"), dpi=300)
