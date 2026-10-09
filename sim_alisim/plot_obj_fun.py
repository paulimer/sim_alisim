import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
from sim_alisim.simulate_infer import RATE_EVOLUTION_DIC
from sim_alisim.simulate_infer import ALIGNER_DELTA
from mosaic_method.global_fitting_reparam_kappa_inv import theoretical_mld_vectorized
import yaml


def plot_obj_fun(res_dirs):
    res_dirs = pathlib.Path(res_dirs)
    sim_res = [x for x in res_dirs.iterdir() if x.is_dir()]
    all_res_fit_all = {}
    print(sim_res)
    for res in sim_res:
        try:
            all_res_df = pd.read_csv(res / "all_res_fit.csv")
        except FileNotFoundError:
            continue
        else:
            all_res_fit_all[res.name] = all_res_df

    all_res_fit_df = (
        pd.concat(all_res_fit_all)
        .reset_index(names=["exp", "drop"])
        .drop(["drop"], axis=1)
    )
    all_res_fit_df = all_res_fit_df.melt(
        id_vars=list(set(all_res_fit_df.columns) - set(RATE_EVOLUTION_DIC.values()))
    ).rename(
        {
            "variable": "rate_evolution_parameter",
            "value": "rate_evolution_parameter_value",
        },
        axis=1,
    )
    sorted_exp = sorted(list(all_res_fit_df["exp"].drop_duplicates()))
    all_res_fit_df["general_exp"] = all_res_fit_df["exp"].apply(
        lambda x: (
            "__".join(x.split("__")[:2]) if x.startswith("rand") else x.split("__")[0]
        )
    )

    g = sns.catplot(
        data=all_res_fit_df,
        col="exp",
        y="minimum",
        x="rate_evolution_parameter_value",
        hue="general_exp",
        kind="box",
        col_wrap=3,
        col_order=sorted_exp,
        sharex=False,
        sharey=False,
    )
    return g


def barplot_fun(res_csv):
    """bizarre pourquoi j'utilise pas les mlds là ? c'était pas ça le but?"""
    res_df = pd.read_csv(res_csv)
    res_df = res_df[
        (res_df["aligner"] == "lastz_uncorrected")
        | (res_df["aligner"] == "lastz_corrected")
    ][["sim_tau", "aligner", "minimum"]]
    res_df = pd.merge(
        res_df[res_df["aligner"] == "lastz_uncorrected"].drop("aligner", axis=1),
        res_df[res_df["aligner"] == "lastz_corrected"].drop("aligner", axis=1),
        on=["sim_tau"],
        suffixes=("_uncorrected", "_corrected"),
    )
    res_df["correction gain"] = (
        res_df["minimum_uncorrected"] - res_df["minimum_corrected"]
    ) / res_df["minimum_uncorrected"]
    res_df["time divergence"] = (10 ** res_df["sim_tau"]) / 2

    theme_dict = {**axes_style("whitegrid"), "grid.linestyle": ":"}
    tick_positions = [
        i * 1e8 for i in range(1, int(res_df["time divergence"].max() / 1e8) + 4)
    ]
    labeled = {1e8, 5e8, 1e9}
    tick_labels = [f"{x:.0e}" if x in labeled else "" for x in tick_positions]
    g = (
        so.Plot(res_df, x="time divergence", y="correction gain")
        .add(so.Dot(pointsize=10))
        .scale(
            x=so.Continuous(trans="log")
            .tick(locator=mticker.FixedLocator(tick_positions))
            .label(formatter=mticker.FixedFormatter(tick_labels)),
        )
        .theme(theme_dict)
    )
    return g


def th_obs(binned_mld, logtau, L0, mus, kappa, aligner):
    ml = binned_mld["match_length"].values
    ml_safe = ml
    # thetas, xis_raw, smal_dif, ml_safe, kappa_raw, delta, L0s
    th_mld = theoretical_mld_vectorized(
        np.array(
            (10**logtau * mus,),
        ),
        np.array(
            (-20,),
        ),
        0.1,
        ml_safe,
        kappa,
        ALIGNER_DELTA[aligner],
        np.array(
            (L0,),
        ),
    )
    return th_mld[0]


def plot_dist_th_obs(
    res_csv,
    binned_mld_csv,
    sim_config,
    inf_config,
    out_dir="/home/paulimer/Downloads",
    wrap=3,
):
    res_df = pd.read_csv(res_csv)
    with open(sim_config, "r") as fin:
        sim_cfg = yaml.safe_load(fin)
    with open(inf_config, "r") as fin:
        inf_cfg = yaml.safe_load(fin)
    binned_mlds = pd.read_csv(binned_mld_csv).rename(
        {"species_1": "genome_1", "species_2": "genome_2"}, axis=1
    )
    res_df = res_df[
        (res_df["aligner"] == "lastz_uncorrected")
        | (res_df["aligner"] == "lastz_corrected")
    ]
    res_binned_mld = pd.merge(
        res_df, binned_mlds, on=["genome_1", "genome_2", "tree_height", "aligner"]
    )
    for k, gdf in res_binned_mld.groupby(
        ["genome_1", "genome_2", "tree_height", "aligner"]
    ):
        th_mld = th_obs(
            gdf[["match_length", "freq"]],
            gdf["sim_tau"].values[0],
            sim_cfg["length_gene"] * sim_cfg["n_gene_trees"],
            gdf["empirical_mus"].values[0],
            gdf["empirical_mus"].values[0] / gdf["empirical_muc"].values[0],
            gdf["aligner"].values[0],
        )
        res_binned_mld.loc[gdf.index, "th_freq"] = np.asarray(th_mld)
    n_plots = res_binned_mld["tree_height"].drop_duplicates().shape[0]
    nrow = -(-n_plots // wrap)
    fig = plt.Figure(layout="constrained", figsize=(wrap * 4, nrow * 4))
    axes = fig.subplots(nrow, wrap).flatten()
    for ax, (k, gdf) in zip(axes, res_binned_mld.groupby("tree_height")):
        ax.scatter(gdf["match_length"], gdf["freq"], label="observed")
        for k2, g2df in gdf.groupby("aligner"):
            ax.scatter(g2df["match_length"], g2df["th_freq"], label=f"th_{k2}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{k:.2e}")
        ax.legend()
    fig.savefig(os.path.join(out_dir, "theoretical_observed_corrections.png"))

    res_binned_mld["chi_dist"] = (
        res_binned_mld["freq"] - res_binned_mld["th_freq"]
    ) ** 2 / res_binned_mld["th_freq"]

    dist_df = (
        res_binned_mld.groupby(["tree_height", "aligner"])["chi_dist"]
        .sum()
        .reset_index()
    )
    dist_df = dist_df.pivot(
        index="tree_height", columns="aligner", values="chi_dist"
    ).reset_index()
    dist_df["difference"] = dist_df["lastz_uncorrected"] - dist_df["lastz_corrected"]
    fig = plt.Figure(figsize=(6, 4), layout="constrained")
    ax = fig.subplots()
    ax.scatter(
        dist_df["tree_height"],
        dist_df["difference"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Tree height")
    ax.set_ylabel("Distance to ???")
    ax.set_ylabel(
        r"$\Delta\chi^2$ divergence from simulation"
        "\n"
        "(uncorrected − corrected theory)"
    )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    fig.savefig(os.path.join(out_dir, "correction_gain.png"), dpi=300)
    # chi dist

    return dist_df
