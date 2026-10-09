#!/usr/bin/env python3

from collections.abc import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, randint

from mosaic_method.global_fitting_reparam_kappa_inv import theoretical_mld_vectorized


def stick_breaking_exp(k, mu, tau, r):
    return k * ((mu * tau) ** 2) * np.exp(-mu * tau * r)


def stick_breaking_exp_theta(k, theta, r):
    return k * ((theta) ** 2) * np.exp(-theta * r)


def stick_breaking_poisson(k: int, theta: float, rng=None):
    """Lengths of the pieces of a stick of size k, broken at n_breaks ~ Poisson(k*theta)
    uniform random locations in [0, k[."""
    rng = rng or np.random.default_rng()
    n_breaks = rng.poisson(k * theta)
    break_locs = np.sort(rng.integers(0, k, size=n_breaks))
    endpoints = np.concatenate(([0], break_locs, [k]))
    matches = np.diff(endpoints)
    matches = matches[matches > 0]
    bins = np.arange(1, k + 1)
    hist, _ = np.histogram(matches, bins)
    return hist


def synthetic_mld(mu_array, tau, gene_length, nbins=100, delta=None):
    """
    Generate the synthetic mld from the empirical distribution of summed mutation rates.
    """
    bins = np.logspace(np.log10(min(mu_array)), np.log10(max(mu_array)), nbins)
    hist, bin_edges = np.histogram(mu_array, bins=bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    if isinstance(gene_length, Iterable):
        max_len_g = max(gene_length, key=len)
    else:
        max_len_g = gene_length
    mld = np.zeros(max_len_g)
    for mu, pmu in zip(bin_centers, hist):
        if pmu == 0:
            continue
        if delta:
            if mu * tau > delta:
                continue
        exp_mld = stick_breaking_exp(gene_length, mu, tau, np.arange(max_len_g))
        exp_mld = np.pad(
            exp_mld, (0, len(mld) - len(exp_mld)), "constant", constant_values=0
        )
        mld += pmu * exp_mld
    return mld


def synthetic_mld_theta(theta_sizes):
    """
    Generate the synthetic mld from the empirical distribution of the distribution of the pdistances.
    """
    mld = np.zeros(max(theta_sizes["size_gene"]))
    for _, row in theta_sizes.iterrows():
        exp_mld = stick_breaking_exp_theta(
            row["size_gene"], row["mutation_rate"], np.arange(row["size_gene"])
        )
        exp_mld = np.pad(
            exp_mld, (0, len(mld) - len(exp_mld)), "constant", constant_values=0
        )
        mld += exp_mld
    mld_df = pd.DataFrame({"match_length": np.arange(1, len(mld) + 1), "freq": mld})
    return mld_df


def plot_mld_syn(mld, syn_mld):
    """
    Plot the synthetic mld and the empirical mld.
    """
    plt.scatter(mld["match_length"], mld["freq"], label="Empirical MLD")
    plt.plot(syn_mld["match_length"], syn_mld["freq"], label="Synthetic MLD")
    plt.xlabel("Match Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.ylim(mld["freq"].min() / 10, mld["freq"].max() * 10)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def plot_mld(ax, mld, label):
    r = np.arange(1, len(mld) + 1)
    ax.plot(r, mld, label=label)


if __name__ == "__main__":
    if False:
        theta = np.array(
            [
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ]
        )
        xis_raw = np.array(
            [
                -2,
                -2,
                -2,
                -2,
                -2,
            ]
        )
        kappa_raw = np.array(
            [
                1000,
                100,
                50,
                10,
                5,
            ]
        )
        ml = np.array(
            [
                np.arange(1, 5000),
                np.arange(1, 5000),
                np.arange(1, 5000),
                np.arange(1, 5000),
                np.arange(1, 5000),
            ]
        )
        L0 = np.array(
            [
                5e6,
                5e6,
                5e6,
                5e6,
                5e6,
            ]
        )
        delta = 0.82

        th_mld = theoretical_mld_vectorized(
            theta, xis_raw, 0.1, ml, kappa_raw, delta, L0
        )
        fig = plt.Figure()
        ax = fig.subplots()
        for i, mld in enumerate(th_mld):
            plot_mld(ax, mld, f"kappa = {kappa_raw[i]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        fig.savefig("/home/paulimer/Downloads/test.png")
