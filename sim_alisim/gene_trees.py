#!/usr/bin/env python3


from sim_alisim.synthetic_mld import stick_breaking_poisson
import argparse
import concurrent.futures
import multiprocessing

_mp_context = multiprocessing.get_context("spawn")
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.random import default_rng
import os
import random
import io
import subprocess as sp
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from scipy.stats import norm, truncnorm, kstest, gmean, uniform, lognorm
from scipy import stats
import seaborn as sns
from skbio import TreeNode
import yaml

from mosaic_method.parsing import bin_mld
from mosaic_method.fitting import theoretical_mld

# TODO: uncorrelated : ideas : white noise process (special case of gamma, Drummond 2006), Cox - Ingersoll - Ross process (lepage 2007), mixed relaxed clock (lartillot 2016), lognormal


def to_list(x):
    """Convert x to a list: wrap non-lists, return copy of lists."""
    return [x] if not isinstance(x, list) else list(x)


# Gene tree zone ---------------------------------------------------------------
def get_time_tree(tree, total_time):
    """
    Scale a tree to represent time.

    Parameters
    ----------
    tree : TreeNode
        The tree to scale.
    total_time : float
        The total time to scale the tree to.

    Returns
    -------
    tree : TreeNode
        The scaled tree.
    """
    if not tree.is_root():
        print("Tree is not rooted")
        return tree
    else:
        current_tree_height, _ = tree.height()
        scaling_factor = total_time / current_tree_height
        time_tree = tree.copy()
        for node in time_tree.traverse(include_self=False):
            node.length *= scaling_factor
        return time_tree


def get_constant_rate_tree(tree, mu):
    """
    Get a "mutation rate" tree with a constant rate.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    mu : float
        The mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    mutation_rate_tree = tree.copy()
    for node in mutation_rate_tree.traverse(include_self=False):
        node.length = mu
    return mutation_rate_tree


def random_walk(mu, n_steps, time, mumin, mumax, linear=False, rng=None):
    """
    Generate a random walk with a given step size.

    Parameters
    ----------
    mu : float
        The initial value of the mutation rate.
    n_step : int
        the number of steps.
    time : float
        The total time of the branch.
    muc : float
        The minimum value of the mutation rate.
    mus : float
        The maximum value of the mutation rate.

    Returns
    -------
    float
        The mutation rate at the end of the branch.
    """
    rw = np.zeros(n_steps)

    lower_bound = True

    if mumin is None:
        lower_bound = False
        # just to define the step size don't worry
        mumin = mu / 100
    if not linear:
        rw[0] = np.log(mu)
        range_mu = np.log(mumax) - np.log(mumin)
        mu_min = np.log(mumin)
        mu_max = np.log(mumax)
    else:
        rw[0] = mu
        range_mu = mumax - mumin
        mu_min = mumin
        mu_max = mumax

    if rng is None:
        rng = np.random.default_rng()
    # TODO is there something better?
    sigma = range_mu / 100
    # sigma = 0.1
    for i in range(1, n_steps):
        if rng.random() > 0.5:
            rw[i] = rw[i - 1] + sigma
        else:
            rw[i] = rw[i - 1] - sigma
        if rw[i] < mu_min and lower_bound:
            rw[i] = mu_min + sigma
        elif rw[i] > mu_max:
            rw[i] = mu_max - sigma

    if not linear:
        rw = np.exp(rw)
    return np.mean(rw), rw, rw[-1]


def kishino_log_brownian(
    mu: float, nu: float, time: float, rng=None
) -> tuple[float, float]:
    """
    Makes a log Brownian motion according to Kishino et al 2001.

    mu: float
        the mutation rate at the start of the branch
    nu: float
        the autocorrelation parameter (0 => constant rate)
    time: float
        the length of the branch
    """
    if rng is None:
        rng = np.random.default_rng()
    # taking into account Jensen's inequality
    s = nu * np.sqrt(time)
    scale = mu / np.exp(s**2 / 2)
    last_mu = lognorm.rvs(s=s, scale=scale, random_state=rng)
    # end_mu = lognorm.rvs(s=nu*np.sqrt(time), loc=mu)

    mean_mu = (mu + last_mu) / 2
    return last_mu, mean_mu


def kishino_bounded_log_brownian(mu, nu, time, muc, mus, rng=None):
    """
    Makes a log Brownian motion according to Kishino et al 2001.

    mu: float
        the mutation rate at the start of the branch
    nu: float
        the autocorrelation parameter (0 => constant rate)
    time: float
        the length of the branch
    muc: float
        the minimum mutation rate
    mus: float
        the maximum mutation rate

    """
    if rng is None:
        rng = np.random.default_rng()
    if nu == 0:
        return mu, mu, 0
    scale = nu * np.sqrt(time)
    a = (np.log(muc) - np.log(mu) + scale**2 / 2) / scale
    b = (np.log(mus) - np.log(mu) + scale**2 / 2) / scale
    log_end_mu = truncnorm.rvs(
        a, b, loc=np.log(mu) - (nu**2) * time / 2, scale=scale, random_state=rng
    )
    end_mu = np.exp(log_end_mu)
    branch_mu = (mu + end_mu) / 2
    movement = mu - end_mu
    return end_mu, branch_mu, movement


def get_random_walk_tree(
    tree, n_steps, mu, mumin, mumax, mean_steps=False, return_instant=False
):
    """
    Get a "mutation rate" tree with a random walk rate variation.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    rw_step_size : float
        The size of one step of the rw.
    mu : float
        The "root" mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    rng = np.random.default_rng()
    # initialize the mutation rate tree at mu
    mutation_rate_tree = tree.copy()
    instant_mutation_rate_tree = tree.copy()
    instant_mutation_rate_tree.length = mu
    for time_node, rate_node, instant_rate_node in zip(
        tree.traverse(include_self=False),
        mutation_rate_tree.traverse(include_self=False),
        instant_mutation_rate_tree.traverse(include_self=False),
    ):
        if not mean_steps:
            mean_mu, _, last_mu = random_walk(
                instant_rate_node.parent.length,
                int(n_steps),
                time_node.length,
                mumin,
                mumax,
                True,
                rng,
            )
            rate_node.length = (instant_rate_node.parent.length + last_mu) / 2
        else:
            mean_mu, _, last_mu = random_walk(
                instant_rate_node.parent.length,
                int(n_steps),
                time_node.length,
                mumin,
                mumax,
                True,
                rng,
            )
            rate_node.length = mean_mu

        instant_rate_node.length = last_mu

    if return_instant:
        return mutation_rate_tree, instant_mutation_rate_tree
    return mutation_rate_tree


def get_kishino_tree(
    tree: TreeNode, nu: float, mu: float, return_instant: bool = False
) -> TreeNode:
    """
    Get a "mutation rate" tree with a rate variation according to Kishino et al 2001.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    nu : float
        The divergence of the log Brownian.
    mu : float
        The "root" mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    rng = np.random.default_rng()
    # initialize the mutation rate tree at mu
    mutation_rate_tree = tree.copy()
    instant_mutation_rate_tree = tree.copy()
    instant_mutation_rate_tree.length = mu
    for time_node, rate_node, instant_rate_node in zip(
        tree.traverse(include_self=False),
        mutation_rate_tree.traverse(include_self=False),
        instant_mutation_rate_tree.traverse(include_self=False),
    ):
        last_mu, mean_mu = kishino_log_brownian(
            instant_rate_node.parent.length, nu, time_node.length, rng
        )
        rate_node.length = mean_mu
        # rate_node.length = mean_mu
        instant_rate_node.length = last_mu

    if return_instant:
        return mutation_rate_tree, instant_mutation_rate_tree
    return mutation_rate_tree


def get_null_model_tree(time_tree, mu, mumin, mumax):
    """
    Get a mutation rate tree with pnas rate variation.

    Parameters
    ----------
    time_tree : TreeNode
        The time tree to copy.
    timestep : float
        The time step at which the mutation rate changes.
    mu : float
        The "root" mutation rate.
    muc: float
        The minimum value of the mutation rate.
    mus: float
        The maximum value of the mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not time_tree.is_root():
        print("Tree is not at root")
        return time_tree
    # initialize the mutation rate tree at mu
    # Use a freshly-seeded RNG (OS entropy) so forked worker processes don't
    # share the same random state and produce identical trees.
    rng = np.random.default_rng()
    mutation_rate_tree = time_tree.copy()
    mutation_rate_tree.length = mu
    for rate_node in mutation_rate_tree.traverse(include_self=False):
        rate_node.length = rng.uniform(mumin, mumax)

    return mutation_rate_tree


def generate_cherries(species_tree, n, mumin, mumax):
    """
    Generate cherries that respect Sheinman 2024 assumptions.

    Parameters
    ----------
    species_tree : TreeNode
        The 2-tipped species tree to generate cherries from.
    n : int
        The number of cherries to generate.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    list
        The list of cherries.
    """
    cherries = []
    mu_a = np.random.uniform(mumin, mumax, n)
    mu_b = np.random.uniform(mumin, mumax, n)
    for i in range(n):
        cherry = species_tree.copy()
        cherry.find("A").length = mu_a[i] * cherry.find("A").length
        cherry.find("B").length = mu_b[i] * cherry.find("B").length
        cherries.append(cherry)
    return cherries


def generate_gene_trees(
    time_tree,
    n,
    mumin=None,
    mumax=None,
    threads=1,
    fixed_mu=None,
    rate_evolution="none",
    rate_evolution_value=None,
    mean_steps=False,
    linear_mu=False,
):
    """
    Generate rate trees from a species tree.

    Each returned tree is a copy of species_tree whose branch lengths encode the mean
    mutation rate for that branch under the chosen rate evolution model.

    Parameters
    ----------
    time_tree : TreeNode
        The (time-scaled) species tree to base rate trees on.
    n : int
        The number of rate trees to generate.
    mumin : float, optional
        Lower bound on the root mutation rate drawn per tree.
    mumax : float, optional
        Upper bound on the root mutation rate drawn per tree.
    threads : int
        Number of parallel workers.
    fixed_mu : float, optional
        If provided, all trees use this fixed root mutation rate.
    rate_evolution : str
        One of "none", "random_walk", "kishino", "null".
    rate_evolution_value : float, optional
        The model-specific parameter: rw_step_size for "random_walk", nu for "kishino".
        Unused for "none" and "null".
    mean_steps : bool
        For "random_walk" only: if True, use the mean rate over steps rather than the
        last-to-first average.
    linear_mu : bool
        For the case of no rate evolution, draw the gene tree rates from a linear distrib

    Returns
    -------
    list of TreeNode
    """
    if not time_tree.is_root():
        print("Species tree is not at root")
        return []

    gene_trees = [time_tree.copy() for _ in range(n)]
    if fixed_mu:
        mu = [fixed_mu] * n
    elif linear_mu:
        dist = dist_lin(mumin, mumax)
        mu = dist.rvs(size=n)
    elif mumin:
        mu = np.random.uniform(mumin, mumax, n)
    else:
        mu = np.random.uniform(mumax / 100, mumax, n)

    if rate_evolution == "kishino":
        mutation_rate_fun = get_kishino_tree
        fun_args = [gene_trees, [rate_evolution_value] * n, mu]
    elif rate_evolution == "random_walk":
        mutation_rate_fun = get_random_walk_tree
        fun_args = [
            gene_trees,
            [rate_evolution_value] * n,
            mu,
            [mumin] * n,
            [mumax] * n,
            [mean_steps] * n,
        ]
    elif rate_evolution == "null":
        mutation_rate_fun = get_null_model_tree
        fun_args = [gene_trees, mu, [mumin] * n, [mumax] * n]
    else:  # "none"
        mutation_rate_fun = get_constant_rate_tree
        fun_args = [gene_trees, mu]

    with concurrent.futures.ProcessPoolExecutor(
        mp_context=_mp_context, max_workers=threads
    ) as ex:
        mutation_rate_trees = list(ex.map(mutation_rate_fun, *fun_args))
    return mutation_rate_trees


def linear_pdf(mu, muc, mus):
    """
    Generate a linear pdf.

    Parameters
    ----------
    mu : np.array
        The mutation rates.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    np.array
        The pdf.
    """
    return np.where((mu >= muc) & (mu <= mus), 2 * mu / (mus**2 - muc**2), 0)


def linear_cdf(mu, muc, mus):
    """
    Generate a linear cdf.

    Parameters
    ----------
    mu : np.array
        The mutation rates.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    np.array
        The cdf.
    """
    return np.where(
        mu < muc, 0, np.where(mu > mus, 1, (mu**2 - muc**2) / (mus**2 - muc**2))
    )


class dist_lin(stats.rv_continuous):
    """to sample in the linear pdf"""

    def __init__(self, x_min, x_max, seed=233423):
        super().__init__(a=x_min, b=x_max, seed=seed)
        self.x_min = x_min
        self.x_max = x_max

    def _pdf(self, x):
        return np.where(
            (x >= self.x_min) & (x <= self.x_max),
            2 * x / (self.x_max**2 - self.x_min**2),
            0,
        )

    def _cdf(self, x):
        return np.clip((x**2 - self.x_min**2) / (self.x_max**2 - self.x_min**2), 0, 1)

    def _ppf(self, q):
        return np.sqrt(q * (self.x_max**2 - self.x_min**2) + self.x_min**2)


def _path_to_lca(tree, tip_name, lca_node):
    """
    Return the branch lengths along the path from lca_node down to the named tip,
    ordered from LCA to tip.

    Parameters
    ----------
    tree : TreeNode
        The tree containing the tip.
    tip_name : str
        Name of the tip.
    lca_node : TreeNode
        The LCA node to stop at (not included in output).

    Returns
    -------
    list of float
    """
    lengths = []
    node = tree.find(tip_name)
    while node is not lca_node:
        lengths.append(node.length)
        node = node.parent
    return list(reversed(lengths))


def get_all_pair_mutation_rate(mutation_rate_trees, time_tree):
    """
    Get the time-weighted mean mutation rate from the LCA to each tip, for all pairs.

    The rate tree encodes mean mutation rates per branch; the time tree encodes branch
    durations. The effective rate for a lineage is:
        mu = sum(rate_i * time_i) / t_LCA
    where the sum is over branches on the path from LCA to tip.

    Parameters
    ----------
    mutation_rate_trees : list of TreeNode
        Rate trees whose branch lengths are mean mutation rates.
    time_tree : TreeNode
        The ultrametric time tree used to weight branches by their duration.

    Returns
    -------
    dict
        Keys are (tip_1, tip_2) tuples; values are lists of [mu_1, mu_2] pairs,
        one per rate tree.
    """
    tips = sorted([tip.name for tip in mutation_rate_trees[0].tips()])

    # Precompute time-tree path lengths and t_LCA for each pair (constant across rate trees)
    lca_info = {}
    for t1, t2 in itertools.combinations(tips, 2):
        time_lca = time_tree.lca([time_tree.find(t1), time_tree.find(t2)])
        time_1 = _path_to_lca(time_tree, t1, time_lca)
        time_2 = _path_to_lca(time_tree, t2, time_lca)
        t_lca = sum(time_1)  # ultrametric: sum(time_1) == sum(time_2)
        lca_info[(t1, t2)] = (time_lca, time_1, time_2, t_lca)

    tips_mut_rate = {}
    for rate_tree in mutation_rate_trees:
        for (t1, t2), (time_lca, time_1, time_2, t_lca) in lca_info.items():
            rate_lca = rate_tree.lca([rate_tree.find(t1), rate_tree.find(t2)])
            rates_1 = _path_to_lca(rate_tree, t1, rate_lca)
            rates_2 = _path_to_lca(rate_tree, t2, rate_lca)
            mu_1 = sum(r * t for r, t in zip(rates_1, time_1)) / t_lca
            mu_2 = sum(r * t for r, t in zip(rates_2, time_2)) / t_lca
            tips_mut_rate.setdefault((t1, t2), []).append([mu_1, mu_2])
    return tips_mut_rate


def plot_mutation_rate_distribution(
    ax, time_tree, tip_name, gene_trees, muc, mus, label="observed"
):
    """
    Plot the distribution of the time-weighted mean mutation rate from root to tip.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    time_tree : TreeNode
        The ultrametric time tree, used to weight branch durations.
    tip_name : str
        The tip name to get the mutation rate from.
    gene_trees : list
        The list of rate trees to plot the distribution of mutation rates from.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    None
    """
    time_root = time_tree
    time_branches = _path_to_lca(time_tree, tip_name, time_root)
    t_total = sum(time_branches)

    rates = []
    for gene_tree in gene_trees:
        rate_branches = _path_to_lca(gene_tree, tip_name, gene_tree)
        rates.append(sum(r * t for r, t in zip(rate_branches, time_branches)) / t_total)

    bins = np.logspace(np.log10(muc), np.log10(mus), 30)
    # hist, bins = np.histogram(rates, bins=bins, density=True)
    # bin_centers = (bins[1:] + bins[:-1]) / 2

    mu = np.linspace(muc, mus, 100)
    pdf = uniform.pdf(mu, muc, mus - muc)

    ax.hist(rates, bins, label=label, alpha=0.5, density=True)
    ax.set_xlabel("Mutation rate")
    ax.set_ylabel("Density")
    ax.set_title(f"Mutation rate distribution, tip : {tip_name}")
    ax.plot(mu, pdf, color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.5, 0.1, f"{tip_name}", transform=ax.transAxes)
    ax.legend()


def plot_rate_correlation(
    ax, time_tree, gene_trees, tip_1, tip_2, muc, mus, label="observed"
):
    """
    Plot the correlation of mutation rates between two tips.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    time_tree : TreeNode
        The ultrametric time tree, used to weight branch durations.
    gene_trees : list
        The list of rate trees to plot the distribution of mutation rates from.
    tip_1 : str
        The first tip name to get the mutation rate from.
    tip_2 : str
        The second tip name to get the mutation rate from.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    None
    """
    time_1 = _path_to_lca(time_tree, tip_1, time_tree)
    time_2 = _path_to_lca(time_tree, tip_2, time_tree)
    t_total_1 = sum(time_1)
    t_total_2 = sum(time_2)
    rates_1 = []
    rates_2 = []
    for gene_tree in gene_trees:
        rate_1 = _path_to_lca(gene_tree, tip_1, gene_tree)
        rate_2 = _path_to_lca(gene_tree, tip_2, gene_tree)
        rates_1.append(sum(r * t for r, t in zip(rate_1, time_1)) / t_total_1)
        rates_2.append(sum(r * t for r, t in zip(rate_2, time_2)) / t_total_2)

    corr = np.corrcoef(rates_1, rates_2)
    label = f"{label} - corr : {corr[0, 1]:.2f}"
    ax.scatter(rates_1, rates_2, label=label)
    ax.plot([muc, mus], [muc, mus], color="red")
    ax.set_xlabel(f"Mutation rate {tip_1}")
    ax.set_ylabel(f"Mutation rate {tip_2}")
    ax.set_title(f"Mutation rate correlation, tips : {tip_1} - {tip_2}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()


def plot_distance_distribution(
    axs,
    mutation_rate_trees,
    muc_mus,
    time_tree,
    label="observed distribution",
    tree_dir=None,
):
    """
    Plot the distribution of distances between tips of gene trees.

    Parameters
    ----------
    axs : plt.Axes
        The axes to plot on.
    mutation_rate_trees : list
        The list of rate trees to plot the distribution of distances from.
    muc_mus : dict
        The dict of muc and mus for each pair of tips.
    time_tree : TreeNode
        The ultrametric time tree, used to weight branch durations when computing rates.

    Returns
    -------
    None
    """
    tips_mut_rate = get_all_pair_mutation_rate(mutation_rate_trees, time_tree)
    n_combinations = len(tips_mut_rate)
    # or theta or mu+mu bref un autre nom
    tips_avg_rates = {
        key: [(a + b) / 2 for a, b in value] for key, value in tips_mut_rate.items()
    }

    # plot the distribution of distances/time
    for i, (pair, avg_rates) in enumerate(tips_avg_rates.items()):
        # compute time distance between tips
        muc, mus = muc_mus[pair]

        # log bin the distances
        bins = np.logspace(np.log10(muc), np.log10(mus), 20)
        hist, bins = np.histogram(avg_rates, bins=bins, density=True)
        bin_centers = np.sqrt(bins[1:] * bins[:-1])

        # linear bin the distances
        bins_lin = np.linspace(muc, mus, 30)
        hist_lin, bins_lin = np.histogram(avg_rates, bins=bins_lin, density=True)
        bin_centers_lin = (bins_lin[1:] + bins_lin[:-1]) / 2

        mu = np.logspace(np.log(muc), np.log(mus), 100, base=np.exp(1))
        pdf = linear_pdf(mu, muc, mus)
        pdf_log = np.where(
            (mu >= muc) & (mu <= mus),
            np.log(mu) / (mu * (np.log(mus) ** 2 - np.log(muc) ** 2)),
            0,
        )

        pdf_uniform = np.where((mu >= muc) & (mu <= mus), 1 / (mus - muc), 0)
        pdf_uniform_log = np.where(
            (mu >= muc) & (mu <= mus), 1 / (mu * (np.log(mus) - np.log(muc))), 0
        )

        if False:
            # correlation
            correlation = np.corrcoef(
                [a for a, _ in tips_mut_rate[pair]], [b for _, b in tips_mut_rate[pair]]
            )[0, 1]

            first_decile = np.percentile(np.array(avg_rates), 10)
            smallest_50 = [
                (a, b) for a, b in tips_mut_rate[pair] if (a + b) < first_decile
            ]
            correlation_smallest_50 = np.corrcoef(
                [a for a, _ in smallest_50], [b for _, b in smallest_50]
            )[0, 1]

            last_decile = np.percentile(np.array(avg_rates), 90)
            highest_50 = [
                (a, b) for a, b in tips_mut_rate[pair] if (a + b) > last_decile
            ]
            correlation_highest_50 = np.corrcoef(
                [a for a, _ in highest_50], [b for _, b in highest_50]
            )[0, 1]

            max_lastz_mu = 0.82 / total_time
            lastz_pairs = [
                (a, b) for a, b in tips_mut_rate[pair] if (a + b) < max_lastz_mu
            ]
            correlation_lastz = np.corrcoef(
                [a for a, _ in lastz_pairs], [b for _, b in lastz_pairs]
            )[0, 1]

        if not isinstance(axs, np.ndarray):
            ax = axs
        elif n_combinations == 1:
            ax = axs[0]
        elif len(axs.shape) == 1:
            ax = axs[i]
        elif len(axs.shape) == 2:
            ax = axs[i, 0]
        # plot the distance distribution
        ax.scatter(bin_centers, hist, label=label)
        # ax.scatter(bin_centers_lin, hist_lin, label=f"{label} linear")
        ax.set_xlabel("Effective mutation rate")
        ax.set_ylabel("Density")
        ax.plot(mu, pdf, color="tab:green", label="linear pdf")
        # ax.plot(mu, pdf_log, color="darkgreen", label="linear pdf (log)")
        ax.plot(mu, pdf_uniform, color="tab:red", label="uniform pdf")
        # ax.plot(mu, pdf_uniform_log, color="darkred", label="uniform pdf (log)")
        # ax.axvline(x=max_lastz_mu, color="purple", label="lastZ detection limit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.text(0.5, 0.1, f"{pair[0]} - {pair[1]} - {total_time:.1e}y", transform=ax.transAxes)
        # ax.text(0.1, 0.5, f"Correlation : {correlation:.2f}", transform=ax.transAxes)#"\nCorrelation smallest 10% : {correlation_smallest_50:.2f}\nCorrelation highest 10% : {correlation_highest_50:.2f}", transform=ax.transAxes)
        # ax.text(0.2, 0.5, f"Correlation under lastz limit : {correlation_lastz:.2f}", transform=ax.transAxes)#"\nCorrelation smallest 10% : {correlation_smallest_50:.2f}\nCorrelation highest 10% : {correlation_highest_50:.2f}", transform=ax.transAxes)
        ax.legend()
        ax.set_title(f"{pair[0]} vs {pair[1]}")
        # plot corresponding tree
        if n_combinations != 1 and False:
            tip_ordered = sorted(pair)
            if len(axs.shape) == 2 and os.path.exists(
                f"{tree_dir}/{tip_ordered[0]}{tip_ordered[1]}.png"
            ):
                tree_img = mpimg.imread(
                    f"{tree_dir}/{tip_ordered[0]}{tip_ordered[1]}.png"
                )
                axs[i, 0].imshow(tree_img)
                axs[i, 0].axis("off")

            # plot the same tree for all combinations
            elif os.path.exists(f"{tree_dir}/tree.png"):
                tree_img = mpimg.imread(f"{tree_dir}/tree.png")
                axs[i, 0].imshow(tree_img)
                axs[i, 0].axis("off")

        # add ks_stat to plot
        # ax.text(0.5, 0.1, f"KS stat : {ks_stat:.2f}", transform=ax.transAxes)
        # ax.text(0.5, 0.15, f"KS where : {ks_where:.2f}", transform=ax.transAxes)


def get_pseudo_empirical(tips_mut_rate, time_tree, gene_length):
    """
    Based on pairwise mutation rates distribution, return pairwise pseudo-empirical binned mlds.

    tips_mut_rate: dict
        The pairwise average mutation rates for each side of the last common ancestor of each pair of gene tree leaves.
    time_tree: TreeNode
        The species tree with branch lengths as time.
    gene_length: int
        The length of genes in simulations.
    """

    avg_mus = {
        key: [(a + b) / 2 for a, b in value] for key, value in tips_mut_rate.items()
    }
    binned_mld_dic = {}
    for pair, rates in avg_mus.items():
        mlds = np.zeros(int(gene_length))
        time_lca = time_tree.lca([time_tree.find(pair[0]), time_tree.find(pair[1])])
        tau = time_lca.height()[0] * 2  # times two because two branches
        for rate in rates:
            mld = stick_breaking_poisson(gene_length, tau * rate)
            mld = np.pad(mld, (0, len(mlds) - len(mld)), "constant", constant_values=0)
            mlds += mld
        summed_mld = pd.DataFrame(mlds, columns=["freq"]).reset_index(
            names="match_length"
        )
        summed_mld["match_length"] += 1
        binned_mld_dic[pair] = bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=1,
        )
    return binned_mld_dic


def plot_pseudo_empirical(axs, tips_mut_rate, time_tree, gene_length):
    # TODO rewrite with get_pseudo_empirical above
    """
    Based on pairwise mutation rates distribution, plots a pseudo-empirical mld.

    axs: plt.Axes
        The axes to plot on
    tips_mut_rate: dict
        The pairwise average mutation rates for each side of the last common ancestor of each pair of gene tree leaves.
    time_tree: TreeNode
        The species tree with branch lengths as time.
    gene_length: int
        The length of genes in simulations.
    """

    avg_mus = {
        key: [(a + b) / 2 for a, b in value] for key, value in tips_mut_rate.items()
    }
    r = np.arange(1, gene_length + 1)
    muc_mus = {pair: [min(avg_mu), max(avg_mu)] for pair, avg_mu in avg_mus.items()}
    for i, (pair, rates) in enumerate(avg_mus.items()):
        ax = axs[i]
        mlds = np.zeros(int(gene_length))
        time_lca = time_tree.lca([time_tree.find(pair[0]), time_tree.find(pair[1])])
        tau = time_lca.height()[0] * 2  # times two because two branches
        for rate in rates:
            mld = stick_breaking_exp(gene_length, rate, tau, r)
            mld = np.pad(mld, (0, len(mlds) - len(mld)), "constant", constant_values=0)
            mlds += mld
        summed_mld = pd.DataFrame(mlds, columns=["freq"]).reset_index(
            names="match_length"
        )
        summed_mld["match_length"] += 1
        binned_mld = bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=len(rates),
        )
        _, th_mc = theoretical_mld(
            [np.log10(tau), -20],
            0.1,
            r,
            muc_mus[pair][1],
            muc_mus[pair][0],
            0.85,
            gene_length,
            False,
        )
        ax.scatter(
            binned_mld["match_length"],
            binned_mld["freq"],
            color="black",
            label="pseudo-empirical",
        )
        ax.plot(r, th_mc, label="mc - expected", color="green")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"MLD fit for {pair[0]} vs {pair[1]}")
        ax.legend()
        ax.set_xlabel("Match length")
        ax.set_ylabel("Normalized count")


def run_alisim(gene_tree, outdir, length_gene, num, seed):
    """
    Create sequences from Alisim for a gene tree.

    Parameters
    ----------
    gene_tree : TreeNode
        The gene tree to simulate.
    outdir : str
        The output directory.
    num : int
        The index of the gene tree.
    seed: int
        Seed for this alisim run.

    Returns
    -------
    """
    os.makedirs(outdir, exist_ok=True)
    output_prefix = os.path.join(outdir, f"gene_tree_{num}")
    tree_path = os.path.join(outdir, f"gene_tree_{num}.newick")
    gene_tree.write(tree_path)
    # TODO why is seed obligatory now
    alisim_cmd = [
        "iqtree3",
        "--alisim",
        output_prefix,
        "-t",
        tree_path,
        "-m",
        "JC",
        "--out-format",
        "fasta",
        "--length",
        str(length_gene),
        "--seed",
        str(seed),
    ]
    try:
        sp.run(alisim_cmd, check=True, capture_output=True)
    except sp.CalledProcessError as e:
        print("Error with command")
        print(" ".join(alisim_cmd))
        raise e

    os.remove(tree_path)
    os.remove(f"{tree_path}.log")


def run_alisim_trees(time_tree, mutation_rate_trees, outdir, length_gene, threads=1):
    """
    Run Alisim on all the gene trees in a parallel.

    Parameters
    ----------

    time_tree:
        The ultrametric time tree.
    mutation_rate_trees : list
        A list of rate trees to generate sequences of.
    outdir: str
        The output directory.
    length_gene : int
        The length of a gene.
    threads : int
        The number of threads to use.

    Returns
    -------
    """
    rng = default_rng()
    seeds = rng.choice(
        len(mutation_rate_trees) * 2, size=len(mutation_rate_trees), replace=False
    )
    gene_trees = [time_tree.copy() for _ in range(len(mutation_rate_trees))]
    for gene_tree, mutation_rate_tree in zip(gene_trees, mutation_rate_trees):
        for gene_node, mutation_rate_node in zip(
            gene_tree.traverse(include_self=False),
            mutation_rate_tree.traverse(include_self=False),
        ):
            gene_node.length = gene_node.length * mutation_rate_node.length

    # Save mutation-rate trees before running AliSim so genomes_dir is self-contained and
    # simulation can be skipped on re-runs by loading these trees and recomputing tips_mut_rate.
    os.makedirs(outdir, exist_ok=True)
    nwk_path = os.path.join(outdir, "mutation_rate_trees.nwk")
    with open(nwk_path, "w") as fh:
        for rate_tree in mutation_rate_trees:
            buf = io.StringIO()
            rate_tree.write(buf)
            fh.write(buf.getvalue())

    # run alisim
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=_mp_context, max_workers=threads
    ) as executor:
        list(
            executor.map(
                run_alisim,
                gene_trees,
                itertools.repeat(outdir),
                itertools.repeat(length_gene),
                range(len(gene_trees)),
                seeds,
            )
        )

    # concatenate fasta
    # and create taxon_csv
    taxon_dic = {"genome": [], "clade": []}
    seq_dic = {node.name: Seq("") for node in gene_trees[0].tips()}
    for i in range(len(gene_trees)):
        seq_path = f"{outdir}/gene_tree_{i}.fa"
        seq_recs = list(SeqIO.parse(seq_path, "fasta"))
        # test all sequences correspond to tips
        assert set([seq.id for seq in seq_recs]) == set(seq_dic.keys())
        for seq in seq_recs:
            seq_dic[seq.id] += seq.seq
        os.remove(seq_path)
    for name, seq in seq_dic.items():
        rec = SeqRecord(seq, id=name, description="")
        SeqIO.write(rec, f"{outdir}/{name}.fasta", "fasta")
        taxon_dic["genome"].append(f"{name}.fasta")
        taxon_dic["clade"].append(name)
    taxon_df = pd.DataFrame(taxon_dic)
    taxon_df.to_csv(f"{outdir}/taxon.csv")


# HGT zone ---------------------------------------------------------------------


def set_color(self, color):
    """
    Set the color of the node.

    Parameters
    ----------
    self : TreeNode
        The node to set the color of.
    color : str
        The color to set the node to.

    Returns
    -------
    None
    """
    self.color = color


def get_color(self):
    """
    Get the color of the node.

    Parameters
    ----------
    self : TreeNode
        The node to get the color of.

    Returns
    -------
    str
        The color of the node.
    """
    return self.color


def get_timeline(self, timeline, depth):
    """
    Get the timeline of the tree.

    The timeline records the depth of each node in the tree along the time axis.

    Parameters
    ----------
    self : TreeNode
        The tree to get the timeline of.
    depth : float
        The depth of the tree.

    Returns
    -------
    list
        The timeline of the tree.
    """
    timeline.append(depth)
    if self.is_tip():
        pass
    else:
        for child in self.children:
            child.get_timeline(timeline, depth + child.length)


def make_nodes_compatible(self, timeline, colors, comp_node_dic, depth):
    """
    Creates new nodes where fit and add relevant nodes to the dictionary of compatible nodes.
    The comp node dictionary is a dictionary of lists of compatible nodes.
    The keys are the colors.

    Parameters
    ----------
    self : TreeNode
        The tree to paint (monkey patched).
    timeline : list
        The timeline of the tree.
    colors : dict
        The dictionary of colors (keys : depth).
    comp_node_dic : dict
        The dictionary of compatible nodes.
    depth : float
        The depth of the node.

    Returns
    -------
    None
    """
    try:
        time_index = timeline.index(depth)
    except ValueError:
        print(f"Depth {depth} not in timeline")
        return None

    if self.is_tip():
        return

    else:
        # this is to prevent the self.children list from changing during the loop
        # which happens when a new node is created
        # I hate this behavior
        children_copy = self.children.copy()
        for child in children_copy:
            if child.length + depth > timeline[time_index + 1]:
                self.remove(child)
                new_node_length = timeline[time_index + 1] - depth
                new_node = TreeNode(
                    f"nn_{child.name}_{timeline[time_index + 1]:.0f}",
                    new_node_length,
                    children=[child],
                )
                child.length -= new_node_length
                self.append(new_node)
                comp_node_dic[timeline[time_index + 1]].append(new_node)
                new_node.make_nodes_compatible(
                    timeline, comp_node_dic, depth + new_node_length
                )
            else:
                comp_node_dic[timeline[time_index + 1]].append(child)
                child.make_nodes_compatible(
                    timeline, comp_node_dic, depth + child.length
                )


def delete_branch(self, child, comp_node_dic):
    """
    Delete the child branch : remove all single child nodes in the child branch.
    Take care of compatible node dic.
    Returns the root of the tree in case the root is deleted.

    Parameters
    ----------
    self : TreeNode
        The parent node.
    child : TreeNode
        The child node to prune.
    comp_node_dic : dict
        The dictionary of compatible nodes.

    Returns
    -------
    TreeNode
        The root of the tree.
    """
    branch_tips = child.tips()
    if not child in self.children:
        print(f"{child.name} is not a child of {self.name}")
        return None
    if len(branch_tips) > 1:
        print(f"Branch has more than one tip")
        return None
    else:
        nodes_to_delete = [node for node in child.traverse()]
        for color, comp_nodes in comp_node_dic.items():
            for node in nodes_to_delete:
                if node in comp_nodes:
                    comp_nodes.remove(node)
        self.remove(child)

    if self.is_root():
        # case where the transfer has happened from the outgroup
        # the root has only one child and needs to be removed
        new_root = self.children[0]
        new_root.parent = None
        new_root.length = 0
        # TODO self still exists ?
        self.children = []
        return new_root
    else:
        return self.root()


def homologous_recombination(self, comp_node_dic):
    """
    Choose one of the compatible nodes to recombine with.
    Technically self is the node that will receive the transfer.
    Recombination in this case is subtree pruning and regrafting with a random compatible node.

    Parameters
    ----------
    self : TreeNode
        The node to recombine.
    comp_node_dic : dict
        The dictionary of compatible nodes.

    Returns
    -------
    TreeNode
        The root of the tree.
    """
    if self.compatible_nodes:
        # choose random compatible node
        recombination_branch = random.choice(self.compatible_nodes)

        # choose where to recombine along the branch
        # this is the distance from the tip of the branch (0 is the tip, branch.length is the root of the branch)
        recombination_point = random.uniform(0, recombination_branch.length)
        # create the new node
        # distance from the tip of the recombination point to the root of the branch
        recomb_node_length = recombination_branch.length - recombination_point
        recombination_node = TreeNode(
            f"hr_{self.name}_{recombination_branch.name}", recomb_node_length
        )
        # set relationships
        recomb_parent = recombination_branch.parent
        recombination_node.append(recombination_branch)
        recomb_parent.append(recombination_node)
        # distance from the tip of the recombination branch to the recombination point
        recombination_branch.length = recombination_point

        # TODO delete single child nodes above self so that compatibility nodes are not left in the tree
        current_parent = self.parent
        current_child = self
        while len(current_parent.children) == 1:
            current_parent = current_parent.parent
            current_child = current_child.parent
        delete_branch(current_parent, current_child, comp_node_dic)

        recombination_node.append(self)
        self.length = recombination_point
        return self.root()
    else:
        print(f"No compatible nodes for {self.name}")
        return self.root()


def run_simulation(cfg, ax=None):
    """
    Run a simulation of a species tree.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    ax : plt.Axes
        The axes to plot the distance distribution on.

    Returns
    -------
    float
        The minimum mutation rate.
    """
    mumax = float(cfg["mumax"])
    mumin = cfg["mumin"]
    print(f"mumin : {mumin}, mumax : {mumax}")
    try:
        mumin = float(mumin)
    except ValueError:
        mumin = None
    tree_height = float(cfg["tree_height"])
    rate_evolution = cfg["rate_evolution"]
    if rate_evolution == "lognormal":
        sys.exit("not implemented anymore")
    elif rate_evolution not in ("random_walk", "kishino", "null", "none"):
        print("Unknown rate evolution")
        return {}

    rate_evolution_value = None
    if rate_evolution == "random_walk":
        rate_evolution_value = float(cfg["n_steps"])
    elif rate_evolution == "kishino":
        rate_evolution_value = float(cfg["nu"])

    os.makedirs(cfg["outdir"], exist_ok=True)

    species_tree = TreeNode.read([cfg["species_tree"]])
    time_tree = get_time_tree(species_tree, tree_height)

    mutation_rate_trees = generate_gene_trees(
        time_tree=time_tree,
        n=cfg["n_gene_trees"],
        mumin=mumin,
        mumax=mumax,
        threads=cfg["threads"],
        rate_evolution=rate_evolution,
        rate_evolution_value=rate_evolution_value,
        mean_steps=bool(cfg.get("mean_steps", False)),
        linear_mu=cfg["linear_mu"],
    )

    n_combinations = len(
        list(itertools.combinations([tip.name for tip in time_tree.tips()], 2))
    )
    tips_mut_rate = get_all_pair_mutation_rate(mutation_rate_trees, time_tree)

    run_alisim_trees(
        time_tree,
        mutation_rate_trees,
        cfg["genomes_dir"],
        cfg["length_gene"],
        threads=cfg["threads"],
    )

    avg_mus = {}
    for pair, rates in tips_mut_rate.items():
        avg_mus[pair] = [(a + b) / 2 for a, b in rates]
    muc_mus = {pair: [min(avg_mu), max(avg_mu)] for pair, avg_mu in avg_mus.items()}

    if ax is not None:
        plot_distance_distribution(
            ax, mutation_rate_trees, muc_mus, time_tree, tree_dir="test_tree_repr"
        )
    else:
        return tips_mut_rate
        fig, axs = plt.subplots(n_combinations, 1, figsize=(10, 10))
        plot_distance_distribution(axs, time_tree, gene_trees, plot_muc, plot_mus)
        fig.tight_layout()
        fig.savefig(f"{cfg['outdir']}/distance_distribution.png")

    return tips_mut_rate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulate gene trees.")
    parser.add_argument("config", help="The yaml configuration file.")
    parser.add_argument("--nvruse", help="Trick.", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_simulation(cfg)

    if args.nvruse:
        muc = 1e-12
        mus = 1e-8
        n_genes = 1e5

        # testing plot_distance_distribution
        # and plot_mutation_rate_distribution
        species_tree = TreeNode.read(["(C:2,(A:1,B:1):1);"])
        time_tree = get_time_tree(species_tree, 1e8)
        rate_trees = []
        for _ in range(int(n_genes)):
            rate_trees.append(
                get_null_model_tree(time_tree, np.random.uniform(muc, mus), muc, mus)
            )
        fig, axs = plt.subplots(3, 1)

        tips_mut_rate = get_all_pair_mutation_rate(rate_trees, time_tree)
        mean_rates = {
            key: [(a + b) / 2 for a, b in value] for key, value in tips_mut_rate.items()
        }
        muc_mus = {pair: [min(muss), max(muss)] for pair, muss in mean_rates.items()}
        plot_distance_distribution(axs, rate_trees, muc_mus, time_tree)

        # all_mus = np.exp(np.random.uniform(np.log(muc), np.log(mus), size=int(n_genes)))
        all_mus = [1e-10] * int(n_genes)
        # all_mus = np.random.uniform(muc, mus, size=int(n_genes))
        cherry_tau = 1
        nus = np.logspace(-6, 0, 6)
        # log normal test
        end_mu_dic = {"rvs": [], "nu": []}
        for nu in nus:
            m = 1e-10
            s = nu * np.sqrt(cherry_tau)
            scale = m / np.exp(s**2 / 2)
            end_mu_dic["rvs"] += list(lognorm.rvs(s=s, scale=scale, size=len(all_mus)))
            end_mu_dic["nu"] += [nu] * len(all_mus)
        lognorm_df = pd.DataFrame.from_dict(end_mu_dic)
        for nu, nu_df in lognorm_df.groupby("nu"):
            print(f"{nu} : mean {nu_df['rvs'].mean()}, std: {nu_df['rvs'].std()}")

        g = sns.FacetGrid(
            data=lognorm_df, col="nu", col_wrap=3, sharex=False
        )  # , hue="empirical")
        g.map_dataframe(sns.histplot, x="rvs", bins=40)
        plt.show()

        # Kishino test
        last_mu_nus = []
        for nu in nus:
            for mu in all_mus:
                end_mu, _ = kishino_log_brownian(mu, nu, cherry_tau)
                last_mu_nus.append({"nu": nu, "endmu": end_mu})

        last_mu_df = pd.DataFrame(last_mu_nus)
        for nu, nu_df in last_mu_df.groupby("nu"):
            print(f"{nu} : {nu_df['endmu'].std()}")

        # sns.displot(data=last_mu_df, x="endmu", col="nu", col_wrap=3, log_scale=True)#, kind="kde")
        g = sns.FacetGrid(
            data=last_mu_df, col="nu", col_wrap=3, sharex=False
        )  # , hue="empirical")
        g.map_dataframe(plt.hist, x="endmu", bins=40)
        plt.show()

        # Kishino correlation/sum
        muc = 6e-13
        mus = 1e-9
        n_genes = 1e3
        all_mus = [np.exp(np.random.uniform(np.log(muc), np.log(mus)))] * int(n_genes)
        cherry_tau = 1e8
        nus = np.logspace(-10, -4, 6)
        # nu = np.linspace(10**-4.2, 10**-3.8, 8)
        branch_mu_nus = []
        for nu in nus:
            for mu in all_mus:
                # _, branch_mu_1, _ = kishino_bounded_log_brownian(mu, nu, cherry_tau, muc, mus)
                # _, branch_mu_2, _ = kishino_bounded_log_brownian(mu, nu, cherry_tau, muc, mus)
                _, branch_mu_1 = kishino_log_brownian(mu, nu, cherry_tau)
                _, branch_mu_2 = kishino_log_brownian(mu, nu, cherry_tau)
                branch_mu_nus.append(
                    {"nu": nu, "mu_1": branch_mu_1, "mu_2": branch_mu_2}
                )
        branch_mu_df = pd.DataFrame(branch_mu_nus)
        branch_mu_df["summed_mu"] = branch_mu_df["mu_1"] + branch_mu_df["mu_2"]
        # sns.displot(data=branch_mu_df, x="summed_mu", col="nu", col_wrap=3, log_scale=(True, True))#, stat="density")#, kind="kde")

        g = sns.FacetGrid(data=branch_mu_df, col="nu", col_wrap=3)  # , hue="empirical")
        g.map_dataframe(plt.hist, x="summed_mu", bins=40)
        g.map(plt.axvline, x=all_mus[0], ls="--", c="red")
        plt.show()

        # branch_mu_df = branch_mu_df.melt(id_vars=["nu"], value_vars=["mu_1", "mu_2"], value_name="Mutation rate")

        # sns.displot(data=branch_mu_df, x="mu", col="nu", hue= col_wrap=3, log_scale=True)#, kind="kde")
        g = sns.PairGrid(branch_mu_df.drop("summed_mu", axis=1), hue="nu")
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot, alpha=0.5)
        g.add_legend()

        plt.show()

        # beta = (np.log(mus) - np.log(muc)) / cherry_tau
        rw_steps = [1e3, 1e4, 1e5]
        # fixed_mu = np.random.uniform(muc, mus)
        cherry_species_tree = TreeNode.read([f"(A:{cherry_tau/2},B:{cherry_tau/2});"])
        fig2, ax2 = plt.subplots(len(rw_steps), 1, figsize=(10, 15))
        for i, rw in enumerate(rw_steps):
            rw_step = cherry_tau / rw
            print(f"rw : {rw}")
            cherries = generate_gene_trees(
                cherry_species_tree, 1000, muc, mus, rw_step=rw_step
            )
            tips_mut_rate = get_all_pair_mutation_rate(cherries, cherry_species_tree)
            thetas = {}
            for pair, rates in tips_mut_rate.items():
                thetas[pair] = [(a + b) for a, b in rates]
            muc_mus = {pair: [min(theta), max(theta)] for pair, theta in thetas.items()}
            # for i, tip in enumerate(cherry_species_tree.tips()):
            # plot_mutation_rate_distribution(ax2[i], cherry_species_tree, tip.name, cherries, muc, mus, label=f"random walk steps ({rw:.0e})")
            # def plot_distance_distribution(axs, species_tree, gene_trees, muc_mus, label="observed distribution", tree_dir=None):

            plot_distance_distribution(
                ax2[i],
                cherry_species_tree,
                cherries,
                muc_mus,
                label=f"random walk steps ({rw:.0e})",
            )
            # plot_rate_correlation(ax3, cherry_species_tree, cherries, "A", "B", muc, mus, label=f"random walk steps ({rw:.0e})")

        fig2.tight_layout()
        plt.show()

        fig2.savefig("distance_distrib_poster.png")

        # rate evolution test
        tree = TreeNode.read(["(((C:0.303,D:0.303)5:0.104,B:0.407)2:0.369,A:0.776);"])
        tree_height = 1e8
        beta_fraction = [10, 1, 1e-5]
        rw_step_fraction = [1e2, 1e3, 1e4]
        timestep_fraction = [5, 10, 20]

        time_tree = get_time_tree(tree, tree_height)

        tree_plot_dir = "test_tree_repr"
        n_combinations = len(
            list(itertools.combinations([tip.name for tip in time_tree.tips()], 2))
        )

        # random walk test
        fig, axs = plt.subplots(n_combinations, 2, figsize=(20, 20))
        for rwf in rw_step_fraction:
            rw_step = tree_height / rwf
            gene_trees = generate_gene_trees(time_tree, 5000, muc, mus, rw_step=rw_step)
            plot_distance_distribution(
                axs,
                time_tree,
                gene_trees,
                muc,
                mus,
                label=f"random walk ({rwf:.0e})",
                tree_dir=tree_plot_dir,
            )
        for ax in axs:
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("random_walk_rate_evolution2.png", dpi=300)

        # timestep/null model test
        fig, axs = plt.subplots(n_combinations, 2, figsize=(20, 20))
        for tsf in timestep_fraction:
            timestep = tree_height / tsf
            gene_trees = generate_gene_trees(
                time_tree, 5000, muc, mus, timestep=timestep
            )
            plot_distance_distribution(
                axs,
                time_tree,
                gene_trees,
                muc,
                mus,
                label=f"null model ({tsf:.0f})",
                tree_dir=tree_plot_dir,
            )
        for ax in axs:
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("null_model_rate_evolution.png", dpi=300)

        # test random walk
        n_rw = 20
        muc = 6e-13
        mus = 5e-9
        for _ in range(n_rw):
            mean, rw, _ = random_walk(
                np.random.uniform(muc, mus), 1e5, 1e8, muc, mus, linear=False
            )
            plt.plot(rw)
        plt.axhline(y=mus, color="blue")
        plt.axhline(y=muc, color="blue")
        plt.yscale("log")
        plt.savefig("random_walk_pres.png", dpi=300)
        plt.close()

        # test random walk distr
        muc = 6e-11
        mus = 5e-9
        n_rw = 1e4
        rw_steps_size = [1e2, 1e3, 1e4]
        tau = 1e8
        fig, ax = plt.subplots()
        for rw_step in rw_steps_size:
            rw_distr = []
            for _ in range(int(n_rw)):
                mean, rw, last_rw = random_walk(
                    np.random.uniform(muc, mus),
                    tau / rw_step,
                    1e8,
                    muc,
                    mus,
                    linear=False,
                )
                rw_distr.append(np.exp(last_rw))
            bins = np.logspace(np.log(muc), np.log(mus), 30, base=np.exp(1))
            hist, _ = np.histogram(rw_distr, bins=bins)  # , density=True)
            bins_centers = np.sqrt(bins[1:] * bins[:-1])
            hist_norm = hist / (bins[1:] - bins[:-1])
            ax.plot(bins_centers, hist_norm, label=f"number of steps : {rw_step:.0e}")
            # ax.plot(bins[:-1], hist, label=f"number of steps : {rw_step:.0e}")
        ax.legend()

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mutation rate")
        ax.set_ylabel("Density")
        # fig.savefig("random_walk_mean_distr.png", dpi=300)
        plt.show()

        # exp muc random walk distr
        muc = 6e-11
        mus = 5e-9
        tau = 2e8
        rw_steps = tau / np.logspace(2, 5, 10)
        n_rw = 5e3
        fig, ax = plt.subplots()
        exp_muc_list = []
        batch_size = 500
        for rw_step in rw_steps:
            rw_distr = []
            for i in range(2):
                rw_distr.append([])
                for _ in range(0, int(n_rw / 2), batch_size):
                    with concurrent.futures.ProcessPoolExecutor(
                        mp_context=_mp_context, max_workers=10
                    ) as executor:
                        res = list(
                            executor.map(
                                random_walk,
                                [
                                    np.random.uniform(muc, mus)
                                    for _ in range(batch_size)
                                ],
                                itertools.repeat(rw_step),
                                itertools.repeat(tau),
                                itertools.repeat(muc),
                                itertools.repeat(mus),
                                itertools.repeat(False),
                            )
                        )
                    for mean, _, _ in res:
                        rw_distr[i].append(mean)
            summed_rw_distr = np.array(rw_distr[0]) + np.array(rw_distr[1])
            exp_muc = np.min(summed_rw_distr / 2)
            exp_muc_list.append({"step_size": rw_step, "exp_muc": exp_muc})
        exp_muc_df = pd.DataFrame(exp_muc_list)
        exp_muc_df["rw_step"] = exp_muc_df["step_size"].apply(lambda x: tau / x)

        fig, ax = plt.subplots()
        ax.scatter(exp_muc_df["rw_step"], exp_muc_df["exp_muc"], label="empirical muc")
        ax.hlines(y=muc, xmin=0, xmax=1e5, label="muc")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Mutation rate")
        ax.legend()
        fig.savefig("exp_muc_rw_steps.png", dpi=300)

        # range mu random walk distr
        muc = 6e-11
        mus = 5e-9
        tau = 2e8
        number_of_steps = np.logspace(2, 5, 30)
        rw_steps_size = tau / number_of_steps
        n_rw = 5e3
        fig, ax = plt.subplots()
        exp_mu_list = []
        batch_size = 500
        for rw_step in rw_steps_size:
            rw_distr = []
            start_mus = np.random.uniform(muc, mus, batch_size)
            for i in range(2):
                rw_distr.append([])
                for _ in range(0, int(n_rw), batch_size):
                    with concurrent.futures.ProcessPoolExecutor(
                        mp_context=_mp_context, max_workers=10
                    ) as executor:
                        res = list(
                            executor.map(
                                random_walk,
                                start_mus,
                                [rw_step] * batch_size,
                                [tau] * batch_size,
                                [muc] * batch_size,
                                [mus] * batch_size,
                                [False] * batch_size,
                            )
                        )
                    for mean, _, _ in res:
                        rw_distr[i].append(mean)
            summed_rw_distr = np.array(rw_distr[0]) + np.array(rw_distr[1])
            correlation = np.corrcoef(rw_distr[0], rw_distr[1])[0, 1]
            exp_muc = np.min(summed_rw_distr / 2)
            exp_mus = np.max(summed_rw_distr / 2)
            exp_mu_list.append(
                {
                    "step_size": rw_step,
                    "exp_muc": exp_muc,
                    "exp_mus": exp_mus,
                    "correlation": correlation,
                }
            )
        exp_mu_df = pd.DataFrame(exp_mu_list)
        exp_mu_df["steps_nb"] = exp_mu_df["step_size"].apply(lambda x: tau / x)
        exp_mu_df["mu_ratio"] = exp_mu_df["exp_mus"] / exp_mu_df["exp_muc"]

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax1.scatter(exp_mu_df["steps_nb"], exp_mu_df["mu_ratio"], label="mu_max/mu_min")
        ax2.scatter(
            exp_mu_df["steps_nb"], exp_mu_df["correlation"], label="correlation"
        )
        ax1.hlines(y=10, xmin=0, xmax=1e5, label="mu_max/mu_min = 10", color="tab:blue")
        ax2.hlines(
            y=0.1, xmin=0, xmax=1e5, label="correlation = 0.1", color="tab:orange"
        )
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_xlabel("Number of steps")
        ax2.set_xlabel("Number of steps")
        ax1.set_ylabel("mu_max/mu_min")
        ax2.set_ylabel("Correlation")
        ax1.legend()
        ax2.legend()

        fig1.savefig("exp_mu_rw_range.png", dpi=300)
        fig2.savefig("exp_mu_rw_corr.png", dpi=300)

        # test lognormal distr
        muc = 6e-11
        mus = 5e-9
        time_tree_height = 2e8
        test_tree = TreeNode.read(["(A:1,B:1);"])
        time_tree = get_time_tree(test_tree, time_tree_height)
        betas = np.logspace(-6, 2, 5)
        tips_mut_distr = {beta: {} for beta in betas}
        fig, axs = plt.subplots(len(betas), 3, figsize=(10, 10))
        for i, beta in enumerate(betas):
            gene_trees = generate_gene_trees(
                time_tree, 1000, muc, mus, beta=beta, fixed_mu=1e-9
            )
            tips_mut_distr[beta] = get_all_pair_mutation_rate(gene_trees, time_tree)
            hist_bins = np.logspace(np.log10(muc), np.log10(mus), 30)
            ax = axs[i, 0]
            hist, bins = np.histogram(
                [a for _, value in tips_mut_distr[beta].items() for a, _ in value],
                bins=hist_bins,
                density=True,
            )
            bins_centers = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(bins_centers, hist, label=f"beta : {beta:.2e}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            ax.set_title(f"A, beta : {beta:.2e}")
            ax.set_xlabel("Mutation rate")
            ax.set_ylabel("Density")
            ax = axs[i, 1]
            hist, bins = np.histogram(
                [b for _, value in tips_mut_distr[beta].items() for _, b in value],
                bins=hist_bins,
                density=True,
            )
            bins_centers = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(bins_centers, hist, label=f"beta : {beta:.2e}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            ax.set_title(f"B, beta : {beta:.2e}")
            ax.set_xlabel("Mutation rate")
            ax.set_ylabel("Density")
            plot_distance_distribution(axs[i, 2], time_tree, gene_trees, muc, mus)
        fig.tight_layout()
        plt.show()

        # playing with truncnorm alone
        muc = -1
        mus = 1
        loc = 0
        scale = 0.5
        rv = truncnorm((muc - loc) / scale, (mus - loc) / scale, loc=loc, scale=scale)
        fig, ax = plt.subplots()
        bins = np.linspace(muc, mus, 30)
        rs = rv.rvs(100000)
        hist, bins = np.histogram(rs, bins=bins, density=True)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        ax.bar(bin_centers, hist, width=bins[1:] - bins[:-1], alpha=0.5)
        ax.set_xlim(muc - 1, mus + 1)
        plt.show()

        # playing with truncnorm and logs
        fig, ax = plt.subplots(1, 2)
        ax1, ax2 = ax
        muc = 6e-11
        mus = 5e-9
        loc = np.log(mus)
        scale = 10
        rv = truncnorm(
            (np.log(muc) - loc) / scale,
            (np.log(mus) - loc) / scale,
            loc=loc,
            scale=scale,
        )
        x = np.linspace(np.log(muc), np.log(mus), 100)
        bins = np.linspace(np.log(muc), np.log(mus), 30)
        rs = rv.rvs(100000)
        # rs_uniform = np.random.uniform(np.log(muc), np.log(mus), 100000)
        ax1.hist(rs, bins=bins, alpha=0.5, density=True, label="truncnorm")
        # ax1.hist(rs_uniform, bins=bins, alpha=0.5, density=True, label="uniform")
        ax1.axvline(x=np.log(muc), color="blue")
        ax1.axvline(x=np.log(mus), color="blue")
        ax1.plot(x, rv.pdf(x), color="red")
        ax1.legend()
        # ax1.set_xlim(muc/10, mus*10)
        ax2.hist(
            np.exp(rs),
            bins=np.logspace(np.log(muc), np.log(mus), 30, base=np.exp(1)),
            alpha=0.5,
            density=True,
            label="truncnorm",
        )
        # ax2.hist(np.exp(rs_uniform), bins=np.logspace(np.log(muc), np.log(mus), 30, base=np.exp(1)), alpha=0.5, density=True, label="uniform")
        ax2.axvline(x=muc, color="blue")
        ax2.axvline(x=mus, color="blue")
        ax2.legend()
        ax2.set_xscale("log")
        # ax2.set_yscale("log")
        fig.savefig("truncnorm_log_large_scale.png", dpi=300)

        # comparing normal distribution and random walk
        tau = 1e8
        n_steps = 1e4
        n_rw = 1000
        muc = 6e-11
        mus = 5e-9
        start_mu = (muc + mus) / 2
        range_mu = mus - muc
        mean_rws = []
        for _ in range(n_rw):
            mean, rw, _ = random_walk(start_mu, n_steps, tau, muc, mus, linear=False)
            mean_rws.append(mean)
        mean_rws = np.array(mean_rws)
        scale = range_mu / 8
        rvs = truncnorm(
            (muc - start_mu) / scale,
            (mus - start_mu) / scale,
            loc=start_mu,
            scale=scale,
        ).rvs(n_rw)
        fig, ax = plt.subplots()
        bins = np.logspace(np.log10(muc), np.log10(mus), 30)
        hist, bins = np.histogram(mean_rws, bins=bins, density=True)
        ax.hist(rvs, bins=bins, alpha=0.5, density=True, label="normal")
        ax.hist(mean_rws, bins=bins, alpha=0.5, density=True, label="random walk")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(x=muc, color="blue")
        ax.axvline(x=mus, color="blue")
        ax.axvline(x=start_mu, color="green")
        ax.legend()
        fig.savefig("normal_vs_rw_log.png", dpi=300)

        # test plot_distance_distribution for rw corr 1e5 steps
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i, steps in enumerate([3]):
            print(f"steps : {steps}, i : {i}")
            gene_trees = []
            gene_tree_dir = f"/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/corr/rw_cherry_rw_1.00e+0{steps}"
            for _ in range(1000):
                gene_trees.append(
                    TreeNode.read(f"{gene_tree_dir}/gene_tree_{_}.newick")
                )
            species_tree = TreeNode.read([f"(A:1,B:1);"])
            tree_height = 2e8
            time_tree = get_time_tree(species_tree, tree_height)
            all_mus = get_all_pair_mutation_rate(gene_trees, time_tree)
            summed_mus = [a + b for a, b in all_mus[("A", "B")]]
            muc = np.min(summed_mus)
            mus = np.max(summed_mus)

            plot_distance_distribution(ax, time_tree, gene_trees, muc, mus)

        # test inner nodes instant distributions (sanity check) -> Works in linear
        tree = TreeNode.read(["(A:2,(B:1,C:1):1);"])
        print(tree.ascii_art())
        tree_height = 1e8
        muc = 6e-13
        mus = 1e-9
        rw_step_fraction = [1e4]
        time_tree = get_time_tree(tree, tree_height)
        n = 1000
        threads = 2
        mutation_rate_fun = get_random_walk_tree

        for rwf in rw_step_fraction:
            rw_step = tree_height / rwf
            gene_trees = [tree.copy() for _ in range(n)]
            fun_args = [
                gene_trees,
                [rw_step] * n,
                np.random.uniform(muc, mus, n),
                [muc] * n,
                [mus] * n,
                [True] * n,
            ]
            # fun_args = [gene_trees, [rw_step]*n, 10**(np.random.uniform(np.log10(muc), np.log10(mus), n)), [muc] * n, [mus] * n, [True] * n]
            both_rate_trees = map(mutation_rate_fun, *fun_args)
            instant_rate_trees = [irt for _, irt in both_rate_trees]
            tips = sorted([tip.name for tip in tree.tips()])
            tips_pairs = list(itertools.combinations(tips, 2))
            dedup_lca = {}
            instant_rates = {pair: [] for pair in tips_pairs}
            for tip_1, tip_2 in tips_pairs:
                for rate_tree in instant_rate_trees:
                    lca_tree = rate_tree.lca(
                        [rate_tree.find(tip_1), rate_tree.find(tip_2)]
                    )
                    instant_rates[(tip_1, tip_2)].append(lca_tree.length)

            # plotting
            fig, ax = plt.subplots()
            mu = np.linspace(muc, mus, 100)
            pdf = linear_pdf(mu, muc, mus)
            pdf_log = np.where(
                (mu >= muc) & (mu <= mus),
                np.log(mu) / (mu * (np.log(mus) ** 2 - np.log(muc) ** 2)),
                0,
            )
            pdf_uniform = np.where((mu >= muc) & (mu <= mus), 1 / (mus - muc), 0)
            ax.plot(mu, pdf, color="green", label="linear pdf")
            ax.plot(mu, pdf_uniform, color="red", label="uniform pdf")
            # bins = np.logspace(np.log10(muc), np.log10(mus), 20)
            bins = np.linspace(muc, mus, 20)
            # bins_centers = np.sqrt(bins[1:] * bins[:-1])
            for tip_1, tip_2 in tips_pairs:
                hist, _ = np.histogram(
                    instant_rates[(tip_1, tip_2)], bins=bins, density=True
                )
                # ax.scatter(bins_centers, hist, label=f"{tip_1} vs {tip_2}, {rwf}", alpha=0.5)
                ax.scatter(
                    bins[:-1], hist, label=f"{tip_1} vs {tip_2}, {rwf}", alpha=0.5
                )
                ax.set_xlabel("instant mutation rate at LCA")
                ax.set_ylabel("Density")
                # ax.set_xscale("log")
                # ax.set_yscale("log")
                ax.legend()
            fig.savefig("lca_instant_rate_3leafed_A_outgroup_liiiiin.png", dpi=300)
