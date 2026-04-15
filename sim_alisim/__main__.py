#!/usr/bin/env python3
import argparse
import cProfile
import os
import pstats
import shutil
import tracemalloc
import yaml

from sim_alisim.simulate_infer import simulate_infer, simulate_trees


def main():
    parser = argparse.ArgumentParser(description="Run the gene tree simulation and the inference")
    parser.add_argument("simulation_cfg", help="Path to the simulation config file")
    parser.add_argument("inference_cfg", help="Path to the inference config file")
    args = parser.parse_args()

    with open(args.simulation_cfg, "r") as f:
        simulation_cfg = yaml.safe_load(f)
    with open(args.inference_cfg, "r") as f:
        inference_cfg = yaml.safe_load(f)

    os.makedirs(simulation_cfg["outdir"], exist_ok=True)
    # copy the configs to the output directories
    if not os.path.exists(os.path.join(simulation_cfg["outdir"], args.simulation_cfg)):
        shutil.copy(args.simulation_cfg, simulation_cfg["outdir"])
        shutil.copy(args.inference_cfg, simulation_cfg["outdir"])

    if not simulation_cfg["sequences"]:
        simulate_trees(simulation_cfg)
        return
    tracemalloc.start()
    with cProfile.Profile() as pr:
        simulate_infer(simulation_cfg, inference_cfg)
        pstats.Stats(pr).sort_stats("cumtime").print_stats(50)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
