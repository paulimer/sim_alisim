import pandas as pd
import os
import shutil
import yaml
import subprocess as sp
import shlex
# run from sim_alisim
thread_each = 30
results_dir = "/project/bacteria_mlds/results_simulation/debug_runs/"
base_inf_conf_path = "/project/bacteria_mlds/sim_alisim/entero_inf_config.yaml"
base_shell_script = """
#!/usr/bin/env sh

source /home/etheimer/miniconda3/etc/profile.d/conda.sh
conda activate alisim
python_command
conda deactivate
"""
with open("../base_entero_sim_config.yaml", "r") as base_f:
    base_entero_conf = yaml.safe_load(base_f)
with open("../base_abc_sim_config.yaml", "r") as base_f:
    base_abc_conf = yaml.safe_load(base_f)

params_df = pd.read_csv("parameters.csv", keep_default_na=False)
params_df = params_df.astype(str)
params_df["name"] = params_df["Evolution method"].str.replace("[ \(\)]", "_", regex=True) + "__" +\
    params_df["mumin"] + "_" + params_df["mumax"] + "__" + \
    params_df["Tree"]

for _, row in params_df.iterrows():
    if row["Tree"] == "entero":
        current_conf = base_entero_conf.copy()
    else:
        current_conf = base_abc_conf.copy()
    current_conf["muc"] = row["mumin"]
    current_conf["mus"] = row["mumax"]
    outdir = os.path.join(results_dir, row["name"])
    os.makedirs(outdir, exist_ok=True)
    current_conf["outdir"] = outdir
    evolution_method = "none"
    mean_steps = False
    if row["Evolution method"].startswith("random"):
        evolution_method = "random_walk"
        if "mean" in row["Evolution method"]:
            mean_steps = True
    elif row["Evolution method"].startswith("log"):
        evolution_method = "kishino"
    elif row["Evolution method"] == "null":
        evolution_method = "null"
    current_conf["rate_evolution"] = evolution_method
    current_conf["threads"] = thread_each
    current_conf["mean_steps"] = mean_steps
    if evolution_method != "null":
        params_list = row["Parameter values"].split(",")
        method_parameter = row["Parameter"]
        current_conf[method_parameter] = params_list
    sim_path = os.path.join(outdir, "sim_conf.yaml")
    with open(sim_path, "w") as out_f:
        out_f.write(yaml.safe_dump(current_conf))
    inf_path = shutil.copy(base_inf_conf_path, outdir)
    py_command = f"sim {sim_path} {inf_path}"
    with open(os.path.join(outdir, "submit_simu.sh"), "w") as out_sh:
        shell_script = base_shell_script.replace("python_command", py_command)
        out_sh.write(shell_script)
    sp_command = shlex.split(f"mxqsub --processors={thread_each} --memory=150G -t 10h --group-name 583497 -o {outdir}/submit_simu.log bash {outdir}/submit_simu.sh")
    sp.run(sp_command, check=True)
