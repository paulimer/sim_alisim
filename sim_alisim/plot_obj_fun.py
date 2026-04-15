import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sim_alisim.simulate_infer import RATE_EVOLUTION_DIC

def plot_obj_fun(res_dirs):
    res_dirs = pathlib.Path(res_dirs)
    sim_res = [x for x in res_dirs.iterdir() if x.is_dir()]
    all_res_fit_all = {}
    for res in sim_res:
        try:
            all_res_df = pd.read_csv(res / "all_res_fit.csv")
        except FileNotFoundError:
            continue
        else:
            all_res_fit_all[res.name] = all_res_df

    all_res_fit_df = pd.concat(all_res_fit_all).reset_index(names=["exp", "drop"]).drop(["drop"], axis=1)
    all_res_fit_df = all_res_fit_df.melt(id_vars=list(set(all_res_fit_df.columns) - set(RATE_EVOLUTION_DIC.values()))).rename({"variable": "rate_evolution_parameter", "value": "rate_evolution_parameter_value"}, axis=1)

    g = sns.catplot(data=all_res_fit_df, col="exp", y="minimum", x="rate_evolution_parameter_value", kind='violin', col_wrap=5, sharex=False)
    return g
