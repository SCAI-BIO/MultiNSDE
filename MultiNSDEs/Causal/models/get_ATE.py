#!/usr/bin/ipython
import os
import re
import numpy as np
import pandas as pd
from parser import base_parser
import warnings
warnings.filterwarnings('ignore')

def get_ep(fname):
    m = re.search(r"EP(\d+)", fname)
    return int(m.group(1)) if m else -1

def AIPW_Y_hat(df, a, t, g=True):
    if a == 1:
        mu_a = df["TRT_SIMS_BL"] if t==0 else df["TRT_SIMS_END"]
        pi_a = df["PS_SCORES"]
    else:
        mu_a = df["PBO_SIMS_BL"] if t==0 else df["PBO_SIMS_END"]
        pi_a = 1.0 - df["PS_SCORES"]

    obs = df["OBS_BL"] if t==0 else df["OBS_END"]
    mask = df["MASK_BL"] if t==0 else df["MASK_END"]

    g_a = df["OBS_SCORES"] if g else 0.7
    indicator = (df["TRT"] == a).astype(float)
    denom = (pi_a * g_a).clip(lower=1e-6)
    return np.round(mu_a + ((indicator * mask / denom )* (obs - mu_a)), 3)

def get_ATE(df):

    # df = df[df.MASK_END == 1.0]
    df["Y1_BL_hat"] = AIPW_Y_hat(df, a=1, t=0)
    df["Y1_END_hat"] = AIPW_Y_hat(df, a=1, t=1)
    df["Y0_BL_hat"] = AIPW_Y_hat(df, a=0, t=0)
    df["Y0_END_hat"] = AIPW_Y_hat(df, a=0, t=1)

    df["Y1_DIFF"] = df["Y1_END_hat"] - df["Y1_BL_hat"]
    df["Y0_DIFF"] = df["Y0_END_hat"] - df["Y0_BL_hat"]
    df["Y_SLOPE"] = df["Y1_DIFF"] - df["Y0_DIFF"]

    Tau_i = np.round((df.groupby("PTNO")["Y_SLOPE"].mean().rename("Y_SLOPE_i")), 3)
    df = df.merge(Tau_i, on="PTNO", how="left")
    
    PSI_hat = np.round(Tau_i.mean(),3)
    n = Tau_i.shape[0]

    Var_PSI = ((Tau_i - PSI_hat) ** 2).sum() / (n ** 2)
    Std_PSI = np.round(np.sqrt(Var_PSI), 3)

    ci_lower = np.round(PSI_hat - 1.96 * Std_PSI, 3)
    ci_upper = np.round(PSI_hat + 1.96 * Std_PSI, 3)

    df["ATE_SLOPE_AIPW"] = PSI_hat
    df["ATE_SLOPE_STD"] = Std_PSI
    df["ATE_SLOPE_LB"] = ci_lower
    df["ATE_SLOPE_UB"] = ci_upper

    print('ATE SLOPE AIPW:', PSI_hat)
    print('CI: (%.3f , %.3f)'%(ci_lower, ci_upper))

    mae_pat = np.round((df.loc[(df["MASK_END"] == 1) & (df["IPRED_END"].notna())]
      .assign(abs_err=lambda x: np.abs(x["OBS_END"] - x["IPRED_END"]))
      .groupby("PTNO")["abs_err"]
      .mean()), 3)
    df["MAE_i"] = df["PTNO"].map(mae_pat)
    df.loc[df["MASK_END"] == 0, "MAE_i"] = np.nan
    mae_global = np.round(mae_pat.mean(), 3)
    df["MAE_avg"] = mae_global
    print('average MAE: ', mae_global )
    return df

if __name__ == '__main__':

    config = base_parser()

    path = os.path.join(config.save_path, config.dataset, config.exp_name)
    folds = os.listdir(path)
    folds = [f for f in folds if f.startswith("Fold")]
    for i, fold in enumerate(folds):
        raw_output_path = os.path.join(path, fold, 'samples/Val_Imgs_Sampling_Prior/Raw_Output')
        raw_outputs = os.listdir(raw_output_path)
        raw_outputs = [f for f in raw_outputs if f.startswith("Output")]
        best_file = max(raw_outputs, key=get_ep)

        output_path = os.path.join(raw_output_path, best_file)
        df = pd.read_csv(output_path, index_col=False)

        if i==0:
            total_df = df
        else:
            total_df = pd.concat((total_df, df), ignore_index=True)
    df = get_ATE(total_df)
    path = os.path.join(path, 'ATE_Output.csv')
    df.to_csv(path, index=False)