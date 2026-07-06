#!/usr/bin/ipython
import os
import re
import numpy as np
import pandas as pd
from parser import base_parser
from ATE_helpers import *
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

def get_HR(df, baseline_df, dataset, horizon=None):

    risk_df = df.copy()
    baseline_df = baseline_df.copy()
    risk_df['Arm'] = pd.to_numeric(risk_df['Arm'], errors='coerce')
    risk_df['REPI'] = pd.to_numeric(risk_df['REPI'], errors='coerce')
    risk_df['Risk'] = pd.to_numeric(risk_df['Risk'], errors='coerce')
    baseline_df['Time'] = pd.to_numeric(baseline_df['Time'], errors='coerce')
    baseline_df['Cumhaz'] = pd.to_numeric(baseline_df['Cumhaz'], errors='coerce')
    risk_df = risk_df[
        (risk_df['REPI'] == 0)
        & (risk_df['Arm'].isin([0, 1, 2, 3]))
        & (risk_df['Risk'].notna())
    ]
    baseline_df = baseline_df[baseline_df['Time'].notna() & baseline_df['Cumhaz'].notna()]

    patient_rows = []
    for var_name, risk_var in risk_df.groupby('Var'):
        baseline_var = baseline_df[baseline_df['Var'] == var_name]
        
        horizon_var, baseline_cumhaz = cumhaz_at_horizon(baseline_var, horizon=horizon)


        arm_frames = {
            arm: risk_var[risk_var['Arm'] == arm][['PTNO', 'Risk']].rename(columns={'Risk': f'Risk_{arm}'})
            for arm in [0, 1, 2, 3]
        }

        for arm in [1, 2, 3]:

            merged = arm_frames[0].merge(arm_frames[arm], on='PTNO', how='inner')

            out = merged[['PTNO']].copy()
            out['Variable'] = var_name
            out['Comparison'] = f'TREAT_{arm}_vs_0'
            out['HORIZON'] = horizon_var
            out['REF_ARM'] = 0
            out['ALT_ARM'] = arm
            cumhaz_ref = baseline_cumhaz * np.exp(merged['Risk_0'])
            cumhaz_alt = baseline_cumhaz * np.exp(merged[f'Risk_{arm}'])
            valid = (cumhaz_ref > 0) & (cumhaz_alt > 0)

            out = out.loc[valid].copy()
            out['LOG_HR_i'] = np.log(cumhaz_alt.loc[valid]) - np.log(cumhaz_ref.loc[valid])
            out['HR_i'] = np.exp(out['LOG_HR_i'])
            patient_rows.append(out)

        merged_group = arm_frames[0]
        for arm in [1, 2, 3]:
            merged_group = merged_group.merge(arm_frames[arm], on='PTNO', how='inner')

        hazard_01 = 0.5 * (np.exp(merged_group['Risk_0']) + np.exp(merged_group['Risk_1']))
        hazard_23 = 0.5 * (np.exp(merged_group['Risk_2']) + np.exp(merged_group['Risk_3']))
        valid = (hazard_01 > 0) & (hazard_23 > 0)

        grouped = merged_group.loc[valid, ['PTNO']].copy()
        grouped['Variable'] = var_name
        grouped['Comparison'] = 'TREAT_2plus3_vs_0plus1'
        grouped['HORIZON'] = horizon_var
        grouped['REF_ARM'] = '0+1'
        grouped['ALT_ARM'] = '2+3'
        grouped['LOG_HR_i'] = np.log((baseline_cumhaz * hazard_23).loc[valid]) - np.log((baseline_cumhaz * hazard_01).loc[valid])
        grouped['HR_i'] = np.exp(grouped['LOG_HR_i'])
        patient_rows.append(grouped)

    return pd.concat(patient_rows, ignore_index=True)

def get_ATE(df, dataset):

    if dataset == "DATATOP_Causal":
        raise ValueError('DATATOP_Causal ATE must be computed from RMST Risk_TE/Baseline_Risk_TE outputs.')

    else:

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
    result_dir = path
    folds = os.listdir(path)
    folds = [f for f in folds if f.startswith("Fold")]
    if config.dataset == 'DATATOP_Causal':
        tau_outputs = []
        hr_outputs = []
        missing_folds = []
        for fold in folds:
            raw_output_path = os.path.join(path, fold, 'samples/Val_Imgs_Sampling_Prior/Raw_Output')
            raw_outputs = os.listdir(raw_output_path)
            risk_files = [f for f in raw_outputs if f.startswith('Risk_TE')]
            baseline_files = [f for f in raw_outputs if f.startswith('Baseline_Risk_TE')]

            if len(risk_files) == 0 or len(baseline_files) == 0:
                missing_folds.append(fold)
                continue

            best_risk = max(risk_files, key=get_ep)
            best_baseline = max(baseline_files, key=get_ep)

            risk_df = pd.read_csv(os.path.join(raw_output_path, best_risk), index_col=False)
            baseline_df = pd.read_csv(os.path.join(raw_output_path, best_baseline), index_col=False)
            hr_outputs.append(
                get_HR(
                    risk_df,
                    baseline_df,
                    config.dataset,
                    horizon=getattr(config, 'survival_target_time', None)
                )
            )
            tau_outputs.append(
                get_survival_tau_datatop(
                    risk_df,
                    baseline_df,
                    horizon=getattr(config, 'survival_target_time', None)
                )
            )

        if missing_folds:
            raise FileNotFoundError(
                'DATATOP_Causal RMST ATE requires Risk_TE and Baseline_Risk_TE files for every fold. '
                f'Missing files for: {missing_folds}'
            )
        if len(tau_outputs) == 0:
            raise FileNotFoundError('DATATOP_Causal RMST ATE found no Risk_TE/Baseline_Risk_TE outputs.')
        df = summarize_survival_tau(pd.concat(tau_outputs, ignore_index=True))
        ate_path = os.path.join(result_dir, 'ATE_Output.csv')
        df.to_csv(ate_path, index=False)
        if len(hr_outputs) > 0:
            hr_df = summarize_log_hr(pd.concat(hr_outputs, ignore_index=True))
            hr_path = os.path.join(result_dir, 'HR_Output.csv')
            hr_df.to_csv(hr_path, index=False)
        raise SystemExit(0)

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
    df = get_ATE(total_df, config.dataset)
    path = os.path.join(path, 'ATE_Output.csv')
    df.to_csv(path, index=False)