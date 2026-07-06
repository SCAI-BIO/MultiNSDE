#!/usr/bin/ipython
import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../')
from models.parser import base_parser
from data.load_data_helpers import *
from syndat.rct.metrics_rct import *
from syndat.rct.preprocessing_tidy_format import *
from syndat.rct.visualization_clinical_trials import *

def round_to_half(x):
    if pd.isna(x):
        return np.nan
    return round(x * 2) / 2

def create_mapping_n_categories(n_categories, max_val, init=0.0):
    step = (max_val - init) / (n_categories - 1)
    mapping = {i: init + i * step for i in range(n_categories)}
    return mapping

def load_long_info(config, read_csv_values):

    fnames = [fname.strip() for fname in config.longtypes_fname.split(",")]

    dfs = []
    for i, fname in enumerate(fnames, start=1):
        path = os.path.join(config.train_dir, fname)
        df = read_csv_values(path, header=0)
        if len(fnames) > 1:
            df["Enc"] = i
        dfs.append(df)

    if len(dfs) > 1:
        long_info = pd.concat(dfs, ignore_index=True)
    else:
        long_info = dfs[0]
    return long_info

def enc_number(fname):
    stem = Path(fname).stem
    match = re.search(r'Enc(\d+)', stem)
    if match:
        return int(match.group(1))
    else:
        return -1

def compute_stats(df, group_cols, metrics, agg, label):
    if group_cols is None:
        out = getattr(df[metrics], agg)().to_frame().T.reset_index(drop=True)
    else:
        out = df.groupby(group_cols, as_index=False)[metrics].agg(agg)
    
    return out.assign(Variable=label)

def filter_extrapolation_endpoint_per_patient(df, patient_col='PDDOCID', time_col='TIME'):
    """
    Gets last obsevation per patient for patient-wise extrapolation scenarios.
    """

    mask_cols = [c for c in df.columns if c.startswith('MASK_')]
    if not mask_cols:
        return df[df[time_col] == df[time_col].max()]

    observed_any = df[mask_cols].fillna(0).gt(0).any(axis=1)
    endpoint_by_patient = (
        df.loc[observed_any, [patient_col, time_col]]
        .groupby(patient_col, as_index=False)[time_col]
        .max()
        .rename(columns={time_col: '_ENDPOINT_TIME'})
    )

    if endpoint_by_patient.empty:
        return df[df[time_col] == df[time_col].max()]

    df_endpoint = df.merge(endpoint_by_patient, on=patient_col, how='inner')
    df_endpoint = df_endpoint[np.isclose(df_endpoint[time_col], df_endpoint['_ENDPOINT_TIME'])]
    return df_endpoint.drop(columns=['_ENDPOINT_TIME'])

def main(config, opcs):

    print('Getting %s Metrics'%(config.dataset))
    sims_files = [p for p in opcs if f'EP{max_ep}' in Path(p).stem and Path(p).stem.startswith('Sims')]
    sims_long = [p for p in sims_files if 'Long' in Path(p).stem and 'Prob' not in Path(p).stem]
    sims_long = sorted(sims_long, key=enc_number)
    long_info = load_long_info(config, read_csv_values)
    long_info['Enc'] = 1

    results_path_LEnc = os.path.join(config.folder_path, "Long_Encs", 'Scenario%d'%(config.Val_Scenario))
    os.makedirs(results_path_LEnc, exist_ok=True)
    long_cols = ['PTNO', 'REPI', 'TIME', 'DRUG']
    cols_extend = ['REC_', 'OBS_', 'SIM_', 'MASK_']
    post_processing_prefix = ['REC_', 'OBS_', 'SIM_']
    if config.dataset == "A4":
        round_05 = ['CADL', 'IF', 'C_Path_Score']
        post_processing_cols = ['MCQT', 'DIGITTOTAL', 'FCTOTF', 'FCTOTC', 'FCTOTAL96']
        only_pos = False # There are some variables with negative values
        max_time = 240
    elif config.dataset == "PROACT":
        post_processing_cols = ['VITALSIGNS@Pulse', 'VITALSIGNS@Blood_Pressure_Diastolic',
                            'VITALSIGNS@Blood_Pressure_Systolic']
        round_05 = []
        only_pos = True # There are NO variables with negative values
        max_time = 17
    elif config.dataset == "DATATOP":
        post_processing_cols=[]
        round_05=[]
        only_pos = True # There are NO variables with negative values
        max_time = 10
    else:
        post_processing_cols=[]
        round_05=[]
        only_pos = True # There are NO variables with negative values

    strat_vars=[]
    group_global = strat_vars + ["TIME"]
    metrics_full_long_cont = []
    metrics_overall_long_cont = []
    metrics_full_long_cat = []
    metrics_overall_long_cat = []
    metrics_path = os.path.join(config.folder_path, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)

    if config.dataset=="DATATOP":
        long_info['Enc']=1

    drug_labels = (
        {0: "Placebo", 1: "Treatment_1", 2: "Treatment_2", 3: "Treatment_3"}
        if config.dataset == "DATATOP"
        else {0: "Placebo", 1: "Treated"}
    )

    # Note that we can't map some of the variables from classes to their original values
    # to calculate the metrics otherwise scores like F1 won't work because of values like 0.5
    for idx, slong_name in enumerate(sims_long):
        ldt_Enc = pd.read_csv(slong_name, na_values='.')
        if config.extrapolation:
            if config.dataset == "DATATOP":
                ldt_Enc = filter_extrapolation_endpoint_per_patient(ldt_Enc)
            else:
                ldt_Enc = ldt_Enc[ldt_Enc.TIME == max_time]
        ldt_Enc["DRUG"] = ldt_Enc["DRUG"].map(drug_labels)
        lt_Enc = long_info[long_info['Enc'] == idx + 1]

        for var in lt_Enc['Variable']:
            print('Long Enc %d - Variable %s'%(idx, var))
            if var in post_processing_cols:
                for prefix in post_processing_prefix:
                    col = f"{prefix}{var}"
                    if col in ldt_Enc.columns:
                        ldt_Enc[col] = ldt_Enc[col].round().astype(int)
            if var in round_05:
                for prefix in post_processing_prefix:
                    col = f"{prefix}{var}"
                    if col in ldt_Enc.columns:
                        ldt_Enc[col] = ldt_Enc[col].apply(round_to_half)

            long_cols_oneVar = long_cols + [s + var for s in cols_extend]

            ldt_Enc_oneVar = ldt_Enc[long_cols_oneVar]
            lt_Enc_oneVar = lt_Enc[lt_Enc['Variable'] == var]
            rp_Enc_oneVar = get_rp(ldt_Enc_oneVar, lt=lt_Enc_oneVar)

            all_parts = []
            unique_ids = ldt_Enc_oneVar['PTNO'].unique()
            for i in range(0, len(unique_ids), 5):
                ids_subset = unique_ids[i:i+5]
                ldt_subset = ldt_Enc_oneVar[ldt_Enc_oneVar["PTNO"].isin(ids_subset)]
                part_df = convert_data_to_tidy(ldt_subset, 'long', only_pos=only_pos,
                                            only_realtimes=(config.Val_Scenario in [0, 1]))
                all_parts.append(part_df)
            ldt_Enc_oneVar = pd.concat(all_parts, axis=0, ignore_index=True)

            if not ldt_Enc_oneVar.empty:
                if var == "PACC":
                    mode="Simulations"
                    dt = ldt_Enc_oneVar.pivot_table(
                        index=strat_vars + ["SUBJID", "Variable"],
                        columns="TYPE",
                        values="DV").dropna(subset=["Observed", "Simulations"])
                    dt["abs_error"] = np.abs(dt["Observed"] - dt[mode])
                    print('Mean PACC', np.round(dt["abs_error"].mean(), 2))
                    print('STD', np.round(dt["abs_error"].std(), 2))

                if lt_Enc_oneVar['Cats'].iloc[0] == 1:
                    ldt_Enc_oneVar["DV"] = (
                        ldt_Enc_oneVar
                        .groupby(["SUBJID", "TIME", "TYPE"])["DV"]
                        .transform("median"))
                else:
                    ldt_Enc_oneVar["DV"] = (
                        ldt_Enc_oneVar
                        .groupby(["SUBJID", "TIME", "TYPE"])["DV"]
                        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA))
                ldt_Enc_oneVar = ldt_Enc_oneVar[ldt_Enc_oneVar.REPI == 1]

                # Given that we are doing it by variable, full and per_time_mean is the same
                # and overall and per_variable are also the same, therefore we are not
                # calculating them
                if lt_Enc_oneVar['Type'].item() == 'cat':
                    metrics = compute_categorical_error_metrics(
                        rp_Enc_oneVar,ldt_Enc_oneVar,
                        mode="Simulations",
                        strat_vars=strat_vars,
                        average="weighted")
                    metrics['overall']['Variable'] = lt_Enc_oneVar['Variable'].item()
                    metrics_full_long_cat.append(metrics['full'])
                    metrics_overall_long_cat.append(metrics['overall'])
                else:
                    metrics = compute_continuous_error_metrics(
                        rp_Enc_oneVar,ldt_Enc_oneVar,
                        mode="Simulations",
                        strat_vars=strat_vars)
                    metrics['overall']['Variable'] = lt_Enc_oneVar['Variable'].item()
                    metrics_full_long_cont.append(metrics['full'])
                    metrics_overall_long_cont.append(metrics['overall'])
    metrics_full_long_cont = pd.concat(metrics_full_long_cont, ignore_index=True)
    metrics_overall_long_cont = pd.concat(metrics_overall_long_cont, ignore_index=True)
    group_strat = strat_vars if strat_vars != [] else None

    try:
        metrics_full_long_cat = pd.concat(metrics_full_long_cat, ignore_index=True)
        metrics_overall_long_cat = pd.concat(metrics_overall_long_cat, ignore_index=True)
        metrics_full_long_cat = pd.concat(
            [metrics_full_long_cat,
            compute_stats(metrics_full_long_cat, group_global, ["F1", "Accuracy", "Precision", "Recall"], "mean", "Average"),
            compute_stats(metrics_full_long_cat, group_global, ["F1", "Accuracy", "Precision", "Recall"], "std", "Std"),
            compute_stats(metrics_full_long_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "mean", "Average (Global)"),
            compute_stats(metrics_full_long_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "std", "Std (Global)")],
            ignore_index=True)
        metrics_full_long_cat[["F1", "Accuracy", "Precision", "Recall"]] = metrics_full_long_cat[["F1", "Accuracy", "Precision", "Recall"]].round(2)
        metrics_full_long_cat.to_csv(os.path.join(metrics_path, 'full_long_cat.csv'), index=False)

        metrics_overall_long_cat = pd.concat(
            [metrics_overall_long_cat,
            compute_stats(metrics_full_long_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "mean", "Average (Global)"),
            compute_stats(metrics_full_long_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "std", "Std (Global)")],
            ignore_index=True)
        metrics_overall_long_cat[["F1", "Accuracy", "Precision", "Recall"]] = metrics_overall_long_cat[["F1", "Accuracy", "Precision", "Recall"]].round(2)
        metrics_overall_long_cat.to_csv(os.path.join(metrics_path, 'overall_long_cat.csv'), index=False)
    except:
        pass

    metrics_full_long_cont = pd.concat(
        [metrics_full_long_cont,            
        compute_stats(metrics_full_long_cont, group_global, ["MAE", "RMSE", "MAPE"], "mean", "Average"),
        compute_stats(metrics_full_long_cont, group_global, ["MAE", "RMSE", "MAPE"], "std", "Std"),
        compute_stats(metrics_full_long_cont, group_strat, ["MAE", "RMSE", "MAPE"], "mean", "Average (Global)"),
        compute_stats(metrics_full_long_cont, group_strat, ["MAE", "RMSE", "MAPE"], "std", "Std (Global)")],
        ignore_index=True)

    metrics_full_long_cont[["MAE", "RMSE", "MAPE"]] = metrics_full_long_cont[["MAE", "RMSE", "MAPE"]].round(2)
    metrics_full_long_cont.to_csv(os.path.join(metrics_path, 'full_long_cont.csv'), index=False)

    metrics_overall_long_cont = pd.concat(
        [metrics_overall_long_cont,
        compute_stats(metrics_overall_long_cont, group_strat, ["MAE", "RMSE", "MAPE"], "mean", "Average (Global)"),
        compute_stats(metrics_overall_long_cont, group_strat, ["MAE", "RMSE", "MAPE"], "std", "Std (Global)")],
        ignore_index=True)
    metrics_overall_long_cont[["MAE", "RMSE", "MAPE"]] = metrics_overall_long_cont[["MAE", "RMSE", "MAPE"]].round(2)
    metrics_overall_long_cont.to_csv(os.path.join(metrics_path, 'overall_long_cont.csv'), index=False)

    if config.static_data:
        sims_stat = [p for p in sims_files if 'Stat' in Path(p).stem]
        static_info = read_csv_values(os.path.join(
            config.train_dir, config.statictypes_fname),
            header=0)
        results_path_SEnc = os.path.join(config.folder_path , "Stat_Encs", 'Scenario%d'%(config.Val_Scenario))
        os.makedirs(results_path_SEnc, exist_ok=True)
        static_cols = ['PTNO', 'REPI', 'DRUG']

        metrics_full_stat_cont = []
        metrics_full_stat_cat = []

        for idx, stat_name in enumerate(sims_stat):
            sdt_Enc = pd.read_csv(stat_name, na_values='.')
            st_Enc = static_info
            if config.dataset == 'A4':
                st_Enc = st_Enc[st_Enc != 'WRKRET'] # A4

            drug_info = ldt_Enc.loc[ldt_Enc["TIME"] == 0, ["PTNO", "REPI", "DRUG"]]
            sdt_Enc = sdt_Enc.merge(drug_info, on=["PTNO", "REPI"], how="left")

            for var in st_Enc['Variable']:
                print('Static Enc %d - Variable %s'%(idx, var))
                if var in post_processing_cols:
                    for prefix in post_processing_prefix:
                        col = f"{prefix}{var}"
                        if col in ldt_Enc.columns:
                            ldt_Enc[col] = ldt_Enc[col].round().astype(int)
                if var in round_05:
                    for prefix in post_processing_prefix:
                        col = f"{prefix}{var}"
                        if col in ldt_Enc.columns:
                            ldt_Enc[col] = ldt_Enc[col].apply(round_to_half)

                static_cols_oneVar = static_cols + [s + var for s in cols_extend]

                sdt_Enc_oneVar = sdt_Enc[static_cols_oneVar]
                st_Enc_oneVar = st_Enc[st_Enc['Variable'] == var]
                rp_Enc_oneVar = get_rp(st=st_Enc_oneVar)

                all_parts = []
                unique_ids = sdt_Enc_oneVar['PTNO'].unique()
                for i in range(0, len(unique_ids), 5):
                    ids_subset = unique_ids[i:i+5]
                    sdt_subset = sdt_Enc_oneVar[sdt_Enc_oneVar["PTNO"].isin(ids_subset)]
                    part_df = convert_data_to_tidy(sdt_subset, 'static', only_pos=True,
                                                   only_realtimes=True)
                    all_parts.append(part_df)  
                sdt_Enc_oneVar = pd.concat(all_parts, axis=0, ignore_index=True)

                if st_Enc_oneVar['Cats'].iloc[0] == 1:
                    sdt_Enc_oneVar["DV"] = (
                        sdt_Enc_oneVar
                        .groupby(["SUBJID", "TYPE"])["DV"]
                        .transform("median"))
                else:
                    sdt_Enc_oneVar["DV"] = (
                        sdt_Enc_oneVar
                        .groupby(["SUBJID", "TYPE"])["DV"]
                        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA))
                sdt_Enc_oneVar = sdt_Enc_oneVar[sdt_Enc_oneVar.REPI == 1]

                # Given that we are doing it by variable  and that it is static
                # full and overall are the same
                if st_Enc_oneVar['Type'].item() == 'cat':
                    metrics = compute_categorical_error_metrics(
                        rp_Enc_oneVar,sdt_Enc_oneVar,
                        mode="Simulations",
                        strat_vars=strat_vars,
                        average="weighted", static=True)
                    metrics_full_stat_cat.append(metrics['full'])
                else:
                    metrics = compute_continuous_error_metrics(
                        rp_Enc_oneVar,sdt_Enc_oneVar,
                        mode="Simulations",
                        strat_vars=strat_vars, static=True)
                    metrics_full_stat_cont.append(metrics['full'])

        try:
            metrics_full_stat_cat = pd.concat(metrics_full_stat_cat, ignore_index=True)
            metrics_full_stat_cat = pd.concat(
                [metrics_full_stat_cat,
                compute_stats(metrics_full_stat_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "mean", "Average"),
                compute_stats(metrics_full_stat_cat, group_strat, ["F1", "Accuracy", "Precision", "Recall"], "std", "Std")],
                ignore_index=True)
            metrics_full_stat_cat[["F1", "Accuracy", "Precision", "Recall"]] = metrics_full_stat_cat[["F1", "Accuracy", "Precision", "Recall"]].round(2)
            metrics_full_stat_cat.to_csv(os.path.join(metrics_path, 'full_stat_cat.csv'), index=False)
        except:
            pass

        metrics_full_stat_cont = pd.concat(metrics_full_stat_cont, ignore_index=True)
        metrics_full_stat_cont = pd.concat(
            [metrics_full_stat_cont,
            compute_stats(metrics_full_stat_cont, group_strat, ["MAE", "RMSE", "MAPE"], "mean", "Average"),
            compute_stats(metrics_full_stat_cont, group_strat, ["MAE", "RMSE", "MAPE"], "std", "Std")],
            ignore_index=True)
        metrics_full_stat_cont[["MAE", "RMSE", "MAPE"]] = metrics_full_stat_cont[["MAE", "RMSE", "MAPE"]].round(2)
        metrics_full_stat_cont.to_csv(os.path.join(metrics_path, 'full_stat_cont.csv'), index=False)

if __name__ == '__main__':

    config = base_parser()
    np.random.seed(config.seed)

    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.dataset, 
                                    config.exp_name, 'Fold%d'%(config.train_fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')

    name_ = 'Val_Imgs_%s'%(config.val_data_type)
    config.folder_path = os.path.join(config.save_path_samples, name_)
    config.save_path = os.path.join(config.save_path_samples, name_, 'Raw_Output')
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)
    opcs = glob(config.save_path + '/*.csv')
    if config.Val_Scenario == 0:
        opcs = [p for p in opcs if not any(p_name.startswith('Val') for p_name in Path(p).stem.split('_'))]
    else:
        opcs = [p for p in opcs if f'Val{config.Val_Scenario}_' in Path(p).stem]

    if config.from_best:
        ep_numbers = []
        for path in opcs:
            stem = Path(path).stem 
            if "EP" in stem:
                ep_numbers.append(int(stem.split("EP")[-1])) 
        max_ep = max(ep_numbers)

    main(config, opcs)
