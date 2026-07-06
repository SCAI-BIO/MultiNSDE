#!/usr/bin/ipython
import os
import numpy as np
import torch
import pandas as pd
from glob import glob
from parser import base_parser
import syndat
import sys
sys.path.append('../../')
from Common_Functions.models.val_utils import breslow_baseline_from_risk
import warnings
warnings.filterwarnings('ignore')


def round_ordinal_columns(df, var_names, var_types, prefixes=('OBS_', 'SIM_', 'REC_')):
    """Round columns tied to ordinal variables while preserving NaNs."""
    var_types = np.asarray(var_types)
    if var_types.ndim == 1:
        return df

    ord_vars = [
        name for name, (type_, _) in zip(var_names, var_types[:, :2])
        if str(type_).lower() == 'ord'
    ]
    if not ord_vars:
        return df

    for var in ord_vars:
        for prefix in prefixes:
            col = f'{prefix}{var}'
            if col in df.columns:
                df[col] = np.rint(df[col])
    return df


def filter_extrapolation_endpoint_per_patient(df, patient_col='PTNO', time_col='TIME'):
    """Keep one extrapolation endpoint row per patient based on observed masks. DATATOP only.

    For each patient, we infer the endpoint as the latest time where at least one
    longitudinal variable is observed (MASK_* == 1). If data curation is messed up this won't work.
    """
    if df.empty or patient_col not in df.columns or time_col not in df.columns:
        return df

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

    # Fallback for rare cases where all mask values are zero/missing.
    if endpoint_by_patient.empty:
        return df[df[time_col] == df[time_col].max()]

    df_endpoint = df.merge(endpoint_by_patient, on=patient_col, how='inner')
    df_endpoint = df_endpoint[np.isclose(df_endpoint[time_col], df_endpoint['_ENDPOINT_TIME'])]
    return df_endpoint.drop(columns=['_ENDPOINT_TIME'])

def main(config):

    print('Getting raw data')
    val_runs = config.nruns_ppd
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)
    epoch = config.epoch_init

    # ===================================
    # ======== Load the data ============
    # ===================================
    data = config.save_path
    data = torch.load(data, weights_only=False)

    dfs_path = config.folder_path
    raw_path = os.path.join(dfs_path, 'Raw_Output')
    os.makedirs(raw_path, exist_ok=True)

    PTNOs = data['PTNO']
    PBO = data['PBO']
    if config.Val_Scenario in [0, 1]:
        z_init = data['REC_Z_init']
        Total_Z_init = data['Z_init'][:, :val_runs]

        z_init = z_init.numpy()
        Total_Z_init = Total_Z_init.numpy()

        # Get the sizes
        B, N = z_init.shape  # For z_init
        B_total, R, N_total = Total_Z_init.shape  # For Total_Z_init (B, R, N)

        # Sanity check that B and N are consistent
        assert B == B_total, "Batch sizes in z_init and Total_Z_init do not match"
        assert N == N_total, "Feature sizes in z_init and Total_Z_init do not match"

        Total_Z_init = Total_Z_init.reshape(B * R, N)  # Flatten Total_Z_init to (B*R) x N

        zinit_data = pd.DataFrame(
            z_init, columns=[f'Feature_{i+1}' for i in range(N)])
        zinit_data['PTNO'] = PTNOs.numpy()
        zinit_data['PBO'] = PBO.numpy()
        zinit_data['REPI'] = [0] * B  # REPI = 0 for z_init

        Total_df = pd.DataFrame(Total_Z_init, columns=[f'Feature_{i+1}' for i in range(N)])
        Total_df['PTNO'] = PTNOs.repeat_interleave(R).numpy()
        Total_df['PBO'] = PBO.repeat_interleave(R).numpy()
        Total_df['REPI'] = [r + 1 for _ in range(B) for r in range(R)]  # REPI > 0 for replicates

        # Concatenate both dataframes
        final_df = pd.concat([zinit_data, Total_df], ignore_index=True)
        final_df = final_df.sort_values(by=['PTNO', 'REPI']).reset_index(drop=True)

        # Reorder columns to have 'Batch' and 'REPI' first
        final_df = final_df[['PTNO', 'REPI', 'PBO'] + [f'Feature_{i+1}' for i in range(N)]]
        Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)
        path = os.path.join(raw_path, 'Zinit_%sEP%d.csv'%(Val_Scenario,epoch))
        final_df.to_csv(path, index=False)
    
    if config.time_to_event:
        # Build one row per (patient, endpoint, replicate) for survival risk export
        risk = data['Risk']
        sim_risk = data['SIMS_Risk'][:, :val_runs]
        time = data['Time_TE']
        event = data['Event_TE']
        te_names = data['VarNames_TE']
    
        B, R, Dim_Te = sim_risk.shape
        risk_records = []

        for b in range(B):
            for d in range(Dim_Te):
                risk_records.append({
                    "PTNO": PTNOs[b].item(),
                    "PBO": PBO[b].item(),
                    "REPI": 0,
                    "Var": te_names[d],
                    "Time": float(time[b, d]),
                    "Event": float(event[b, d]),
                    "Risk": float(risk[b, d])})

        for b in range(B):
            for d in range(Dim_Te):
                for n in range(R):
                    risk_records.append({
                        "PTNO": PTNOs[b].item(),
                        "PBO": PBO[b].item(),
                        "REPI": n + 1,
                        "Var": te_names[d],
                        "Time": float(time[b, d]),
                        "Event": float(event[b, d]),
                        "Risk": float(sim_risk[b, n, d])
                    })

        risk_records = pd.DataFrame(risk_records)
        risk_records = risk_records.sort_values(by=["PTNO", "REPI"]).reset_index(drop=True)
        path = os.path.join(raw_path, 'Risk_TE_%sEP%d.csv'%(Val_Scenario,epoch))
        risk_records.to_csv(path, index=False)

        if config.Val_Scenario == 0:
            # Baseline hazard is computed from observed/reconstructed risk only
            baseline_risk = breslow_baseline_from_risk(risk_records[risk_records.REPI==0])
            path = os.path.join(raw_path, 'Baseline_Risk_TE_EP%d.csv'%(epoch))
            baseline_risk.to_csv(path, index=False)

    distribution_similarity = {}
    correlation_scores = {}

    drug_type = data['Drug_Type']
    num_enc = len(data['Obs_Long'])
    long_types = data['Real_VarTypes_Long']
    long_names = data['VarNames_Long']
    cont_types_correlation = ["mse", "real", "pos", "count"]
    ord_types_correlation = ["ord"]
    full_types_correlation = cont_types_correlation + ord_types_correlation
    feature_to_enc = {}
    if config.extrapolation and config.dataset != 'DATATOP':
        # Keep legacy behavior for non-DATATOP datasets. For datatip we don't need it because we will rebuild the patient-specific extrapolation time
        T_DE = data['T_DE']
    for k in range(num_enc):
        # Export long trajectories for all DATATOP scenarios used downstream by get_scores_long
        if config.Val_Scenario in [0, 1, 2, 3, 4, 5]:
            PTNO, REP, TIME = [], [], []
            OBS_LONG, SIMS_LONG = [], []
            REC_LONG, MASK_LONG = [], []
            DRUG = []

            Obs_long = data['Obs_Long'][k]
            Mask_long = data['Mask_Long'][k]
            SIMS_long = data['SIMS_Long'][k]
            REC_long = data['REC_Long'][k]
            VarNames_long = data['VarNames_Long'][k]
            T = data['T'][k]

            for nr in range(val_runs):
                Obs_long_rep = Obs_long[:, nr]
                SIMS_long_rep = SIMS_long[:, nr]
                Mask_long_rep = Mask_long[:, nr]
                if config.val_data_type == 'NoVI':
                    REC_long_rep = REC_long
                else:
                    REC_long_rep = REC_long[:, nr]

                for ptno in range(Obs_long.shape[0]):
        
                    Obs_long_ptno = Obs_long_rep[ptno]
                    SIMS_long_ptno = SIMS_long_rep[ptno]
                    Mask_long_ptno = Mask_long_rep[ptno]
                    REC_long_ptno = REC_long_rep[ptno]

                    l_ = Obs_long_ptno.shape[0]
                    rep_ = torch.tensor([nr + 1]).repeat(l_)
                    ptno_ = torch.tensor(PTNOs[ptno]).repeat(l_)
                    if drug_type is not None:
                        drug_ = drug_type[ptno][1].repeat(l_)

                    PTNO.append(ptno_.unsqueeze(1))
                    REP.append(rep_.unsqueeze(1))
                    TIME.append(T.unsqueeze(1))
                    if drug_type is not None:
                        DRUG.append(drug_.unsqueeze(1))
                    OBS_LONG.append(Obs_long_ptno)
                    SIMS_LONG.append(SIMS_long_ptno)
                    MASK_LONG.append(Mask_long_ptno)
                    REC_LONG.append(REC_long_ptno)
                        
            # We are also doing with vstack for PTNO, REP and TIME
            # to be sure that all the values will match
            PTNO = torch.vstack(PTNO)[:, 0].numpy()
            REP = torch.vstack(REP)[:, 0].numpy()
            TIME = torch.vstack(TIME)[:, 0].numpy()
            if drug_type is not None:
                DRUG = torch.vstack(DRUG)[:, 0].numpy()
                df_sims_l = pd.DataFrame(
                    {'PTNO':PTNO, 'REPI':REP,
                    'TIME':TIME, 'DRUG':DRUG})
            else:
                df_sims_l = pd.DataFrame(
                    {'PTNO':PTNO, 'REPI':REP,
                    'TIME':TIME})
            
            OBS_LONG = torch.vstack(OBS_LONG).numpy()
            SIMS_LONG = torch.vstack(SIMS_LONG).numpy()
            MASK_LONG = torch.vstack(MASK_LONG).numpy()
            REC_LONG = torch.vstack(REC_LONG).numpy()
        
            OBS_VarNames_LONG = ['OBS_' + var for var in VarNames_long]
            SIM_VarNames_LONG = ['SIM_' + var for var in VarNames_long]
            MASK_VarNames_LONG = ['MASK_' + var for var in VarNames_long]
            REC_VarNames_LONG = ['REC_' + var for var in VarNames_long]
        
            df_OBS_long = pd.DataFrame(OBS_LONG, columns=OBS_VarNames_LONG)
            df_SIMS_long = pd.DataFrame(SIMS_LONG, columns=SIM_VarNames_LONG)
            df_MASK_long = pd.DataFrame(MASK_LONG, columns=MASK_VarNames_LONG)
            df_REC_long = pd.DataFrame(REC_LONG, columns=REC_VarNames_LONG)
        
            df_sims_long = pd.concat([df_sims_l, df_OBS_long,
                                    df_SIMS_long, df_MASK_long,
                                    df_REC_long], axis=1)

            long_type_k = long_types[k]
            long_type_k = long_type_k[:, :2] # Because of PROACT
            df_sims_long = round_ordinal_columns(df_sims_long, VarNames_long, long_type_k)

            name_enc = 'Enc%d'%(k)
            path = os.path.join(raw_path, 'Sims_Long_%s_%sEP%d.csv'%(
                name_enc,Val_Scenario,epoch))

            df_sims_long.to_csv(path, index=False)

            if config.extrapolation:
                if config.dataset == 'DATATOP':
                    # DATATOP can have patient-specific extrapolation endpoints.
                    df_sims_long = filter_extrapolation_endpoint_per_patient(df_sims_long)
                else:
                    # Legacy behavior: one shared endpoint time for all patients.
                    df_sims_long = df_sims_long[df_sims_long['TIME'] == T_DE[-1].item()]

            # Score one representative replicate after applying observation masks
            df = df_sims_long[df_sims_long.REPI==1]
            for col in df.columns:
                if col.startswith("MASK"):
                    var_name = col.split("_", 1)[1]
                    mask = df[col] == 0
                    # Setting values to NAN where  mask is 0
                    df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
                    df[f'SIM_{var_name}'] = df[f'SIM_{var_name}'].mask(mask)
            # Removing times where all columns are missing due to the SDE
            cols_obs_rec = [c for c in df.columns if c.startswith('OBS_') or c.startswith('REC_')]
            df = df.dropna(subset=cols_obs_rec, how='all').reset_index(drop=True)
            observed_df = df[[col for col in df.columns if col.startswith("OBS")]]

            # Step 3: Create predictions_df with all SIM_x columns
            predictions_df = df[[col for col in df.columns if col.startswith("SIM")]]
            
            # Step 4: Rename columns of predictions_df to match observed_df
            predictions_df.columns = observed_df.columns
            try:
                distribution_similarity.update(syndat.metrics.jensen_shannon_distance(observed_df, predictions_df))
            except:
                pass

            long_name_k = long_names[k]
            for name in long_name_k:
                feature_to_enc[name] = k + 1
            mask = [type_ in cont_types_correlation for type_, _ in long_type_k]
            cols = [f"OBS_{var}" for var in long_name_k[mask]]
            correlation_scores.update({'%s_DiffCont_Pearson'%(name_enc): syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='pearson')})
            correlation_scores.update({'%s_DiffCont_Spearman'%(name_enc): syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

            mask = [type_ in ord_types_correlation for type_, _ in long_type_k]
            cols = [f"OBS_{var}" for var in long_name_k[mask]]
            correlation_scores.update({'%s_DiffOrd_Spearman'%(name_enc): syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

            mask = [type_ in full_types_correlation for type_, _ in long_type_k]
            cols = [f"OBS_{var}" for var in long_name_k[mask]]
            correlation_scores.update({'%s_DiffContOrd_Spearman'%(name_enc): syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

    if config.Val_Scenario in [0, 1]:
        feature_to_senc = {}
        static_types = data['Real_VarTypes_Stat']
        static_types = static_types[:, :2] # Because of PROACT
        static_names = data['VarNames_Stat']
        static_names = static_names[static_names != 'WRKRET'] # A4

        for name in static_names:
            feature_to_senc[name] = 1

        PTNO_S, REP_S = [], []
        OBS_STAT, SIMS_STAT = [], []
        REC_STAT, MASK_STAT = [], []

        Obs_stat = data['Obs_Stat']
        Mask_stat = data['Mask_Stat']
        SIMS_stat = data['SIMS_Stat']
        REC_stat = data['REC_Stat']
        VarNames_stat = data['VarNames_Stat']

        for nr in range(val_runs):

            Obs_stat_rep = Obs_stat[:, nr]
            SIMS_stat_rep = SIMS_stat[:, nr]
            Mask_stat_rep = Mask_stat[:, nr]

            for ptno in range(Obs_stat.shape[0]):
    
                Obs_stat_ptno = Obs_stat_rep[ptno]
                SIMS_stat_ptno = SIMS_stat_rep[ptno]
                Mask_stat_ptno = Mask_stat_rep[ptno]
                REC_stat_ptno = REC_stat[ptno]
    
                rep_s = torch.tensor([nr + 1])
                ptno_s = torch.tensor(PTNOs[ptno]).repeat(1)

                OBS_STAT.append(Obs_stat_ptno)
                SIMS_STAT.append(SIMS_stat_ptno)
                MASK_STAT.append(Mask_stat_ptno)
                REC_STAT.append(REC_stat_ptno)
                PTNO_S.append(ptno_s.unsqueeze(1))
                REP_S.append(rep_s.unsqueeze(1))

        PTNO_S = torch.vstack(PTNO_S)[:, 0].numpy()
        REP_S = torch.vstack(REP_S)[:, 0].numpy()

        df_sims_s = pd.DataFrame({'PTNO':PTNO_S, 'REPI':REP_S})

        OBS_STAT = torch.vstack(OBS_STAT).numpy()
        SIMS_STAT = torch.vstack(SIMS_STAT).numpy()
        MASK_STAT = torch.vstack(MASK_STAT).numpy()
        REC_STAT = torch.vstack(REC_STAT).numpy()

        OBS_VarNames_STAT = ['OBS_' + var for var in VarNames_stat]
        SIMS_VarNames_STAT = ['SIM_' + var for var in VarNames_stat]
        MASK_VarNames_STAT = ['MASK_' + var for var in VarNames_stat]
        REC_VarNames_STAT = ['REC_' + var for var in VarNames_stat]

        df_OBS_stat = pd.DataFrame(OBS_STAT, columns=OBS_VarNames_STAT)
        df_SIM_stat = pd.DataFrame(SIMS_STAT, columns=SIMS_VarNames_STAT)
        df_MASK_stat = pd.DataFrame(MASK_STAT, columns=MASK_VarNames_STAT)
        df_REC_stat = pd.DataFrame(REC_STAT, columns=REC_VarNames_STAT)

        df_sims_stat = pd.concat([df_sims_s, df_OBS_stat,
                                df_SIM_stat, df_MASK_stat,
                                df_REC_stat], axis=1)
        if config.dataset == "DATATOP":
            df_sims_stat = round_ordinal_columns(df_sims_stat, VarNames_stat, static_types)
            df_sims_stat = df_sims_stat.loc[:, ~df_sims_stat.columns.str.endswith('_WRKRET')]

        path = os.path.join(raw_path, 'Sims_Stat_%sEP%d.csv'%(Val_Scenario,epoch))
        df_sims_stat.to_csv(path, index=False)
        df = df_sims_stat[df_sims_stat.REPI==1]
        for col in df.columns:
            if col.startswith("MASK"):
                var_name = col.split("_", 1)[1]
                mask = df[col] == 0
                # Setting values to NAN where  mask is 0
                df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
                df[f'SIM_{var_name}'] = df[f'SIM_{var_name}'].mask(mask)
        cols_obs_rec = [c for c in df.columns if c.startswith('OBS_') or c.startswith('SIM_')]
        df = df.dropna(subset=cols_obs_rec, how='all').reset_index(drop=True)
        observed_df = df[[col for col in df.columns if col.startswith("OBS")]]

        # Step 3: Create predictions_df with all SIM_x columns
        predictions_df = df[[col for col in df.columns if col.startswith("SIM")]]
        
        # Step 4: Rename columns of predictions_df to match observed_df
        predictions_df.columns = observed_df.columns
        try:
            distribution_similarity.update(syndat.metrics.jensen_shannon_distance(observed_df, predictions_df))
        except:
            pass

        mask = [type_ in cont_types_correlation for type_, _ in static_types]
        cols = [f"OBS_{var}" for var in static_names[mask]]
        correlation_scores.update({'Stat_DiffCont_Pearson': syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='pearson')})
        correlation_scores.update({'Stat_DiffCont_Spearman': syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

        mask = [type_ in ord_types_correlation for type_, _ in static_types]
        cols = [f"OBS_{var}" for var in static_names[mask]]
        correlation_scores.update({'Stat_DiffOrd_Spearman': syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

        mask = [type_ in full_types_correlation for type_, _ in static_types]
        cols = [f"OBS_{var}" for var in static_names[mask]]
        correlation_scores.update({'Stat_DiffContOrd_Spearman': syndat.metrics.normalized_correlation_difference(observed_df[cols], predictions_df[cols], method='spearman')})

        print('Raw outputs were saved')

        loss_path =  os.path.join(config.save_path_losses, 'losses.xlsx')
        df = pd.read_excel(loss_path, index_col=0)
        df = df[df.Epoch == epoch]
        path = os.path.join(raw_path, 'Losses_EP%d.csv'%(epoch))
        df.to_csv(path, index=False)

        print('Getting synthetic data scores')
        distribution_similarity = {key.replace("OBS_", ""): value for key, value in distribution_similarity.items()}

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(list(distribution_similarity.items()), columns=['Feature', 'JS Div'])

        Enc = pd.DataFrame(list(feature_to_enc.items()), columns=['Feature', 'Enc'])
        SEnc = pd.DataFrame(list(feature_to_senc.items()), columns=['Feature', 'SEnc'])

        df = df.merge(Enc, on='Feature', how='left')
        df = df.merge(SEnc, on='Feature', how='left')

        stats_by_enc = df.groupby('Enc')['JS Div'].agg(['mean', 'std']).reset_index()
        stats_by_enc.rename(columns={'mean': 'Average Enc', 'std': 'Std Enc'}, inplace=True)

        stats_by_senc = df.groupby('SEnc')['JS Div'].agg(['mean', 'std']).reset_index()
        stats_by_senc.rename(columns={'mean': 'Average SEnc', 'std': 'Std SEnc'}, inplace=True)

        df = df.merge(stats_by_enc, on='Enc', how='left')
        df = df.merge(stats_by_senc, on='SEnc', how='left')

        average_all_enc = df.loc[df['Enc'].notna(), 'JS Div'].mean()
        std_all_enc = df.loc[df['Enc'].notna(), 'JS Div'].std()

        average_all_senc = df.loc[df['SEnc'].notna(), 'JS Div'].mean()
        std_all_senc = df.loc[df['SEnc'].notna(), 'JS Div'].std()

        df['Average All Enc'] = average_all_enc
        df['Std All Enc'] = std_all_enc
        df['Average All SEnc'] = average_all_senc
        df['Std All SEnc'] = std_all_senc

        path = os.path.join(raw_path, 'Similarity_%sEP%d.csv'%(Val_Scenario,epoch))
        df.to_csv(path, index=False)
        print(df)

        df = pd.DataFrame(list(correlation_scores.items()), columns=['Set', 'Correlation'])
        path = os.path.join(raw_path, 'Correlation_%sEP%d.csv'%(Val_Scenario,epoch))
        df.to_csv(path, index=False)
        print(df)

        print('Similarity scores were saved')

if __name__ == '__main__':

    config = base_parser()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.dataset, 
                                    config.exp_name, 'Fold%d'%(config.train_fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    name_ = 'Val_Imgs_%s'%(config.val_data_type)
    config.folder_path = os.path.join(config.save_path_samples, name_)
    save_path = os.path.join(config.save_path_samples, name_)
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)

    # If there are more than 1 checkpoint, the model will do the plots
    # for all the saved checkpoints
    if config.from_best:
        opcs = glob(save_path + '/*.pth')
        epoch = 1
        for opc in opcs:
            if 'Best' in opc:
                epoch_opc = opc.split('/')[-1].split('Ep')[-1].split('.pth')[0]
    
                if int(epoch_opc) > epoch:
                    epoch = int(epoch_opc)
        config.epoch_init = epoch
        config.save_path = os.path.join(
            save_path, 'Results_Best_%sEp%d.pth'%(Val_Scenario,config.epoch_init))
    else:
        config.save_path = os.path.join(
            save_path, 'Results_%sEp%d.pth'%(Val_Scenario,config.epoch_init))
    main(config)