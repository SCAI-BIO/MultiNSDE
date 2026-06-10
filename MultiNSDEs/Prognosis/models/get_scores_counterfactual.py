#!/usr/bin/ipython
import os
import numpy as np
import torch
import pandas as pd
from glob import glob
from parser import base_parser
import syndat
import warnings
warnings.filterwarnings('ignore')

def main(config):

    print('Getting Scores')
    Val_Scenario = 'Val%s_'%(config.Val_Scenario)
    epoch = config.epoch_init
    drop_na_threshold= 0.8 if config.dataset == "A4" else 0.5

    # ===================================
    # ======== Load the data ============
    # ===================================
    data = config.save_path
    data = torch.load(data, weights_only=False)

    dfs_path = config.folder_path
    raw_path = os.path.join(dfs_path, 'Raw_Output')
    os.makedirs(raw_path, exist_ok=True)
    shap_path = os.path.join(dfs_path, 'SHAP_Discrimination')
    os.makedirs(shap_path, exist_ok=True)

    distribution_similarity = {}
    correlation_scores = {}

    num_enc = len(data['Obs_Long'])
    long_types = data['Real_VarTypes_Long']
    long_names = data['VarNames_Long']
    cont_types_correlation = ["mse", "real", "pos", "count"]
    ord_types_correlation = ["ord"]
    full_types_correlation = cont_types_correlation + ord_types_correlation
    feature_to_enc = {}
    if config.extrapolation:
        T_DE = data['T_DE']
    for k in range(num_enc):

        name_enc = 'Enc%d'%(k)
        path = os.path.join(raw_path, 'Sims_Long_%s_%sEP%d.csv'%(
            name_enc,Val_Scenario,epoch))
        name_enc = 'Enc%d'%(k+1)
        df = pd.read_csv(
            path,
            usecols=lambda c: c.startswith(('OBS_', 'SIM_', 'MASK_', 'REPI', 'DRUG')))
        df = df[df.REPI<=5]
        if config.extrapolation:
            df_sims_long = df_sims_long[df_sims_long['TIME'] == T_DE[-1].item()]

        for col in df.columns:
            if col.startswith("MASK"):
                var_name = col.split("_", 1)[1]
                mask = df[col] == 0
                # Setting values to NAN where  mask is 0
                df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
                df[f'SIM_{var_name}'] = df[f'SIM_{var_name}'].mask(mask)
        cols_obs_rec = [c for c in df.columns if c.startswith('OBS_') or c.startswith('SIM_')]
        df = df.dropna(subset=cols_obs_rec, how='all').reset_index(drop=True)

        if config.Val_Scenario == 2:
            # Placebo
            observed_df = df[df.DRUG == 0]
            predictions_df = df[df.DRUG == 1]
        else:
            # Treated
            observed_df = df[df.DRUG == 1]
            predictions_df = df[df.DRUG == 0]

        observed_df = observed_df[[col for col in df.columns if col.startswith("OBS")]]

        # Step 3: Create predictions_df with all SIM_x columns
        # predictions_df = predictions_df[[col for col in df.columns if col.startswith("REC")]]
        predictions_df = df[[col for col in df.columns if col.startswith("SIM")]]

        # Step 4: Rename columns of predictions_df to match observed_df
        predictions_df.columns = observed_df.columns
        try:
            distribution_similarity.update(syndat.metrics.jensen_shannon_distance(observed_df, predictions_df))
        except:
            pass

        long_type_k = long_types[k]
        long_type_k = long_type_k[:, :2] # Because of PROACT
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

    print('Getting synthetic data scores')
    distribution_similarity = {key.replace("OBS_", ""): value for key, value in distribution_similarity.items()}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(distribution_similarity.items()), columns=['Feature', 'JS Div'])

    Enc = pd.DataFrame(list(feature_to_enc.items()), columns=['Feature', 'Enc'])
    df = df.merge(Enc, on='Feature', how='left')

    stats_by_enc = df.groupby('Enc')['JS Div'].agg(['mean', 'std']).reset_index()
    stats_by_enc.rename(columns={'mean': 'Average Enc', 'std': 'Std Enc'}, inplace=True)

    df = df.merge(stats_by_enc, on='Enc', how='left')
    average_all_enc = df.loc[df['Enc'].notna(), 'JS Div'].mean()
    std_all_enc = df.loc[df['Enc'].notna(), 'JS Div'].std()

    df['Average All Enc'] = average_all_enc
    df['Std All Enc'] = std_all_enc

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

