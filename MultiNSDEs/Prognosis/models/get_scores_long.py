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

    num_enc = len(data['Obs_Long'])
    long_types = data['Real_VarTypes_Long']
    long_names = data['VarNames_Long']
    cont_types_correlation = ["mse", "real", "pos"]
    ord_types_correlation = ["ord"]
    full_types_correlation = cont_types_correlation + ord_types_correlation
    feature_to_enc = {}
    base_names = ["PTNO", "REPI", "DRUG", "TIME"]
    cols_cont_corr, cols_ord_corr, cols_full_corr = [], [], []
    if config.extrapolation:
        T_DE = data['T_DE']
    for k in range(num_enc):
        name_enc = 'Enc%d'%(k)
        path = os.path.join(raw_path, 'Sims_Long_%s_%sEP%d.csv'%(
            name_enc,Val_Scenario,epoch))

        df = pd.read_csv(
            path,
            usecols=lambda c: c.startswith(('PTNO', 'TIME','OBS_', 'SIM_', 'MASK_', 'REPI', 'DRUG')))
        df = df[df.REPI==1]
        if config.extrapolation:
            df = df[df['TIME'] == T_DE[-1].item()]
            
        for col in df.columns:
            if col.startswith("MASK"):
                var_name = col.split("_", 1)[1]
                mask = df[col] == 0
                # Setting values to NAN where  mask is 0
                df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
                df[f'SIM_{var_name}'] = df[f'SIM_{var_name}'].mask(mask)
        # Removing times where all columns are missing
        cols_obs_rec = [c for c in df.columns if c.startswith('OBS_') or c.startswith('SIM_')]
        cols = np.concatenate([base_names, cols_obs_rec])
        if k == 0:
            full_df = df[cols]
        else:
            full_df = full_df.merge(df[cols], on=base_names, how="outer")
        long_type_k = long_types[k]
        long_type_k = long_type_k[:, :2] # Because of PROACT
        long_name_k = long_names[k]
        for name in long_name_k:
            feature_to_enc[name] = k + 1
        mask = [type_ in cont_types_correlation for type_, _ in long_type_k]
        # long_name_k = long_name_k[mask]
        cols_cont_corr = np.concatenate((cols_cont_corr, [f"OBS_{var}" for var in long_name_k[mask]]))

        mask = [type_ in ord_types_correlation for type_, _ in long_type_k]
        cols_ord_corr = np.concatenate((cols_ord_corr, [f"OBS_{var}" for var in long_name_k[mask]]))

        mask = [type_ in full_types_correlation for type_, _ in long_type_k]
        cols_full_corr = np.concatenate((cols_full_corr, [f"OBS_{var}" for var in long_name_k[mask]]))

    observed_df = full_df[[col for col in full_df.columns if col.startswith("OBS")]]
    predictions_df = full_df[[col for col in full_df.columns if col.startswith("REC")]]
    predictions_df.columns = observed_df.columns

    print('Difference Spearmen correlation for all long continuous variables')
    print(syndat.metrics.normalized_correlation_difference(observed_df[cols_cont_corr], predictions_df[cols_cont_corr]))

    print('Difference Spearmen correlation for all long ordinal variables')
    print(syndat.metrics.normalized_correlation_difference(observed_df[cols_ord_corr], predictions_df[cols_ord_corr]))

    print('Difference Spearmen correlation for all long continuous and ordinal variables')
    print(syndat.metrics.normalized_correlation_difference(observed_df[cols_full_corr], predictions_df[cols_full_corr]))

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