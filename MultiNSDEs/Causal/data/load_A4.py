import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from Prognosis.models.parser import adapt_parser
from Common_Functions.data.load_data_helpers import *

def load_data_A4(config):

    train_dir = config.train_dir
    long_data, long_names, long_types = [], [], []
    for i in range(config.num_lenc):
        fname = config.longdata_fname[i]
        print('Loading ', fname)
        long_data_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_data_.replace(".", np.nan, inplace=True)
        if config.mode == 'cv':
            long_data_ = get_data_cv(config, long_data_)
        else:
            long_data_ = get_data_fold(config, long_data_)
        
        if i==0:
            ids_nan_pacc = long_data_.loc[
                (long_data_.TIME == 240) & (long_data_.PACC.isna()),
                "NEW_ID"].unique()

        fname = config.longtypes_fname[i]
        long_info_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_names_ = long_info_['Variable'].values
        long_types_ = long_info_.values[:, 1:]

        if config.mse_head:
            long_types_[:, 0] = np.where(np.isin(long_types_[:, 0], ['pos', 'real']), 'mse', long_types_[:, 0])

        cols_to_convert = [col for col in long_names_ if col in long_data_.columns]
        long_data_[cols_to_convert] = long_data_[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        long_data.append(long_data_)
        long_names.append(long_names_)
        long_types.append(long_types_)

    # Total
    print('Total long missing values')
    get_mean_std_missings(long_data, long_names)

    print('Loading doses')
    dose_data = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname), header=0)
    dose_data.replace(".", np.nan, inplace=True)
    if config.mode == 'cv':
        dose_data = get_data_cv(config, dose_data)
    else:
        dose_data = get_data_fold(config, dose_data)

    print('Loading static data')
    static_data = read_csv_values(os.path.join(
        train_dir, config.staticdata_fname),
        header=0)
    static_data.replace(".", np.nan, inplace=True)
    column_names_list = static_data.columns[1:]
    static_data[column_names_list] = static_data[column_names_list].apply(
        pd.to_numeric, errors='coerce')

    static_info = read_csv_values(os.path.join(
        train_dir, config.statictypes_fname),
        header=0)
    static_names = static_info['Variable'].values
    static_types = static_info.values[:, 1:]
    # in this way because of HIVAE normalization layer
    static_data_dim = sum(entry[1] for entry in static_types)

    static_vals_dim_ind = max(static_data.NEW_ID)
    static_vals_dim = len(static_names)
    if config.mode == 'cv':
        static_data = get_data_cv(config, static_data)
    else:
        static_data = get_data_fold(config, static_data)

    missing_per_column = static_data[static_names].isna().mean() * 100
    print(f"Mean missing Stat %: {missing_per_column.mean():.2f}%")
    print(f"STD missing Stat %: {missing_per_column.std():.2f}%")

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim
    config = adapt_parser(config)
    config.n_drug_var = 1

    # Loading OR data
    or_path = os.path.join(config.or_model_path, 'samples')
    obs_scores = read_csv_values(os.path.join(
        or_path, config.do_scores_fname), header=0)
    pi_scores = read_csv_values(os.path.join(
        or_path, config.ps_scores_fname), header=0)
    
    obs_scores = obs_scores[obs_scores['NEW_ID'].isin(static_data['NEW_ID'])]
    pi_scores = pi_scores[pi_scores['NEW_ID'].isin(static_data['NEW_ID'])]
    scores = obs_scores.merge(pi_scores, on='NEW_ID', how='inner')
    scores = scores.merge(static_data[["NEW_ID", "TRAIN"]], on="NEW_ID", how="left")

    return (config, long_data, long_names,
            long_types, dose_data, static_data,
            static_names, static_types, scores)