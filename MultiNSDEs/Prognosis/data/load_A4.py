import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from Prognosis.models.parser import adapt_parser
from Common_Functions.data.load_data_helpers import *

def load_only_A4_types(config):
    train_dir = config.train_dir
    long_types = []
    for i in range(config.num_lenc):
        fname = config.real_longtypes_fname[i]
        long_info_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_types_ = long_info_.values[:, 1:]

        if config.mse_head:
            long_types_[:, 0] = np.where(np.isin(long_types_[:, 0], ['pos', 'real']), 'mse', long_types_[:, 0])

        long_types.append(long_types_)

    static_info = read_csv_values(os.path.join(
        train_dir, config.real_statictypes_fname),
        header=0)
    static_types = static_info.values[:, 1:]
    return long_types, static_types

def load_data_A4(config):

    train_dir = config.train_dir
    long_data, long_names, long_types = [], [], []
    for i in range(config.num_lenc):
        fname = config.longdata_fname[i]
        print('Loading ', fname)
        long_data_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_data_.replace(".", np.nan, inplace=True)
        long_data_ = get_data_fold(config, long_data_)

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
    time_event_df = time_event_names = None

    print('Loading doses')
    dose_data = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname), header=0)
    dose_data.replace(".", np.nan, inplace=True)
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

    static_vals_dim = len(static_names)
    static_data = get_data_fold(config, static_data)

    missing_per_column = static_data[static_names].isna().mean() * 100
    print(f"Mean missing Stat %: {missing_per_column.mean():.2f}%")
    print(f"STD missing Stat %: {missing_per_column.std():.2f}%")

    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim
    config = adapt_parser(config)
    config.n_drug_var = 1

    # Assumes you use at least 1 static variable
    if config.type_hivae in ['IC_HIVAE', 'SLR_HIVAE']:
        bl_types = np.concatenate(long_types)
        bl_data_dim = sum(entry[1] for entry in bl_types)
        config.s_data_dim += bl_data_dim

    return (config, long_data, long_names,
            long_types, dose_data, time_event_df,
            time_event_names, static_data,
            static_names, static_types) 