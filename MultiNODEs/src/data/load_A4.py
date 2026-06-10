import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from data.load_data_helpers import *

def load_only_A4_types(config):
    train_dir = config.train_dir
    long_types = []
    for i in range(config.num_lenc):
        fname = config.real_longtypes_fname[i]
        long_info_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_types_ = long_info_.values[:, 1:]

        long_types.append(long_types_)
    long_types = np.vstack(long_types)
    
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

        cols_to_convert = [col for col in long_names_ if col in long_data_.columns]
        long_data_[cols_to_convert] = long_data_[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        if i == 0:
            long_data = long_data_.drop(columns=["SUBSTUDY", "VISCODE", "VISATTR"])
            long_names = long_names_
            long_types = long_types_
        else:
            base_names = ["BID", "NEW_ID", "TIME"]
            cols = np.concatenate([base_names, long_names_])
            long_data = long_data.merge(long_data_[cols], on=base_names, how="outer")
            long_names = np.concatenate([long_names, long_names_])
            long_types = np.concatenate([long_types, long_types_])

    long_data = long_data.sort_values(by=["BID", "TIME"])


    print('Loading doses')
    dose_data = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname), header=0)
    dose_data.replace(".", np.nan, inplace=True)
    dose_data = get_data_fold(config, dose_data)

    if config.static_data:
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
        static_data = get_data_fold(config, static_data)
    else:
        static_vals_dim_ind = None
        static_vals_dim, static_data_dim = None, None
        static_data, static_types, static_names = None, None, None
    missing_per_column = static_data[static_names].isna().mean() * 100
    print(f"Mean missing Stat %: {missing_per_column.mean():.2f}%")
    print(f"STD missing Stat %: {missing_per_column.std():.2f}%")

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim

    return (config, long_data, long_names,
            long_types, dose_data, static_data,
            static_names, static_types) 
