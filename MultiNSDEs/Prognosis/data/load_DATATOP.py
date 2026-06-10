import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from models.parser import adapt_parser
from Common_Functions.data.load_data_helpers import *

def load_data_DATATOP(config):

    train_dir = config.train_dir
    long_data, long_names, long_types = [], [], []
    for i in range(config.num_lenc):
        fname = config.longdata_fname
        print('Loading ', fname)
        long_data_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_data_.replace(".", np.nan, inplace=True)
        long_data_ = get_data_fold(config, long_data_)

        fname = config.longtypes_fname
        long_info_ = read_csv_values(os.path.join(
            train_dir, fname), header=0)
        long_names_ = long_info_['Variable'].values
        long_types_ = long_info_.values[:, 1:]

        if config.mse_head: #Likely not using here
            long_types_[:, 0] = np.where(np.isin(long_types_[:, 0], ['pos', 'real']), 'mse', long_types_[:, 0])

        cols_to_convert = [col for col in long_names_ if col in long_data_.columns]
        long_data_[cols_to_convert] = long_data_[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        long_data.append(long_data_)
        long_names.append(long_names_)
        long_types.append(long_types_)

    
    if config.time_to_event: #
        time_event_df = read_csv_values(os.path.join(
            train_dir, config.timetoevent_fname),
            header=0)
        # time_event_df = time_event_df[time_event_df.subject_id.isin(first_pat)]
        time_event_df = get_data_fold(config, time_event_df)
        time_event_names = time_event_df.Var.unique()
        config.num_var_te = len(time_event_names)
    else:
        time_event_df = time_event_names = None

    # DE_RHS data
    de_rhs_df = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname),
        header=0)
    columns_de_rhs = de_rhs_df.columns 
    de_rhs_df[columns_de_rhs] = de_rhs_df[columns_de_rhs].apply(
        pd.to_numeric, errors='coerce')
    de_rhs_df = get_data_fold(config, de_rhs_df)

    # static_data
    static_data = read_csv_values(os.path.join(
        train_dir, config.staticdata_fname),
        header=0)
    static_data.replace(".", np.nan, inplace=True)
    column_names_list = static_data.columns
    static_data[column_names_list] = static_data[column_names_list].apply(
        pd.to_numeric, errors='coerce')

    static_info = read_csv_values(os.path.join(
        train_dir, config.statictypes_fname),
        header=0)
    static_names = static_info['Variable'].values

    static_types = static_info.values[:, 1:]

    # in this way because of HIVAE normalization layer
    static_data_dim = sum(entry[1] for entry in static_types)

    static_vals_dim_ind = len(static_data.PDDOCID.unique())
    static_vals_dim = len(static_names)
    static_data = get_data_fold(config, static_data)

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim
    config.num_lenc = len(long_names)
    config = adapt_parser(config)
    config.n_drug_var = 1

    return (config, long_data, long_names,
            long_types, de_rhs_df, time_event_df,
            time_event_names, static_data,
            static_names, static_types)