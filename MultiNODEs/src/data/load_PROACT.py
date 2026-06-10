import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from data.load_data_helpers import *

def load_only_PROACT_types(config):
    train_dir = config.train_dir
    long_info_ = read_csv_values(os.path.join(
        train_dir, config.real_longtypes_fname), header=0)
    long_types = long_info_.values[:, 1:]

    static_info = read_csv_values(os.path.join(
        train_dir, config.real_statictypes_fname),
        header=0)
    static_types = static_info.values[:, 1:]

    return long_types, static_types


def load_data_PROACT(config):

    train_dir = config.train_dir
    long_data_df = read_csv_values(os.path.join(
        train_dir, config.longdata_fname),
        header=0)
    # first_pat = long_data_df.subject_id.unique()[:30]
    # long_data_df = long_data_df[long_data_df.subject_id.isin(first_pat)]
    long_data_df.replace(".", np.nan, inplace=True)
    column_names_list = long_data_df.columns[1:]
    long_data_df[column_names_list] = long_data_df[column_names_list].apply(
        pd.to_numeric, errors='coerce')
    if long_data_df["TIME"].min() == 1:
        long_data_df["TIME"] = long_data_df["TIME"] - 1
    long_data_df = get_data_fold(config, long_data_df)

    long_info = read_csv_values(os.path.join(
        train_dir, config.longtypes_fname),
        header=0)
    long_names = long_info['Variable'].values
    long_types = long_info.values[:, 1:]
    fix_colnames = ['subject_id', 'TIME', 'TRAIN']
    cols_ = fix_colnames + long_names.tolist()
    long_data = long_data_df[cols_]

    # DE_RHS data
    de_rhs_df = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname),
        header=0)
    # de_rhs_df = de_rhs_df[de_rhs_df.subject_id.isin(first_pat)]
    if de_rhs_df["TIME"].min() != 0:
        de_rhs_df["TIME"] = de_rhs_df["TIME"] - de_rhs_df["TIME"].min()
    columns_de_rhs = de_rhs_df.columns 
    de_rhs_df[columns_de_rhs] = de_rhs_df[columns_de_rhs].apply(
        pd.to_numeric, errors='coerce')
    de_rhs_df = get_data_fold(config, de_rhs_df)
    de_rhs_df["PBO"] = 1 - de_rhs_df.groupby("subject_id")["Treatment_Group_Delta"].transform("max")

    # static_data
    if config.static_data:
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

        static_vals_dim_ind = len(static_data.subject_id.unique())
        static_vals_dim = len(static_names)
        static_data = get_data_fold(config, static_data)
    else:
        static_vals_dim_ind = None
        static_vals_dim, static_data_dim = None, None
        static_data, static_types, static_names = None, None, None

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim

    return (config, long_data, long_names,
            long_types, de_rhs_df, static_data,
            static_names, static_types)