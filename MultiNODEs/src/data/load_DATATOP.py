import os
import numpy as np
import pandas as pd
import sys
from .load_data_helpers import *
sys.path.append('../')


def load_only_DATATOP_types(config):
    train_dir = config.train_dir

    long_info_ = read_csv_values(os.path.join(
        train_dir, config.real_longtypes_fname), header=0)
    long_types = long_info_.values[:, 1:]

    static_info = read_csv_values(os.path.join(
        train_dir, config.real_statictypes_fname),
        header=0)
    static_types = static_info.values[:, 1:]

    return long_types, static_types


def load_data_DATATOP(config):

    train_dir = config.train_dir

    # Longitudinal data
    long_data_df = read_csv_values(os.path.join(
        train_dir, config.longdata_fname), header=0)
    long_data_df.replace(".", np.nan, inplace=True)

    long_data_df[long_data_df.columns] = long_data_df[long_data_df.columns].apply(
        pd.to_numeric, errors='coerce')

    # Normalize TIME to start at 0
    if long_data_df["TIME"].min() != 0:
        long_data_df["TIME"] = long_data_df["TIME"] - long_data_df["TIME"].min()

    long_data_df = get_data_fold(config, long_data_df)

    long_info = read_csv_values(os.path.join(
        train_dir, config.longtypes_fname), header=0)
    long_names = long_info['Variable'].values
    long_types = long_info.values[:, 1:]

    fix_colnames = ['PDDOCID', 'TIME', 'TRAIN']
    cols_ = fix_colnames + long_names.tolist()
    long_data = long_data_df[cols_]

    # RHS data
    de_rhs_df = read_csv_values(os.path.join(
        train_dir, config.dosesdata_fname), header=0)

    if de_rhs_df["TIME"].min() != 0:
        de_rhs_df["TIME"] = de_rhs_df["TIME"] - de_rhs_df["TIME"].min()

    de_rhs_df[de_rhs_df.columns] = de_rhs_df[de_rhs_df.columns].apply(
        pd.to_numeric, errors='coerce')
    de_rhs_df = get_data_fold(config, de_rhs_df)

    if 'TREAT' not in de_rhs_df.columns:
        raise ValueError('DATATOP doses data must include a TREAT column.')
    if de_rhs_df['TREAT'].isna().any():
        raise ValueError('DATATOP doses data contains non-numeric or missing TREAT values.')

    de_rhs_df["PBO"] = (de_rhs_df["TREAT"] == 0).astype(int)

    # Static data
    if config.static_data:
        static_data = read_csv_values(os.path.join(
            train_dir, config.staticdata_fname),
            header=0)
        static_data.replace(".", np.nan, inplace=True)
        static_data[static_data.columns] = static_data[static_data.columns].apply(
            pd.to_numeric, errors='coerce')

        static_info = read_csv_values(os.path.join(
            train_dir, config.statictypes_fname),
            header=0)
        static_names = static_info['Variable'].values
        static_types = static_info.values[:, 1:]

        # HIVAE normalization layer expected dims
        static_data_dim = sum(entry[1] for entry in static_types)
        static_vals_dim_ind = len(static_data.PDDOCID.unique())
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