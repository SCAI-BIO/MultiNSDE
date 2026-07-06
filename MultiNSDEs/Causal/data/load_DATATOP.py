import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from Causal.models.parser import adapt_parser
from Common_Functions.data.load_data_helpers import *


def load_data_DATATOP(config):

    train_dir = config.train_dir
    long_data, long_names, long_types = [], [], []
    for _ in range(config.num_lenc):
        fname = config.longdata_fname
        print('Loading ', fname)
        long_data_ = read_csv_values(os.path.join(train_dir, fname), header=0)
        long_data_.replace('.', np.nan, inplace=True)
        if config.mode == 'cv':
            long_data_ = get_data_cv(config, long_data_)
        else:
            long_data_ = get_data_fold(config, long_data_)

        fname = config.longtypes_fname
        long_info_ = read_csv_values(os.path.join(train_dir, fname), header=0)
        long_names_ = long_info_['Variable'].values
        long_types_ = long_info_.values[:, 1:]

        if config.mse_head:
            long_types_[:, 0] = np.where(np.isin(long_types_[:, 0], ['pos', 'real']), 'mse', long_types_[:, 0])

        cols_to_convert = [col for col in long_names_ if col in long_data_.columns]
        long_data_[cols_to_convert] = long_data_[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        long_data.append(long_data_)
        long_names.append(long_names_)
        long_types.append(long_types_)

    if config.time_to_event:
        time_event_df = read_csv_values(
            os.path.join(train_dir, config.timetoevent_fname), header=0
        )
        if config.mode == 'cv':
            time_event_df = get_data_cv(config, time_event_df)
        else:
            time_event_df = get_data_fold(config, time_event_df)
        time_event_names = time_event_df.Var.unique()
        config.num_var_te = len(time_event_names)
    else:
        time_event_df = None
        time_event_names = None

    de_rhs_df = read_csv_values(os.path.join(train_dir, config.dosesdata_fname), header=0)
    columns_de_rhs = de_rhs_df.columns
    de_rhs_df[columns_de_rhs] = de_rhs_df[columns_de_rhs].apply(pd.to_numeric, errors='coerce')
    if config.mode == 'cv':
        de_rhs_df = get_data_cv(config, de_rhs_df)
    else:
        de_rhs_df = get_data_fold(config, de_rhs_df)

    if 'TREAT' not in de_rhs_df.columns:
        raise ValueError('DATATOP doses data must include a TREAT column with values in {0,1,2,3}.')
    de_rhs_df['TREAT'] = pd.to_numeric(de_rhs_df['TREAT'], errors='coerce')
    if de_rhs_df['TREAT'].isna().any():
        raise ValueError('DATATOP doses data contains non-numeric or missing TREAT values.')

    static_data = read_csv_values(os.path.join(train_dir, config.staticdata_fname), header=0)
    static_data.replace('.', np.nan, inplace=True)
    static_data[static_data.columns] = static_data[static_data.columns].apply(pd.to_numeric, errors='coerce')

    static_info = read_csv_values(os.path.join(train_dir, config.statictypes_fname), header=0)
    static_names = static_info['Variable'].values
    static_types = static_info.values[:, 1:]

    static_data_dim = sum(entry[1] for entry in static_types)
    static_vals_dim_ind = len(static_data.PDDOCID.unique())
    static_vals_dim = len(static_names)

    if config.mode == 'cv':
        static_data = get_data_cv(config, static_data)
    else:
        static_data = get_data_fold(config, static_data)

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_data_dim = static_data_dim
    config.num_lenc = len(long_names)
    config = adapt_parser(config)
    config.n_drug_var = 4

    if config.type_hivae in ['IC_BL_Causal_HIVAE', 'SLR_BL_Causal_HIVAE']:
        bl_types = np.concatenate(long_types)
        bl_data_dim = sum(entry[1] for entry in bl_types)
        if config.s_data_dim is None:
            config.s_data_dim = bl_data_dim
        else:
            config.s_data_dim += bl_data_dim

    or_path = os.path.join(config.or_model_path, 'samples')
    obs_scores = read_csv_values(os.path.join(or_path, config.do_scores_fname), header=0)
    pi_scores = read_csv_values(os.path.join(or_path, config.ps_scores_fname), header=0)

    id_df = static_data[['PDDOCID', 'TRAIN']].drop_duplicates(subset=['PDDOCID'])
    obs_scores = obs_scores[obs_scores['PDDOCID'].isin(id_df['PDDOCID'])]
    pi_scores = pi_scores[pi_scores['PDDOCID'].isin(id_df['PDDOCID'])]
    scores = obs_scores.merge(pi_scores, on='PDDOCID', how='inner')
    scores = scores.merge(id_df, on='PDDOCID', how='left')

    return (
        config,
        long_data,
        long_names,
        long_types,
        de_rhs_df,
        time_event_df,
        time_event_names,
        static_data,
        static_names,
        static_types,
        scores,
    )
