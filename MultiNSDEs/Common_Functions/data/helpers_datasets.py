import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch


# ====================================================
# =========== Loading static informaiton =============
# ====================================================
def process_static(static_data, static_names, static_types):
    continuous_var = [col for col, col_type in zip(static_names,
                      static_types) if col_type[0] in ['real', 'pos', 'gamma', 'mse', 'count']]

    # Ones where we have a value 0 for missing values
    static_miss_mask = (1 - pd.isna(static_data[static_names]).astype(int)).values
    static_miss_mask = torch.from_numpy(static_miss_mask)

    # Filling nans will only play a role if an imputation layer is not being used
    # Filling nan values of the continuous variables with the median
    median_dict = static_data[continuous_var].median()
    static_data.fillna(value=median_dict, inplace=True)

    static_data = torch.from_numpy(static_data[static_names].values)
    # Filling nan values of the categorical variables with 0
    static_data = torch.nan_to_num(static_data, nan=0.0)

    return static_data, static_miss_mask, static_types


def build_long_lists(long_data, long_names, long_types, unique_ptnos):

    long_miss_mask_list, long_data_list, long_Ts = [], [], []
    for i, (long_data_, long_names_, long_types_) in enumerate(zip(long_data, long_names, long_types)):
        # just to be sure
        aux_ = np.append(long_names_, 'TIME')
        unique_times = long_data_[aux_].TIME.unique()
        # To be sure it begins in 0 in case baseline is coded as 1
        unique_times = unique_times - unique_times[0]
        long_shape = (len(unique_ptnos), len(unique_times), len(long_names_))

        long_Ts.append(torch.from_numpy(unique_times).float())
        long_miss_mask = (1 - pd.isna(long_data_[long_names_]).astype(int)).values
        long_miss_mask = torch.from_numpy(long_miss_mask)
        long_miss_mask = torch.reshape(long_miss_mask, long_shape)
        long_miss_mask_list.append(long_miss_mask)

        continuous_var_ = [col for col, col_type in zip(long_names_,
            long_types_) if col_type[0] in ('real', 'pos', 'gamma', 'mse')]

        ep_l = len(continuous_var_)

        if ep_l > 0:
            # Filling nans will only play a role if an imputation layer is not being used
            # Filling nan values of the continuous variables with the median
            median_dict = long_data_[continuous_var_].median()
            long_data_[continuous_var_] = long_data_[continuous_var_].fillna(value=median_dict)

            f_names = np.setdiff1d(long_names_, continuous_var_)
        else:
            f_names = long_names_

        if len(f_names) > 0:
            # Filling nan values of the categorical variables with the mode
            mode_dict = long_data_[f_names].mode().iloc[0]
            long_data_[f_names] = long_data_[f_names].fillna(value=mode_dict)
        long_data_ = torch.from_numpy(long_data_[long_names_].values)

        # Filling nan values of the categorical variables with 0
        # long_data = torch.nan_to_num(long_data, nan=0.0)
        long_data_ = torch.reshape(long_data_, long_shape).float()
        long_data_list.append(long_data_)

    return long_miss_mask_list, long_data_list, long_Ts

def log_scaler(long_data, long_names, long_types, log_scaler='log1p'):
    for i in range(len(long_names)):
        continuous_var_ = [col for col, col_type in zip(long_names[i],
            long_types[i]) if col_type[0] in ('real', 'mse')]

        if len(continuous_var_) > 0 and log_scaler == 'log1p':
            long_data[i][continuous_var_] = np.log1p(long_data[i][continuous_var_])
        elif len(continuous_var_) > 0 and log_scaler == 'log':
            long_data[i][continuous_var_] = long_data[i][continuous_var_] + 1e-12
            long_data[i][continuous_var_] = np.log(long_data[continuous_var_])
    return long_data

def scale_data(config, data, names, types, path_, HIVAE_IL=False):

    continuous_var, var_medians, var_iqrs, scale, datamin, min_ = [], [], [], [], [], []
    if config.mode == 'train':
        for i in range(len(names)):
            continuous_var_ = [
                col for col, col_type in zip(names[i], types[i])
                if col_type[0] in ('real', 'pos', 'gamma', 'mse')
            ]
            if HIVAE_IL:
                continuous_var += continuous_var_
                continue

            if len(continuous_var_) > 0 and config.scaler != 'none':
                scaler = RobustScaler() if config.scaler == 'robust' else MinMaxScaler()
                scaler.fit(data[i][continuous_var_])

                data[i][continuous_var_] = scaler.transform(data[i][continuous_var_])
                continuous_var += continuous_var_

                if config.scaler == 'robust':
                    var_medians += scaler.center_.tolist()
                    var_iqrs += scaler.scale_.tolist()
                elif config.scaler == 'minmax':
                    scale += scaler.scale_.tolist()
                    datamin += scaler.data_min_.tolist()
                    min_ += scaler.min_.tolist()

        if config.scaler == 'robust':
            scaling_stats = {
                'columns': continuous_var,
                'median': var_medians,
                'iqr': var_iqrs
            }
        elif config.scaler == 'minmax':
            scaling_stats = {
                'columns': continuous_var,
                'scale': scale,
                'datamin': datamin,
                'min_': min_
            }
        else:
            scaling_stats = {
                'columns': continuous_var,
                'scale': scale
            }
        with open(path_, 'w') as f:
            json.dump(scaling_stats, f)
    else:
        with open(path_, 'r') as f:
            scaling_stats = json.load(f)

        continuous_var = scaling_stats['columns']
        if config.scaler == 'robust':
            var_medians = np.array(scaling_stats['median'])
            var_iqrs = np.array(scaling_stats['iqr'])
        elif config.scaler == 'minmax':
            scale = np.array(scaling_stats['scale'])
            datamin = np.array(scaling_stats['datamin'])
            min_ = np.array(scaling_stats['min_'])

        ep_i = 0
        for i in range(len(names)):
            if HIVAE_IL:
                continue
            continuous_var_ = [
                col for col, col_type in zip(names[i], types[i])
                if col_type[0] in ('real', 'pos', 'gamma', 'mse')
            ]
            ep_l = len(continuous_var_)
            if ep_l > 0:
                if config.scaler == 'robust':
                    data[i][continuous_var_] = (
                        data[i][continuous_var_] - var_medians[ep_i:ep_i + ep_l]
                    ) / var_iqrs[ep_i:ep_i + ep_l]
                elif config.scaler == 'minmax':
                    data[i][continuous_var_] = (
                        (data[i][continuous_var_] - datamin[ep_i:ep_i + ep_l])
                        * scale[ep_i:ep_i + ep_l]
                        + min_[ep_i:ep_i + ep_l]
                    )
                ep_i += ep_l

    return data, scaling_stats


# According to the protocol:
# Subjects will receive 400 mg solanezumab or placebo via IV infusion every 4 weeks for at least
# 2 doses,  followed by 800 mg every 4 weeks for at least 2 doses, and then 1600 mg every 4 weeks
# until the conclusion of the study
def assign_a4_dose_protocol(t):
    if t in [0, 4]:
        return 400
    elif t in [8, 12]:
        return 800
    elif t >= 16 and t % 4 == 0:
        return 1600
    else:
        return 0 