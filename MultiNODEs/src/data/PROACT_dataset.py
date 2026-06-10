import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .helpers_datasets import (
    process_static, build_long_lists,
    scale_data, log_scaler)
from .load_data_helpers import (normalize_time)

class PROACT_Dataset(Dataset):
    def __init__(self, config, data):

        (long_data, long_names,
         long_types, de_rhs_df, static_data,
         static_names, static_types)  = data

        self.static = config.static_data
        self.var_names_static = static_names


        path_ = os.path.join(config.save_path_models, 'scaling_long_stats.json')
        path_stat_ = os.path.join(config.save_path_models, 'scaling_static_stats.json')

        if self.static:
            static_data, self.scaling_stat_stats = scale_data(
                config, [static_data], [static_names], [static_types],
                path_=path_stat_, HIVAE_IL=True)
            static_data, static_miss_mask, static_types = process_static(
                static_data[0], static_names, static_types)
        else:
            static_miss_mask = None

        self.static_data = static_data
        # Patients x Num_Static_Variables
        self.static_miss_mask = static_miss_mask
        self.static_types = static_types

        long_data = [long_data]
        long_names = [long_names]
        long_types = [long_types]

        # log transformation
        if config.log_scaler != 'none':
            long_data = log_scaler(long_data, long_names, long_types, log_scaler=config.log_scaler)

        unique_ptnos = np.sort(long_data[0]['subject_id'].unique())
        long_data, self.scaling_long_stats = scale_data(
            config, long_data, long_names, long_types, path_=path_)
        long_miss_mask_list, long_data_list, long_Ts = build_long_lists(
            long_data, long_names, long_types, unique_ptnos)

        # 1 is PBO and 0 is TRT
        placebo_flag = de_rhs_df[de_rhs_df.TIME == 0]
        self.placebo_flag = torch.tensor(placebo_flag.PBO.values).unsqueeze(-1).float()

        T = long_Ts[0]

        self.long_data = long_data_list[0]
        self.long_miss_mask = long_miss_mask_list[0]
        #self.long_Ts, self.tmax_list = zip(*[(lt / max(lt), max(lt)) for lt in long_Ts])
        self.var_names_long = long_names[0]
        self.long_types = long_types[0]

        self.long_Ts, self.tmax_list, self.Tmax, self.T = normalize_time(config, long_Ts, T)


        # T is the time that we will use for solving the NDE
        #self.Tmax = max(T)
        #self.T = T / self.Tmax
        self.SUBJID = torch.from_numpy(long_data[0].subject_id.values).unique()

        drug_type = self.placebo_flag.expand(-1, self.T.size(0))
        # Done in this way because we only do have one medicament that allows to
        # calculate PBO or TRT patients
        self.drug_type = 1 - drug_type


    def __getitem__(self, idx):
        X = self.long_data[idx, ...]
        W = self.long_miss_mask[idx, ...]
        Placebo_Flag = self.placebo_flag[idx, :]
        PTNO = self.SUBJID[idx]
        Drug_type = self.drug_type[idx]

        if self.static:
            s_data = self.static_data[idx, :]
            S_Mask = self.static_miss_mask[idx, :]
            return (X, W, s_data.float(),
                    S_Mask, Drug_type, Placebo_Flag, PTNO)
        else:
            return (X, W, Drug_type, Placebo_Flag, PTNO)

    def __len__(self):
        return len(self.long_data)

    def get_T(self):
        return self.T, self.Tmax

    def get_TEncs(self):
        return self.long_Ts, self.tmax_list

    def get_XW(self):
        return self.long_data, self.long_miss_mask, self.long_Ts

    def get_static(self):
        return self.static_data.float(), self.static_types, self.static_miss_mask

    def get_var_names(self):
        return self.var_names_long, self.var_names_static

    def get_static_types(self):
        return self.static_types

    def get_long_types(self):
        return self.long_types

    def get_pbo_flags(self):
        return self.placebo_flag.squeeze().numpy()
