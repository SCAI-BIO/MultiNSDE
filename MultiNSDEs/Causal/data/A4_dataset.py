import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from Common_Functions.data.helpers_datasets import (
    process_static, build_long_lists,
    scale_data, log_scaler)
from Common_Functions.data.load_data_helpers import normalize_time

class A4Dataset(Dataset):
    def __init__(self, config, data):

        (long_data, long_names,
         long_types, de_rhs_df, static_data,
         static_names, static_types, scores)  = data

        if config.mode == 'cv':
            config.mode = 'train' if static_data.TRAIN[0] == 1 else 'val'

        self.var_names_static = static_names

        try:
            path_ = os.path.join(config.save_path_models, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_models, 'scaling_static_stats.json')
        except:
            path_ = os.path.join(config.save_path_trial_fold, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_trial_fold, 'scaling_static_stats.json')

        static_data, self.scaling_stat_stats = scale_data(
            config, [static_data], [static_names], [static_types],
            path_=path_stat_, HIVAE_IL=True)
        static_data, static_miss_mask, static_types = process_static(
            static_data[0], static_names, static_types)

        self.static_data = static_data
        # Patients x Num_Static_Variables
        self.static_miss_mask = static_miss_mask
        self.static_types = static_types

        # log transformation
        if config.log_scaler != 'none':
            long_data = log_scaler(long_data, long_names, long_types, log_scaler=config.log_scaler)

        unique_ptnos = long_data[0]['NEW_ID'].unique()
        long_data, self.scaling_long_stats = scale_data(
            config, long_data, long_names, long_types, path_=path_)
        long_miss_mask_list, long_data_list, long_Ts = build_long_lists(
            long_data, long_names, long_types, unique_ptnos)

        # 1 is PBO and 0 is TRT
        placebo_flag = de_rhs_df[de_rhs_df.TIME == 0]
        self.placebo_flag = torch.tensor(placebo_flag.PB.values).unsqueeze(-1).float()

        # Time for the DE
        union_set = set().union(*[set(t.tolist()) for t in long_Ts])
        doses_times = set(de_rhs_df['TIME'].unique().tolist()) # Times begin in 0
        union_set = union_set.union(doses_times)

        self.rhs_data = (1 - self.placebo_flag).unsqueeze(1).repeat(1, len(union_set), 1)

        # To know which variables needs to be use to define the factor in the MIXs DEs
        self.n_RHS_Feat = self.rhs_data.shape[-1]
        self.DRHS_PBO_slide = (slice(None), slice(0, 1))      

        T = torch.tensor(sorted(union_set)).float()
        df_de_miss = pd.DataFrame({'DETIME': T})
        de_miss = []
        for i in range(len(long_Ts)):
            name = 'TIME_%d'%(i)
            df_de_miss[name] = df_de_miss['DETIME'].apply(lambda x: 1 if x in long_Ts[i].numpy() else 0)

            de_miss_i = torch.from_numpy(df_de_miss[name].repeat(len(long_names[i])).values.reshape(-1, len(long_names[i])))
            de_miss.append(de_miss_i)
        long_de_miss_mask = torch.cat(de_miss, -1)
        long_de_miss_mask = long_de_miss_mask.unsqueeze(0).expand(len(unique_ptnos), -1, -1)

        self.long_data = long_data_list
        self.long_miss_mask = long_miss_mask_list
        self.long_Ts, self.tmax_list, self.Tmax, self.T = normalize_time(config, long_Ts, T)

        self.var_names_long = long_names
        self.long_types = long_types
        self.long_de_miss_mask= long_de_miss_mask

        # OR Scores
        self.Pi_Score = torch.from_numpy(scores["Propensity_Score"].values)
        self.Obs_Score = torch.from_numpy(scores["Observation_Score"].values)

        # T is the time that we will use for solving the NDE
        self.SUBJID = torch.from_numpy(long_data[0].NEW_ID.values).unique()

        drug_type = self.placebo_flag.expand(-1, self.T.size(0))
        # Done in this way because we only do have one medicament that allows to
        # calculate PBO or TRT patients
        self.drug_type = 1 - drug_type

    def __getitem__(self, idx):
        X = [ld[idx].float() for ld in self.long_data]
        W = [ld[idx].float() for ld in self.long_miss_mask]
        T = [ld.float() for ld in self.long_Ts]
        W_DE = self.long_de_miss_mask[idx, :]
        D_Data = self.rhs_data[idx, :].float()
        Placebo_Flag = self.placebo_flag[idx, :]
        PTNO = self.SUBJID[idx]
        Pi_Score = self.Pi_Score[idx]
        Obs_Score = self.Obs_Score[idx]
        s_data = self.static_data[idx, :]
        S_Mask = self.static_miss_mask[idx, :]
        return (X, W, T, W_DE, D_Data, s_data.float(), S_Mask,
                Pi_Score, Obs_Score, Placebo_Flag, PTNO)

    def __len__(self):
        return len(self.long_data[0])

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