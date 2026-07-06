import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .load_data_helpers import normalize_time
from .helpers_datasets import (
    process_static, build_long_lists,
    scale_data, log_scaler)


class DATATOP_Dataset(Dataset):
    def __init__(self, config, data):

        (long_data, long_names,
         long_types, de_rhs_df, static_data,
         static_names, static_types) = data

        self.static = config.static_data
        self.var_names_static = static_names

        try:
            path_ = os.path.join(config.save_path_models, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_models, 'scaling_static_stats.json')
        except Exception:
            path_ = os.path.join(config.save_path_trial_fold, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_trial_fold, 'scaling_static_stats.json')

        if self.static:
            static_data, self.scaling_stat_stats = scale_data(
                config, [static_data], [static_names], [static_types],
                path_=path_stat_, HIVAE_IL=True)
            static_data, static_miss_mask, static_types = process_static(
                static_data[0], static_names, static_types)
        else:
            static_miss_mask = None

        self.static_data = static_data
        self.static_miss_mask = static_miss_mask
        self.static_types = static_types

        # Make DATATOP longitudinal structure consistent with PROACT: single block.
        if not isinstance(long_data, list):
            long_data = [long_data]

        # long_names/long_types must be list-of-blocks for helpers (same contract as PROACT)
        if isinstance(long_names, np.ndarray) or (len(long_names) > 0 and isinstance(long_names[0], str)):
            long_names = [long_names]

        if isinstance(long_types, np.ndarray) or (
            len(long_types) > 0 and not isinstance(long_types[0], (list, tuple, np.ndarray))
        ):
            long_types = [long_types]

        # log transformation
        if config.log_scaler != 'none':
            long_data = log_scaler(long_data, long_names, long_types, log_scaler=config.log_scaler)

        unique_ptnos = np.sort(long_data[0]['PDDOCID'].unique())
        long_data, self.scaling_long_stats = scale_data(
            config, long_data, long_names, long_types, path_=path_)
        long_miss_mask_list, long_data_list, long_Ts = build_long_lists(
            long_data, long_names, long_types, unique_ptnos)

        # Preserve DATATOP's original 4-arm treatment code for downstream scoring,
        # while keeping the placebo flag for existing compatibility paths.
        baseline_treat_df = (
            de_rhs_df[de_rhs_df['TIME'] == 0][['PDDOCID', 'TREAT']]
            .drop_duplicates(subset=['PDDOCID'])
            .set_index('PDDOCID')
        )
        baseline_treat = baseline_treat_df.reindex(unique_ptnos)['TREAT']
        if baseline_treat.isna().any():
            missing = int(baseline_treat.isna().sum())
            raise ValueError(f"Missing baseline TREAT value for {missing} patient(s) in DATATOP doses file.")

        baseline_treat = baseline_treat.astype(int)
        self.placebo_flag = torch.tensor((baseline_treat == 0).astype(float).values).unsqueeze(-1).float()

        T = long_Ts[0]

        self.long_data = long_data_list[0]
        self.long_miss_mask = long_miss_mask_list[0]
        self.var_names_long = long_names[0]
        self.long_types = long_types[0]

        self.long_Ts, self.tmax_list, self.Tmax, self.T = normalize_time(config, long_Ts, T)

        self.SUBJID = torch.from_numpy(unique_ptnos)

        drug_type = torch.tensor(baseline_treat.values, dtype=torch.float32).unsqueeze(-1)
        self.drug_type = drug_type.expand(-1, self.T.size(0))

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
