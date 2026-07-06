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



class DATATOP_Dataset(Dataset):
    def __init__(self, config, data):

        (long_data, long_names,
         long_types, de_rhs_df, time_event_df,
         time_event_names, static_data,
         static_names, static_types)  = data 

        self.var_names_static = static_names

        try:
            path_ = os.path.join(config.save_path_models, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_models, 'scaling_static_stats.json')
            path_bl_ = os.path.join(config.save_path_models, 'scaling_bl_stats.json')
        except:
            path_ = os.path.join(config.save_path_trial_fold, 'scaling_long_stats.json')
            path_stat_ = os.path.join(config.save_path_trial_fold, 'scaling_static_stats.json')
            path_bl_ = os.path.join(config.save_path_trial_fold, 'scaling_bl_stats.json')

        static_data, self.scaling_stat_stats = scale_data(
            config, [static_data], [static_names], [static_types],
            path_=path_stat_, HIVAE_IL=True)
        static_data, static_miss_mask, static_types = process_static(
            static_data[0], static_names, static_types)

        if config.type_hivae in ['IC_HIVAE', 'SLR_HIVAE']:
            bl_data_ = [bl_df[bl_df["TIME"] == 0] for bl_df in long_data]
            merge_cols = ["PDDOCID", "TIME", "TRAIN"]
            bl_data = bl_data_[0]
            for df in bl_data_[1:]:
                bl_data = bl_data.merge(df, on=merge_cols, how="left")

            bl_names = np.concatenate(long_names)
            bl_types = np.concatenate(long_types)
            bl_data, self.scaling_bl_stats = scale_data(
                config, [bl_data], [bl_names], [bl_types],
                path_=path_bl_, HIVAE_IL=True)
            bl_data, bl_miss_mask, bl_types = process_static(
                bl_data[0], bl_names, bl_types)
            
            static_data = torch.cat((static_data, bl_data), dim=1)
            static_miss_mask = torch.cat((static_miss_mask, bl_miss_mask), dim=1)
            static_types = np.concatenate((static_types, bl_types), axis=0)

        self.static_data = static_data
        # Patients x Num_Static_Variables
        self.static_miss_mask = static_miss_mask
        self.static_types = static_types

        # log transformation
        if config.log_scaler != 'none':
            long_data = log_scaler(long_data, long_names, long_types, log_scaler=config.log_scaler)

        unique_ptnos = np.sort(long_data[0]['PDDOCID'].unique())
        long_data, self.scaling_long_stats = scale_data(
            config, long_data, long_names, long_types, path_=path_)
        long_miss_mask_list, long_data_list, long_Ts = build_long_lists(
            long_data, long_names, long_types, unique_ptnos)

        # Time for the DE
        union_set = set().union(*[set(t.tolist()) for t in long_Ts])
        doses_times = set(de_rhs_df['TIME'].unique().tolist()) # Times begin in 0
        union_set = union_set.union(doses_times)

        # pids = de_rhs_df['PDDOCID'].unique()
        base_cols = ['PDDOCID', 'TIME']
        cols = [c for c in de_rhs_df.columns if c not in ["TRAIN", "TREAT"]]
        cols_ffil = [c for c in cols if c not in ['PDDOCID', 'TIME']]

        all_times_doses = pd.DataFrame([(pid, time) for pid in unique_ptnos for time in union_set], columns=base_cols)
        amt_df = all_times_doses.merge(de_rhs_df[cols], on=base_cols, how='left')
        amt_df = amt_df.sort_values(base_cols).reset_index(drop=True)
        # In case something goes wrong. However should be ok
        amt_df[cols_ffil] = amt_df.groupby('PDDOCID')[cols_ffil].ffill()

        sorted_times = sorted(union_set)
        base_rhs_feat = len(cols_ffil)
        arr = np.zeros((len(unique_ptnos), len(sorted_times), base_rhs_feat))

        amt_df['TIME'] = pd.Categorical(amt_df['TIME'], categories=sorted_times, ordered=True)
        for d, col in enumerate(cols_ffil):
            aux = amt_df.pivot(index='PDDOCID', columns='TIME', values=col).sort_index(axis=1)
            aux = aux.loc[unique_ptnos, sorted_times]
            arr[:, :, d] = aux.values

        # Align baseline treatment by patient once, then broadcast over all times.
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

        # Add a one-hot treatment block [TREAT_0, TREAT_1, TREAT_2, TREAT_3]
        # so DATATOP can represent placebo + 3 active treatment groups.
        n_treat = 4
        treat_block = np.zeros((len(unique_ptnos), len(sorted_times), n_treat))
        treat_vals = baseline_treat.values
        for k in range(n_treat):
            treat_block[:, :, k] = (treat_vals[:, None] == k).astype(float)

        arr = np.concatenate([arr, treat_block], axis=-1)
        treat_start = base_rhs_feat
        self.n_RHS_Feat = arr.shape[-1]

        # To know which variables needs to be used to define treatment in MIX dynamics.
        self.DRHS_PBO_slide = (slice(None), slice(treat_start + 1, treat_start + n_treat))

        if config.Val_Scenario in [2, 3, 4, 5]:
            # Counterfactual scenarios: force everyone to one treatment arm.
            forced_treat = {2: 0, 3: 1, 4: 2, 5: 3}[config.Val_Scenario]
            arr[:, :, treat_start:treat_start + n_treat] = 0
            arr[:, :, treat_start + forced_treat] = 1

        self.rhs_data = torch.tensor(arr)

        # 1 for placebo and 0 for non-placebo.
        self.placebo_flag = torch.tensor((baseline_treat == 0).astype(float).values).unsqueeze(-1).float()


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

        # T is the time used for solving the DE/NDE and is already handled by normalize_time.
        self.SUBJID = torch.tensor(unique_ptnos)

        # drug_type = self.placebo_flag.expand(-1, self.T.size(0))
        # # Done in this way because we only do have one medicament that allows to
        # # calculate PBO or TRT patients
        # self.drug_type = 1 - drug_type

        # Keep explicit DATATOP arm ids over time: 0=placebo, 1/2/3=active groups.
        self.drug_type = torch.tensor(baseline_treat.values).unsqueeze(-1).float().expand(-1, self.T.size(0))

        self.time_event_names = time_event_names
        if config.time_to_event:

            time_wide = time_event_df.pivot(index="PDDOCID", columns="Var", values="TIME").sort_index()
            event_wide = time_event_df.pivot(index="PDDOCID", columns="Var", values="Event").sort_index()
            time_wide = time_wide.reindex(unique_ptnos)
            event_wide = event_wide.reindex(unique_ptnos)
            self.Time_TE = torch.tensor(time_wide.values)
            self.Event_TE = torch.tensor(event_wide.values)
        else:
            self.Time_TE = torch.ones(len(unique_ptnos), 1) * -1
            self.Event_TE = torch.ones(len(unique_ptnos), 1) * -1
        
    def __getitem__(self, idx):
        X = [ld[idx].float() for ld in self.long_data]
        W = [ld[idx].float() for ld in self.long_miss_mask]
        T = [ld.float() for ld in self.long_Ts]
        W_DE = self.long_de_miss_mask[idx, :]
        D_Data = self.rhs_data[idx, :].float()
        Placebo_Flag = self.placebo_flag[idx, :]
        PTNO = self.SUBJID[idx]
        Drug_type = self.drug_type[idx]
        Time_TE = self.Time_TE[idx, :]
        Event_TE = self.Event_TE[idx, :]
        s_data = self.static_data[idx, :]
        S_Mask = self.static_miss_mask[idx, :]
        return (X, W, T, W_DE, D_Data, Time_TE, Event_TE, s_data.float(),
                S_Mask, Drug_type, Placebo_Flag, PTNO)

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
        return self.var_names_long, self.var_names_static, self.time_event_names

    def get_static_types(self):
        return self.static_types

    def get_long_types(self):
        return self.long_types

    def get_pbo_flags(self):
        return self.placebo_flag.squeeze().numpy()
