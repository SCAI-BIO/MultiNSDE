import pandas as pd
import sys
import torch
import os
sys.path.append('../')

def read_csv_values(path, sep=',', header=None,
                    index_col=False, engine='python'):
                       
    return pd.read_csv(path, sep=sep, header=header,
                       index_col=index_col, engine=engine)

def process_time_groups(time_group):
    duplicates = time_group.duplicated(subset='PTNO', keep=False)
    # Taking the median for different measurements at Time = t
    if duplicates.any():
        median_values = time_group[duplicates].groupby('PTNO').median().reset_index()
        time_group = time_group.drop(time_group[duplicates].index)
        time_group = pd.concat([time_group, median_values], ignore_index=True)
    return time_group


def get_data_fold(config, data_df):
    n_columns = data_df.keys()
    n_fold = 'Fold%d'%(config.train_fold)
    use_train_split = (
        config.train_fold == 0
        or config.mode == 'train'
        or config.Val_Scenario == 0
        or (config.dataset != 'DATATOP' and config.Val_Scenario in [4, 5])
    )

    # Use not all time points for training the model
    # ...... explain how it works
    # extract maximum day and set it to validation .....
    if config.extrapolation: #Use all time points except last for training
        if config.mode == 'train':
            data_df[n_fold] = 1  
        else:
            data_df[n_fold] = 0 

        if config.dataset == 'A4' and 'TIME' in data_df.columns:
            data_df.loc[data_df['TIME'] == 240, n_fold] = 0
        elif config.dataset == 'PROACT' and 'TIME' in data_df.columns:
            data_df.loc[data_df.groupby('subject_id')['TIME'].transform('max') == data_df['TIME'], n_fold] = 0
        elif config.dataset == 'DATATOP' and 'LAST_VISIT_FLAG' in data_df.columns:
            data_df.loc[data_df['LAST_VISIT_FLAG'] == 1, n_fold] = 0
            
    for n_c in n_columns:
        if 'Fold' in n_c and n_c != n_fold:
            data_df.drop(n_c, inplace=True, axis=1)
    data_df = data_df.rename(columns={n_fold: 'TRAIN'})

    # Optuna
    if hasattr(config, 'studies_save_path'):
        data_df = data_df[data_df.TRAIN == 1]
    elif use_train_split:
        if config.extrapolation:
            data_df['TRAIN'] = 1
            if config.dataset == 'PROACT' and 'TIME' in data_df.columns:
                if 'Month' in config.longdata_fname:
                    data_df.loc[data_df['TIME'] == 17, 'TRAIN'] = 0
                else:
                    data_df.loc[data_df['TIME'] == 500, 'TRAIN'] = 0
            elif config.dataset == 'A4' and 'TIME' in data_df.columns:
                data_df.loc[data_df['TIME'] == 240, 'TRAIN'] = 0
            elif config.dataset == 'DATATOP' and 'LAST_VISIT_FLAG' in data_df.columns:
                data_df.loc[data_df['LAST_VISIT_FLAG'] == 1, 'TRAIN'] = 0

        data_df = data_df[data_df.TRAIN == 1]
    elif config.train_fold: # any fold different than 0
        data_df = data_df[data_df.TRAIN == 0]

    return data_df

def get_mean_std_missings(long_data, long_names):
    missing_per_time_list = []
    for df, cols in zip(long_data, long_names):
        missing_per_time = (
            df.groupby("TIME")[cols]
                .apply(lambda g: g.isna().sum().sum())
        )
        missing_per_time_list.append(missing_per_time)

    total_missing_per_time = sum(missing_per_time_list)
    total_missing_per_time = (
        missing_per_time_list[0]
        .add(missing_per_time_list[1], fill_value=0)
        .add(missing_per_time_list[2], fill_value=0))

    possible_values_per_time = (
        (long_data[0].groupby("TIME").size() * len(long_names[0]))
        .add(long_data[1].groupby("TIME").size() * len(long_names[1]), fill_value=0)
        .add(long_data[2].groupby("TIME").size() * len(long_names[2]), fill_value=0))
    percent_missing_per_time = 100 * (total_missing_per_time / possible_values_per_time)
    print(f"Mean missing %: {percent_missing_per_time.mean():.2f}%")
    print(f"STD missing %: {percent_missing_per_time.std():.2f}%")


def normalize_time(config, long_Ts, T, filename="time_max_train.txt"):

    save_dir = os.path.join(config.save_path, "models")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if config.mode == "train":

        long_Ts_norm, tmax_list = zip(*[(lt / lt.max(), lt.max()) for lt in long_Ts])
        Tmax = max(max(tmax_list), T.max().item())
        if not isinstance(Tmax, torch.Tensor):
            Tmax = torch.tensor(Tmax, dtype=T.dtype)
        with open(filepath, "w") as f:
            for tm in tmax_list:
                f.write(f"{tm}\n")
            f.write(str(Tmax.item()))

        T_norm = T / Tmax 
    else:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                tmax_list = [float(line.strip()) for line in f.readlines()]
                Tmax_train = tmax_list[-1]
                tmax_list = tmax_list[:-1] 
        else:
            Tmax_train = max([lt.max().item() for lt in long_Ts])

        long_Ts_norm = [lt / tm for lt, tm in zip(long_Ts, tmax_list)]
        Tmax = Tmax_train
        T_norm = T / Tmax 

        if not isinstance(Tmax, torch.Tensor):
            Tmax = torch.tensor(Tmax, dtype=T.dtype)

    return long_Ts_norm[0], tmax_list, Tmax, T_norm