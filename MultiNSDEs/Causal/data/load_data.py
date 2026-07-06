import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import torch
from .load_A4 import load_data_A4
from .load_DATATOP import load_data_DATATOP
from .A4_dataset import A4Dataset
from .DATATOP_dataset import DATATOP_Dataset
from Common_Functions.data.samplers import *

def load_dataset(config):

    if config.dataset == 'A4_Causal':
        data = load_data_A4(config)
        config = data[0]
        data = data[1:]
        dataset = A4Dataset(config, data)
    elif config.dataset == 'DATATOP_Causal':
        data = load_data_DATATOP(config)
        config = data[0]
        data = data[1:]
        dataset = DATATOP_Dataset(config, data)
    else:
        raise NotImplementedError("======= DATASET NOT IMPLEMENTED YET ========")

    config.n_long_var = [len(sublist) for sublist in dataset.var_names_long]
    config.n_RHS_Feat = dataset.n_RHS_Feat
    config.DRHS_PBO_slide = dataset.DRHS_PBO_slide
    if config.mode == 'train' and config.inv_ic_loss:
        config, dataloader = get_loader_domain_adapt(config, dataset)
    else:
        config, dataloader = get_loader(config, dataset)
    return config, dataloader

def load_cv_datasets(config):

    if config.dataset == 'A4_Causal':
        data = load_data_A4(config)
        config = data[0]
        data = data[1:]
        data_train, data_val = split_by_train(data)
        train_dataset = A4Dataset(config, data_train)
    elif config.dataset == 'DATATOP_Causal':
        data = load_data_DATATOP(config)
        config = data[0]
        data = data[1:]
        data_train, data_val = split_by_train(data)
        train_dataset = DATATOP_Dataset(config, data_train)
    else:
        raise NotImplementedError("======= DATASET NOT IMPLEMENTED YET ========")

    config.mode = 'cv' # being sure that the if inside the dataset always works
    if config.dataset == 'A4_Causal':
        test_dataset = A4Dataset(config, data_val)
    elif config.dataset == 'DATATOP_Causal':
        test_dataset = DATATOP_Dataset(config, data_val)
    config.mode = 'cv'

    config.n_long_var = [len(sublist) for sublist in train_dataset.var_names_long]
    config.n_RHS_Feat = train_dataset.n_RHS_Feat
    config.DRHS_PBO_slide = train_dataset.DRHS_PBO_slide
    if config.inv_ic_loss:
        config, dataloader = get_loader_domain_adapt(config, train_dataset)
    else:
        config, dataloader = get_loader(config, train_dataset, fitting='cv_train')
    config, dataloader_val = get_loader(config, test_dataset, fitting='cv_val')
    return config, dataloader, dataloader_val

def get_loader(config, dataset, fitting='normal'):
    if fitting == 'cv_train':
        shuffle = True
        batch_size = config.batch_size
    elif fitting == 'cv_val':
        shuffle = False
        batch_size = len(dataset.long_data[0])
    else:
        batch_size = config.batch_size if config.mode == 'train' else len(dataset.long_data[0])
        shuffle = True if config.mode == 'train' else False

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = False)
    return config, dataloader

def get_loader_domain_adapt(config, dataset):

    sampler = BalancedNonPlaceboSampler(dataset, batch_size=config.batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_sampler = sampler)
    return config, dataloader

def split_df(df):
    train = df[df["TRAIN"] == 1].reset_index(drop=True)
    test  = df[df["TRAIN"] == 0].reset_index(drop=True)
    return train, test

def split_by_train(x):

    if isinstance(x, pd.DataFrame):
        return split_df(x)

    if isinstance(x, np.ndarray):
        return x, x

    if isinstance(x, (list, tuple)):
        train_list, test_list = [], []
        for i, item in enumerate(x):
            tr, te = split_by_train(item)
            train_list.append(tr)
            test_list.append(te)
        return type(x)(train_list), type(x)(test_list)

    raise TypeError(f"Non-supported type: {type(x)}")
