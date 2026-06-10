import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import torch
from .load_A4 import load_data_A4
from .load_PROACT import load_data_PROACT
from .A4_dataset import A4Dataset
from .PROACT_dataset import PROACT_Dataset
from Common_Functions.data.samplers import *

def load_dataset(config, only_data=False):

    if config.dataset == 'A4':
        data = load_data_A4(config)
        config = data[0]
        data = data[1:]
        if only_data:
            return config, data
        dataset = A4Dataset(config, data)
    elif config.dataset == 'PROACT':
        data = load_data_PROACT(config)
        config = data[0]
        data = data[1:]
        if only_data:
            return config, data
        dataset = PROACT_Dataset(config, data)
    else:
        print('============================================')
        print('======= DATASET NOT IMPLEMENTED YET=========')
        print('============================================')

    config.n_long_var = [len(sublist) for sublist in dataset.var_names_long]
    config.n_RHS_Feat = dataset.n_RHS_Feat
    config.DRHS_PBO_slide = dataset.DRHS_PBO_slide
    if config.mode == 'train' and config.inv_ic_loss:
        config, dataloader = get_loader_domain_adapt(config, dataset)
    else:
        config, dataloader = get_loader(config, dataset)
    return config, dataloader

def get_loader(config, dataset):
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
