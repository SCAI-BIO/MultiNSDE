import numpy as np
import torch
from .load_A4 import load_data_A4
from .load_PROACT import load_data_PROACT
from .load_DATATOP import load_data_DATATOP
from .A4_dataset import A4Dataset
from .PROACT_dataset import PROACT_Dataset
from .DATATOP_dataset import DATATOP_Dataset

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
    elif config.dataset == 'DATATOP':
        data = load_data_DATATOP(config)
        config = data[0]
        data = data[1:]
        if only_data:
            return config, data
        dataset = DATATOP_Dataset(config, data)
    else:
        print('============================================')
        print('======= DATASET NOT IMPLEMENTED YET=========')
        print('============================================')
    config.n_long_var = len(dataset.var_names_long)
    config, dataloader = get_loader(config, dataset)

    return config, dataloader

def get_loader(config, dataset):
    batch_size = config.batch_size if config.mode == 'train' else len(dataset.long_data)
    shuffle = True if config.mode == 'train' else False

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = False)
    return config, dataloader