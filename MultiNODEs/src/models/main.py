#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
from parser import base_parser
from utils import define_logs
from train import Train
from validation import Validation
import sys
sys.path.append('../')
from data.load_data import load_dataset
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

def main(config):

    config, dataloader = load_dataset(config)
    print('Mode:', config.mode)

    if 'train' in config.mode:
        Train(config, dataloader)
        print('Training has finished')
    else:
        Validation(config, dataloader)
        print('Validation has finished')


if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in config.GPU.split(',')]
    else:
        config.GPU = False

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    if 'A4' == config.dataset:
        config.longdata_fname = [i for i in config.longdata_fname.split(',')]
        config.longtypes_fname = [i for i in config.longtypes_fname.split(',')]
        config.real_longtypes_fname = [i for i in config.real_longtypes_fname.split(',')]
        config.num_lenc = len(config.longdata_fname)

    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.dataset, 
                                    config.exp_name, 'Fold%d'%(config.train_fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')

    # Print the parser options of the current experiment
    # in a txt file saved in repo/models/exp_name/log.txt
    if 'train' in config.mode:
        define_logs(config)

    main(config)
