import imp
import os
import warnings
import numpy as np
import torch
from Networks.longitudinal import (
    RevRNNEncoder, RNNEncoder,
    RNNDecoder, Decoder)
from Networks.hivae import HIVAE
from Networks.odes import (
    RNNODEEncoder, LatentODE)
from utils import print_network
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, dataloader):

        self.config = config

        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader

        # Steps time are the same
        self.T, self.Tmax = self.dataloader.dataset.get_T()
        self.T, self.Tmax = self.T.to(self.device), self.Tmax.to(self.device)

        self.method = self.config.method_solver
        self.rtol = self.config.rtol
        self.atol = self.config.atol
        self.build_model()

    def build_model(self):

        # ODE and Static NN if Static data is available
        if self.config.static_data:
            data_implayer = (self.dataloader.dataset.get_static() if
                            self.config.hivae_implayer else None)
            _, static_types, _ = self.dataloader.dataset.get_static()
            self.static_types = static_types

            self.HIVAE = HIVAE(self.config, static_types,
                    data_implayer=data_implayer).to(self.device)
        else:
            self.static_types = None
        self.ODE = LatentODE(self.config).to(self.device)
        self.long_types = self.dataloader.dataset.get_long_types()
        self.ordreg_ids = np.where(np.isin(self.long_types[:, 0], ['cat', 'ord']))[0]
        self.cont_long_ids = np.where(np.isin(self.long_types[:, 0], ['real', 'pos', 'mse', 'gamma']))[0]

        data_implayer = (self.dataloader.dataset.get_XW() if
                         self.config.long_implayer else None)
        

        # Longitudinal Encoder
        if 'ODE' in self.config.type_lenc:
            self.L_Enc = RNNODEEncoder(self.config,
                        data_implayer=data_implayer)
        elif self.config.rev_lenc:
            self.L_Enc = RevRNNEncoder(self.config,
                        data_implayer=data_implayer)
        else:
            self.L_Enc = RNNEncoder(self.config,
                        data_implayer=data_implayer)

        # Longitudinal Decoder
        if self.config.type_dec == 'MLP':
            self.L_Dec = Decoder(self.config)
        else:
            self.L_Dec = RNNDecoder(self.config)

        if 'train' in self.config.mode:
            self.get_optimizer()

        print('Models were built')

        # It does not matter if it is for continue training
        #   or for loading the model to generate predictions
        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15

        self.print_models()
        if 'train' in self.config.mode:
            self.set_nets_train()
        else:
            self.set_nets_eval()

    def get_optimizer(self):
        params = list(self.L_Enc.parameters()) + list(self.L_Dec.parameters()) + \
            list(self.ODE.parameters())
        if self.config.static_data:
            params = params + list(self.HIVAE.parameters())

        self.optimizer = torch.optim.Adam(params, self.config.lr)

    def load_models(self, best=False):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'))
            self.config.epoch_init = weights['Epoch']
            epoch = self.config.epoch_init            
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        self.best_loss = weights['Avg_Loss']
        self.ODE.load_state_dict(weights['ODE'])
        self.L_Enc.load_state_dict(weights['L_Enc'])
        self.L_Dec.load_state_dict(weights['L_Dec'])
        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])

        if self.config.static_data:
            self.HIVAE.load_state_dict(weights['HIVAE'])
        
        print('Models have loaded from epoch:', epoch)

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['ODE'] = self.ODE.state_dict()
        weights['L_Enc'] = self.L_Enc.state_dict()
        weights['L_Dec'] = self.L_Dec.state_dict()
        weights['Opt'] = self.optimizer.state_dict()

        if self.config.static_data:
            weights['HIVAE'] = self.HIVAE.state_dict()

        weights['Avg_Loss'] = loss
        if best:
            weights['Epoch'] = epoch
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Best.pth'))
        else:
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        print('Models have been saved')

    def print_models(self):

        enc_name = '%s Encoder'%(self.config.type_lenc)
        dec_name = '%s Decoder'%(self.config.type_dec)
        print_network(self.L_Enc, enc_name)
        print_network(self.L_Dec, dec_name)
        print_network(self.ODE, 'NODE')
        if self.config.static_data:
            print_network(self.HIVAE, 'HIVAE')

    def set_nets_train(self):
        self.L_Enc.train()
        self.L_Dec.train()
        self.ODE.train()
        if self.config.static_data:
            self.HIVAE.train()

    def set_nets_eval(self):

        self.L_Enc.eval()
        self.L_Dec.eval()
        self.ODE.eval()
        if self.config.static_data:
            self.HIVAE.eval()

    def run_ODE(self, z_init):

        if self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        pred_z = odeint(self.ODE, z_init, self.T, rtol=self.rtol,
                        atol=self.atol, method=self.method).permute(1, 0, 2)
        return pred_z