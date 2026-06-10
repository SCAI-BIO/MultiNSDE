import sys
sys.path.append('../../')
import os
import warnings
import numpy as np
import torch
import torchsde
from Networks.modular_encoders import Mod_Encoder_BL_HIVAE
from Networks.sdes import get_nsde_class
from Networks.long_decoders import Decoder_Dist
from Common_Functions.models.utils import print_network
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, dataloader, dataloader_val=None):

        self.config = config

        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        # For optuna
        self.dataloader_val = dataloader_val

        # Steps time are the same
        self.T, self.Tmax = self.dataloader.dataset.get_T()
        self.T, self.Tmax = self.T.to(self.device), self.Tmax.to(self.device)

        var_names_long, var_names_static = self.dataloader.dataset.get_var_names()
        self.var_names_long = var_names_long
        self.var_names_static = var_names_static

        self.method = self.config.method_solver
        self.rtol = self.config.rtol
        self.atol = self.config.atol
        self.build_model()

    def build_model(self):

        sdata_implayer = (self.dataloader.dataset.get_static() if
                        self.config.hivae_implayer else None)
        static_types = self.dataloader.dataset.get_static_types()
        self.static_types = static_types
        scaling_stat_stats = self.dataloader.dataset.scaling_stat_stats 

        ldata_implayer = (self.dataloader.dataset.get_XW() if
                         self.config.long_implayer else [[None] * self.config.num_lenc])
        self.long_types = self.dataloader.dataset.get_long_types()

        self.ordreg_ids = [np.where(np.isin(long_type[:, 0], ['cat', 'ord']))[0] for long_type in self.long_types]
        self.cont_long_ids = [np.where(np.isin(long_type[:, 0], ['real', 'pos', 'mse', 'gamma']))[0] for long_type in self.long_types]

        scaling_long_stats = self.dataloader.dataset.scaling_long_stats

        self.Encoder = Mod_Encoder_BL_HIVAE(
            self.config, self.static_types,
            ldata_implayer=ldata_implayer,
            sdata_implayer=sdata_implayer,
            scaling_static_stats=scaling_stat_stats).to(self.device)

        self.DL = get_nsde_class(self.config).to(self.device)
        # Longitudinal Decoder
        self.L_Dec = Decoder_Dist(self.config, self.long_types,
                                    scaling_long_stats=scaling_long_stats).to(self.device)
        
        if self.config.mode in ['train', 'cv']:
            self.get_optimizer()

        print('Models were built')

        # It does not matter if it is for continue training
        #   or for loading the model to generate predictions
        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15

        if self.config.mode in ['train', 'cv']:
            self.set_nets_train()
        else:
            self.set_nets_eval()
        self.print_models()

    def get_optimizer(self):
        params = list(self.Encoder.parameters()) + list(self.L_Dec.parameters()) + \
            list(self.DL.parameters())

        self.optimizer = torch.optim.Adam(params, self.config.lr)

    def load_models(self, best=False):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'))
            epoch = weights['Epoch']         
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        self.best_loss = weights['Avg_Loss']
        # To be able to load DL correctly using SDE and NCDE...
        # Not happy with the current solutions but I have not found other sofar
        strict=False if 'NODE' not in self.config.type_dynamics_lerner else True
        self.DL.load_state_dict(weights['DL'], strict=strict)
        self.Encoder.load_state_dict(weights['Enc'])
        self.L_Dec.load_state_dict(weights['L_Dec'])
        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])
        
        print('Models have loaded from epoch:', epoch)
        if self.config.mode == 'train':
            epoch += 1
        self.config.epoch_init = epoch

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['DL'] = self.DL.state_dict()
        weights['Enc'] = self.Encoder.state_dict()
        weights['L_Dec'] = self.L_Dec.state_dict()
        weights['Opt'] = self.optimizer.state_dict()
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

        enc_name = 'BL Encoder'
        dec_name = 'DIST Decoder'
        print_network(self.Encoder, enc_name)
        print_network(self.L_Dec, dec_name)
        print_network(self.DL, self.config.type_dynamics_lerner)

    def set_nets_train(self):
        self.Encoder.train()
        self.L_Dec.train()
        self.DL.train()

    def set_nets_eval(self):
        for param in self.Encoder.parameters():
            param.requires_grad = False

        for param in self.DL.parameters():
            param.requires_grad = False

        for param in self.L_Dec.parameters():
            param.requires_grad = False

        self.Encoder.eval()
        self.L_Dec.eval()
        self.DL.eval()

    def run_ODE(self, z_init):

        if self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        if self.config.ANDE:
            z_init_anode = torch.zeros(z_init.size(0), self.config.ANDE_dim).to(self.device)
            z_init = torch.cat((z_init, z_init_anode), -1)

        pred_z = odeint(self.DL, z_init, self.T, rtol=self.rtol,
                        atol=self.atol, method=self.method).permute(1, 0, 2)
        return pred_z

    def run_SDE(self, z_init):
        if self.config.ANDE:
            z_init_anode = torch.zeros(z_init.size(0), self.config.ANDE_dim).to(self.device)
            z_init = torch.cat((z_init, z_init_anode), -1)

        if self.method == "euler":
            print("euler")
            pred_z = torchsde.sdeint(self.DL, z_init, self.T,
                                 dt=0.001, method=self.method,
                                 adaptive=False).permute(1, 0, 2)
        else:
            pred_z = torchsde.sdeint(self.DL, z_init, self.T, rtol=self.rtol,
                        atol=self.atol, method=self.method).permute(1, 0, 2)
        return pred_z