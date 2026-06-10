import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.distributions as dist
from solver import Solver
import sys
sys.path.append('../')
from data.load_PROACT import load_only_PROACT_types
from data.load_A4 import load_only_A4_types
from val_utils import (
    unscale, transform_back)
warnings.filterwarnings('ignore')


class Validation(Solver):
    def __init__(self, config, dataloader):
        super(Validation, self).__init__(config, dataloader)
        self.scaling_long_stats = self.dataloader.dataset.scaling_long_stats
        self.scaler = self.config.scaler
        self.log_scaler = self.config.log_scaler 
        self.run()

    @torch.no_grad()
    def combine_long_preds(self, pred_x):
        i_c, e_c = 0, len(self.cont_long_ids)

        pred_x = unscale(pred_x, self.cont_long_ids,
                         i_c, e_c,
                         self.scaling_long_stats,
                         scaler=self.scaler)
        pred_x = transform_back(pred_x, self.cont_long_ids, self.log_scaler)
        pred_x[..., self.ordreg_ids] = torch.round(pred_x[..., self.ordreg_ids]).clamp(min=0)
        return pred_x

    @torch.no_grad()
    def get_pred_val(self, data):

        val_runs = self.config.nruns_ppd

        total_real_long_ = list()
        total_mask_long_ = list()
        total_sim_long_ = list()
        total_real_stat = list()
        total_mask_stat = list()
        total_rec_stat = list()
        total_Z0_Long = list()
        total_Z0_Stat = list()
        total_zinit = list()

        X, W = data[0].to(self.device), data[1].to(self.device)
        b = X.size(0)

        z0_long, Means_Long, Std_Long = self.L_Enc(X, W, self.T, val=True)
        if 'Sampling_PSD' in self.config.val_data_type:
            self.Long_Dists = dist.normal.Normal(Means_Long, Std_Long)
        
        if self.config.static_data:
            S_Data = data[2].to(self.device)
            S_Mask = data[3].to(self.device)

            z0_stat, Means_Stat, Std_Stat = self.HIVAE(
                S_Data, S_Mask, self.tau, val=True)
            if 'Sampling_PSD' in self.config.val_data_type:
                self.Stat_Dists = dist.normal.Normal(Means_Stat, Std_Stat)

        for nrun in range(val_runs):
            print('Begin---- VI sampling. Nrun', nrun + 1, 'of', val_runs)

            if 'PPD' in self.config.val_data_type:
                z0_long = []
                for _ in range(b):
                    z0_long.append(self.Long_Dists.sample())
                z0_long = torch.stack(z0_long, 0)
            elif 'PSD' in self.config.val_data_type:
                z0_long = self.Long_Dists.sample()

            total_Z0_Long.append(z0_long.unsqueeze(1))

            if self.config.static_data:
                if 'PPD' in self.config.val_data_type:
                    z0_stat = []
                    for _ in range(b):
                        z0_stat.append(self.Stat_Dists.sample())
                    z0_stat = torch.stack(z0_stat, 0)
                elif 'PSD' in self.config.val_data_type:
                    z0_stat = self.Stat_Dists.sample()

                total_Z0_Stat.append(z0_stat.unsqueeze(1))

                rec_samples = self.HIVAE.dec_samplings(
                    S_Data, S_Mask, z0_stat, self.tau)

                if self.config.ode_static_data == 'ADD_NN':
                    self.ODE.add_static_data(z0_stat)
                    z_init = z0_long
                elif self.config.ode_static_data == 'ADD':
                    z_init = z0_long + z0_stat
                else:
                    z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0

                total_real_stat.append(S_Data.unsqueeze(1))
                total_mask_stat.append(S_Mask.unsqueeze(1))
                total_rec_stat.append(rec_samples.unsqueeze(1))

            else:
                Means_Stat = Std_Stat = None
                z_init = z0_long

            pred_z = self.run_ODE(z_init) # z_init shape 1120, 59
            rec_x = self.L_Dec(pred_z) # pred_z shape 1120, 48, 59
            rec_x = self.combine_long_preds(rec_x) # rec_x shape 1120, 48, 110

            total_real_long_.append(X)
            total_mask_long_.append(W)
            total_sim_long_.append(rec_x)
            total_zinit.append(z_init.unsqueeze(1))

        # The metrics as well as the GOF plots need to be done with the
        # patient specific mean
        z0_long = Means_Long
        if self.config.static_data:
            z0_stat = Means_Stat

            REC_STAT = self.HIVAE.dec_samplings(
                S_Data, S_Mask, z0_stat, self.tau).detach().cpu()

            if self.config.ode_static_data == 'ADD_NN':
                self.ODE.add_static_data(z0_stat)
                z_init = z0_long
            elif self.config.ode_static_data == 'ADD':
                z_init = z0_long + z0_stat
            else:
                z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
        else:
            REC_STAT = None
            z_init = z0_long

        pred_z = self.run_ODE(z_init)
        REC_LONG = self.L_Dec(pred_z).detach().cpu()
        REC_LONG = self.combine_long_preds(REC_LONG)

        total_real_long = torch.stack(total_real_long_, dim=1).detach().cpu().float()
        total_mask_long = torch.stack(total_mask_long_, dim=1).detach().cpu()
        total_sim_long = torch.stack(total_sim_long_, dim=1).detach().cpu().float()

        total_real_stat = torch.cat(total_real_stat, dim=1).detach().cpu()
        total_mask_stat = torch.cat(total_mask_stat, dim=1).detach().cpu()
        total_rec_stat = torch.cat(total_rec_stat, dim=1).detach().cpu()
        total_zinit = torch.cat(total_zinit, dim=1).detach().cpu()
        total_Z0_Long = torch.cat(total_Z0_Long, dim=1).detach().cpu()
        total_Z0_Stat = torch.cat(total_Z0_Stat, dim=1).detach().cpu()
        z_init = z_init.detach().cpu()
        REC_STAT = REC_STAT.detach().cpu()

        return (total_zinit, z_init,
                Means_Long, Std_Long, total_Z0_Long,
                Means_Stat, Std_Stat, total_Z0_Stat,
                total_real_long, total_mask_long,
                total_sim_long, REC_LONG,
                total_real_stat, total_mask_stat,
                total_rec_stat, REC_STAT)

    @torch.no_grad()
    def run(self):

        epoch = self.config.epoch_init
        desc_bar = '[Val - %d] Epoch: %d' % (self.config.Val_Scenario, epoch)

        progress_bar_val = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

        name_ = 'Val_Imgs_%s'%(self.config.val_data_type)
        save_path = os.path.join(self.config.save_path_samples,
                                 name_)
        os.makedirs(save_path, exist_ok=True)

        self.tau = 1e-3
        var_names_long, var_names_static = self.dataloader.dataset.get_var_names()

        if 'Sampling_PPD' in self.config.val_data_type:

            VI_data_path = os.path.join(
                os.path.join(self.config.save_path_samples,
                             'Val_Imgs_Sampling_PSD'),
                'Latent_Space_Ep%d.pth'%(self.config.epoch_init))

            params_data = torch.load(VI_data_path)
            z0_long = params_data['Z0_Long']
            num_z0 = z0_long.size(-1)
            z0_long = z0_long.view(-1, num_z0)
            means, stds = z0_long.mean(0), z0_long.std(0)
            self.Long_Dists = dist.normal.Normal(means, stds)

            if self.config.static_data:
                z0_stat = params_data['Z0_Stat']
                num_z0 = z0_stat.size(-1)
                z0_stat = z0_stat.view(-1, num_z0)
                means, stds = z0_stat.mean(0), z0_stat.std(0)
                self.Stat_Dists = dist.normal.Normal(means, stds)

        for iter, data in progress_bar_val:


            preds = self.get_pred_val(data)

            (total_zinit, z_init,
            Means_Long, Std_Long, total_Z0_Long,
            Means_Stat, Std_Stat, total_Z0_Stat,
            total_real_long, total_mask_long,
            total_sim_long, total_rec_long,
            total_real_stat, total_mask_stat,
            total_rec_stat, REC_STAT) = preds

            VI = {}
            VI['Means_Long'] = Means_Long
            VI['Std_Long'] = Std_Long
            VI['Z0_Long'] = total_Z0_Long

            VI['Means_Stat'] = Means_Stat
            VI['Std_Stat'] = Std_Stat
            VI['Z0_Stat'] = total_Z0_Stat
            VI['PBO'] = data[-2]
            VI['Drug_Type'] = data[-3]
            VI['PTNO'] = data[-1].cpu()

            DATA = {}
            DATA['Z_init'] = total_zinit
            DATA['REC_Z_init'] = z_init

            DATA['Obs_Long'] = total_real_long
            DATA['Mask_Long'] = total_mask_long
            DATA['SIMS_Long'] = total_sim_long
            DATA['PBO'] = data[-2]
            DATA['Drug_Type'] = data[-3]
            DATA['REC_Long'] = total_rec_long
            DATA['VarNames_Long'] = var_names_long
            DATA['VarTypes_Long'] = self.long_types
            lt_norm_list, tmax_list = self.dataloader.dataset.get_TEncs()
            #DATA['T'] = [(lt_norm * tmax).round() for lt_norm, tmax in zip(lt_norm_list, tmax_list)]
            DATA['T'] = [(lt_norm_list * tmax_list[0]).round()]
            DATA['T_DE'] = self.T.detach().cpu() * self.Tmax.detach().cpu()

            DATA['Obs_Stat'] = total_real_stat
            DATA['Mask_Stat'] = total_mask_stat
            DATA['SIMS_Stat'] = total_rec_stat
            DATA['REC_Stat'] = REC_STAT
            DATA['VarNames_Stat'] = var_names_static
            DATA['VarTypes_Stat'] = self.static_types
            DATA['PTNO'] = data[-1].cpu()
            if self.config.dataset == 'A4':
                real_long_types, real_static_types = load_only_A4_types(self.config)
            elif self.config.dataset == 'PROACT':
                real_long_types, real_static_types = load_only_PROACT_types(self.config)

            DATA['Real_VarTypes_Long'] = real_long_types
            DATA['Real_VarTypes_Stat'] = real_static_types
            
        best = 'Best_' if self.config.from_best else ''
        Val_Scenario = '' if self.config.Val_Scenario == 0 else 'Val%s_'%(self.config.Val_Scenario)
        if "Sampling_PSD" in self.config.val_data_type:
            torch.save(VI, 
                os.path.join(save_path, 'Latent_Space_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))
        torch.save(DATA, 
            os.path.join(save_path, 'Results_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))