import os
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.distributions as dist
from solver import Solver
import sys
sys.path.append('../../')
from Common_Functions.models.val_utils import (
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
        i_c, e_c = 0, 0
        for i, c_long_ids in enumerate(self.cont_long_ids):

            e_c += len(c_long_ids)
            # Unscaling predictions
            pred_x[i] = unscale(pred_x[i], c_long_ids, i_c, e_c,
                                     self.scaling_long_stats,
                                     scaler=self.scaler)
            # Transforming back the predictions
            pred_x[i] = transform_back(pred_x[i], c_long_ids, self.log_scaler)
            i_c = e_c
        return pred_x

    @torch.no_grad()
    def get_pred_val(self, data):

        val_runs = self.config.nruns_ppd

        total_real = []
        total_mask = []
        total_pbo_sim = []
        total_trt_sim = []
        total_Z0_Long = []
        total_Z0_Stat = []
        total_zinit = []

        total_IPRED = []
        total_Z0_Long_IPRED = []
        total_Z0_Stat_IPRED = []
        total_zinit_IPRED = []

        L_Data = [ld.to(self.device) for ld in data[0]]
        L_Mask = [ld.to(self.device) for ld in data[1]]
        # It needs to be the same for all patients
        L_T = [ld[0].to(self.device) for ld in data[2]]
        L_Mask_DE = data[3].to(self.device)
        if 'DRHS' in self.config.type_dynamics_lerner:
            RHS_Data = data[4].to(self.device)
        b = L_Data[0].size(0)

        _, Means_Long, Std_Long = self.Encoder.encode_long(
            L_Data, L_Mask, L_T, tau=self.tau, val=True)
        IPRED_Long_Dists = dist.normal.Normal(Means_Long, Std_Long)

        i_c, e_c = 0, 0
        for i, c_long_ids in enumerate(self.cont_long_ids):
            c_long_ids = self.cont_long_ids[i]
            e_c += len(c_long_ids)
            L_Data[i] = unscale(L_Data[i], c_long_ids, i_c, e_c,
                                     self.scaling_long_stats,
                                     scaler=self.scaler)
            L_Data[i] = transform_back(L_Data[i], c_long_ids, self.log_scaler)
            i_c = e_c

        # IPRED
        S_Data = data[5].to(self.device)
        S_Mask = data[6].to(self.device)
        _, _, Means_Stat, Std_Stat = self.Encoder.encode_stat(
            S_Data, S_Mask, self.tau, val=True)
        IPRED_Stat_Dists = dist.normal.Normal(Means_Stat, Std_Stat)

        # Static part
        prior_s_probs = self.Encoder.HIVAE.Decoder.prior_s_val
        prior_s_probs = prior_s_probs.unsqueeze(0).expand(b, -1)
        prior_s_dist = dist.OneHotCategorical(probs=prior_s_probs)

        prior_bl_size = sum(self.config.long_ldim)
        loc = torch.zeros(b, prior_bl_size, device=self.device)
        scale = torch.ones(b, prior_bl_size, device=self.device)

        Long_Dists = dist.Normal(loc=loc, scale=scale)

        for nrun in range(val_runs):
            print('Begin---- . Nrun', nrun + 1, 'of', val_runs)

            print('--------- IPRED')
            z0_stat = IPRED_Stat_Dists.sample()
            z0_long = IPRED_Long_Dists.sample()
            rhs_long = z0_long

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
            z_init = self.Encoder.project_zinit(z_init)

            if 'Stat' in self.config.type_dynamics_lerner:
                rhs_long = torch.cat((rhs_long, z0_stat), 1)

            # IPRED
            self.DL.set_data(RHS_Data, self.T, rhs_long)
            pred_z = self.run_SDE(z_init)

            IPRED, _ = self.L_Dec.dec_samplings(
                pred_z, L_Mask, L_Mask_DE)
            IPRED = self.combine_long_preds(IPRED)

            total_Z0_Stat_IPRED.append(z0_stat.unsqueeze(1))
            total_Z0_Long_IPRED.append(z0_long.unsqueeze(1))
            total_zinit_IPRED.append(z_init.unsqueeze(1))

            print('--------- SIMS')

            samples_s = prior_s_dist.sample()
            mean_pz = self.Encoder.HIVAE.Decoder.prior_loc_z(samples_s)
            std_pz = torch.ones_like(mean_pz)
            prior_z_dist = dist.normal.Normal(mean_pz, std_pz)
            z0_stat = prior_z_dist.sample()

            z0_long = Long_Dists.sample()
            rhs_long = z0_long

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
            z_init = self.Encoder.project_zinit(z_init)

            if 'Stat' in self.config.type_dynamics_lerner:
                rhs_long = torch.cat((rhs_long, z0_stat), 1)

            # TRT Simulation
            TRT_RHS_Data = torch.ones_like(RHS_Data)
            self.DL.set_data(TRT_RHS_Data, self.T, rhs_long)
            pred_z = self.run_SDE(z_init)

            sim_trt, _ = self.L_Dec.dec_samplings(
                pred_z, L_Mask, L_Mask_DE)
            sim_trt = self.combine_long_preds(sim_trt)

            # PBO Simulation
            PBO_RHS_Data = torch.ones_like(RHS_Data)
            self.DL.set_data(PBO_RHS_Data, self.T, rhs_long)
            pred_z = self.run_SDE(z_init)

            sim_pbo, _ = self.L_Dec.dec_samplings(
                pred_z, L_Mask, L_Mask_DE)
            sim_pbo = self.combine_long_preds(sim_pbo)

            total_Z0_Stat.append(z0_stat.unsqueeze(1))
            total_Z0_Long.append(z0_long.unsqueeze(1))
            total_zinit.append(z_init.unsqueeze(1))

            if self.config.dataset == 'A4_Causal':
                idx = [0, -1]
                # PACC at week 0 and 240
                total_real.append(L_Data[0][:, idx, :1].unsqueeze(1))
                total_mask.append(L_Mask[0][:, idx, :1].unsqueeze(1))
                total_IPRED.append(IPRED[0][:, idx, :1].unsqueeze(1))
                total_trt_sim.append(sim_trt[0][:, idx, :1].unsqueeze(1))
                total_pbo_sim.append(sim_pbo[0][:, idx, :1].unsqueeze(1))

        total_Z0_Stat = torch.cat(total_Z0_Stat, dim=1).detach().cpu()
        total_Z0_Long = torch.cat(total_Z0_Long, dim=1).detach().cpu()
        total_zinit = torch.cat(total_zinit, dim=1).detach().cpu()

        total_real = torch.cat(total_real, dim=1).detach().cpu()
        total_mask = torch.cat(total_mask, dim=1).detach().cpu()
        total_IPRED = torch.cat(total_IPRED, dim=1).detach().cpu()
        total_trt_sim = torch.cat(total_trt_sim, dim=1).detach().cpu()
        total_pbo_sim = torch.cat(total_pbo_sim, dim=1).detach().cpu()

        total_Z0_Stat_IPRED = torch.cat(total_Z0_Stat_IPRED, dim=1).detach().cpu()
        total_Z0_Long_IPRED = torch.cat(total_Z0_Long_IPRED, dim=1).detach().cpu()
        total_zinit_IPRED = torch.cat(total_zinit_IPRED, dim=1).detach().cpu()

        return(total_real, total_mask, total_IPRED,
               total_trt_sim, total_pbo_sim,
               total_Z0_Stat_IPRED, total_Z0_Long_IPRED,
               total_zinit_IPRED, total_Z0_Stat,
               total_Z0_Long, total_zinit)


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
        for iter, data in progress_bar_val:

            preds = self.get_pred_val(data)
            (total_real, total_mask, total_IPRED,
               total_trt_sim, total_pbo_sim,
               total_Z0_Stat_IPRED, total_Z0_Long_IPRED,
               total_zinit_IPRED, total_Z0_Stat,
               total_Z0_Long, total_zinit) = preds

            VI = {}
            VI['Z0_Long'] = total_Z0_Long
            VI['Z0_Stat'] = total_Z0_Stat
            VI['Z0_Long_IPRED'] = total_Z0_Long_IPRED
            VI['Z0_Stat_IPRED'] = total_Z0_Stat_IPRED
            VI['PBO'] = data[-2]
            VI['PTNO'] = data[-1].cpu()

            DATA = {}
            DATA['Z_init_IPRED'] = total_zinit_IPRED
            DATA['Z_init'] = total_zinit
            DATA['OBS'] = total_real
            DATA['MASK'] = total_mask
            DATA['IPRED'] = total_IPRED
            DATA['TRT_SIMS'] = total_trt_sim
            DATA['PBO_SIMS'] = total_pbo_sim
            DATA['PTNO'] = data[-1].cpu()
            DATA['PBO'] = data[-2]
            DATA['Obs_Scores'] = data[-3]
            DATA['PS_Scores'] = data[-4]

        best = 'Best_' if self.config.from_best else ''
        Val_Scenario = '' if self.config.Val_Scenario == 0 else 'Val%s_'%(self.config.Val_Scenario)
        torch.save(VI, 
                os.path.join(save_path, 'Latent_Space_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))
        torch.save(DATA, 
            os.path.join(save_path, 'Results_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))