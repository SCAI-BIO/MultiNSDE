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
from Prognosis.data.load_A4 import load_only_A4_types
from Prognosis.data.load_PROACT import load_only_PROACT_types
from Prognosis.data.load_DATATOP import load_only_DATATOP_types
warnings.filterwarnings('ignore')


class Validation(Solver):
    def __init__(self, config, dataloader):
        super(Validation, self).__init__(config, dataloader)
        self.scaling_long_stats = self.dataloader.dataset.scaling_long_stats
        self.scaler = self.config.scaler
        self.log_scaler = self.config.log_scaler 
        self.run()

    @torch.no_grad()
    def combine_long_preds(self, pred_x, logits, dist=False):
        i_c, e_c = 0, 0
        for i, (c_long_ids, ordreg_ids) in enumerate(zip(self.cont_long_ids, self.ordreg_ids)):

            e_c += len(c_long_ids)
            # Unscaling predictions
            pred_x[i] = unscale(pred_x[i], c_long_ids, i_c, e_c,
                                     self.scaling_long_stats,
                                     scaler=self.scaler)
            # Transforming back the predictions
            pred_x[i] = transform_back(pred_x[i], c_long_ids, self.log_scaler)
            i_c = e_c
            if not dist and len(ordreg_ids) > 0:

                logits_i = logits[i]
                long_cat = []
                for logits_j in logits_i:
                    _, long_cat_j = logits_j.max(-1, keepdim=True)
                    long_cat.append(long_cat_j)
                long_cat = torch.cat(long_cat, -1)
                # Assigning the categorical predictions
                pred_x[i][..., ordreg_ids] = long_cat.to(torch.float)
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
        total_RHS_Long = list()
        total_sim_probs_ = list()
        total_risk = list()
        total_zinit = list()

        L_Data = [ld.to(self.device) for ld in data[0]]
        L_Mask = [ld.to(self.device) for ld in data[1]]
        # It needs to be the same for all patients
        L_T = [ld[0].to(self.device) for ld in data[2]]
        L_Mask_DE = data[3].to(self.device)
        if 'DRHS' in self.config.type_dynamics_lerner:
            RHS_Data = data[4].to(self.device)
        b = L_Data[0].size(0)

        S_Data = data[7].to(self.device)
        S_Mask = data[8].to(self.device)
        s_samples, z0_stat, Means_Stat, Std_Stat = self.Encoder.encode_stat(
            S_Data, S_Mask, self.tau, val=True)

        self.Stat_Dists = dist.normal.Normal(Means_Stat, Std_Stat)
        aux = self.Encoder.encode_long(
            L_Data, L_Mask, L_T, tau=self.tau, val=True)
        z0_long, Means_Long, Std_Long = aux[0]
        _, Means_RHS_Long, Std_RHS_Long = aux[1]

        i_c, e_c = 0, 0
        for i, c_long_ids in enumerate(self.cont_long_ids):
            c_long_ids = self.cont_long_ids[i]
            e_c += len(c_long_ids)
            L_Data[i] = unscale(L_Data[i], c_long_ids, i_c, e_c,
                                     self.scaling_long_stats,
                                     scaler=self.scaler)
            L_Data[i] = transform_back(L_Data[i], c_long_ids, self.log_scaler)
            i_c = e_c

        self.Long_Dists = dist.normal.Normal(Means_Long, Std_Long)
        self.RHS_Long_Dists = dist.normal.Normal(Means_RHS_Long, Std_RHS_Long)

        for nrun in range(val_runs):
            print('Begin---- VI and NSDE sampling. Nrun', nrun + 1, 'of', val_runs)

            z0_long = self.Long_Dists.sample() if self.config.type_hivae not in ['IC_HIVAE', 'SLR_HIVAE'] else Means_Long
            rhs_long = self.RHS_Long_Dists.sample() if 'LongIC' not in self.config.type_hivae else Means_RHS_Long

            total_Z0_Long.append(z0_long.unsqueeze(1))
            total_RHS_Long.append(rhs_long.unsqueeze(1))

            z0_stat = self.Stat_Dists.sample()

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
            if self.config.type_hivae != 'SLR_HIVAE':
                z_init = self.Encoder.project_zinit(z_init)
            total_Z0_Stat.append(z0_stat.unsqueeze(1))

            if 'Stat' in self.config.type_dynamics_lerner:
                rhs_long = torch.cat((rhs_long, z0_stat), 1)

            rec_samples = self.Encoder.decode_stat(
                S_Data, S_Mask, z0_stat, self.tau)

            total_real_stat.append(S_Data.unsqueeze(1))
            total_mask_stat.append(S_Mask.unsqueeze(1))
            total_rec_stat.append(rec_samples.unsqueeze(1))

            
            if self.config.time_to_event:
                risk = self.Encoder.get_risk(z_init, RHS_Data[:, 0])
                total_risk.append(risk.unsqueeze(1))

            self.DL.set_data(RHS_Data, self.T, rhs_long)
            pred_z = self.run_SDE(z_init)

            # probs here is an empty list if not self monotonic heads
            rec_x, probs = self.L_Dec.dec_samplings(
                pred_z, L_Mask, L_Mask_DE)
            rec_x = self.combine_long_preds(rec_x, probs, dist=True)

            total_real_long_.append(L_Data)
            total_mask_long_.append(L_Mask)
            total_sim_long_.append(rec_x)
            total_sim_probs_.append(probs)
            total_zinit.append(z_init.unsqueeze(1))

        # The metrics as well as the GOF plots need to be done with the
        # patient specific mean
        z0_long = Means_Long
        rhs_long = Means_RHS_Long
        z0_stat = torch.cat(total_Z0_Stat, dim=1).mean(1) if 'PPD' in self.config.val_data_type else Means_Stat

        z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
        if self.config.type_hivae != 'SLR_HIVAE':
            z_init = self.Encoder.project_zinit(z_init)

        if 'Stat' in self.config.type_dynamics_lerner:
            rhs_long = torch.cat((rhs_long, z0_stat), 1)

        REC_STAT = self.Encoder.decode_stat(
            S_Data, S_Mask, z0_stat, self.tau).detach().cpu()

        if self.config.time_to_event:
            risk = self.Encoder.get_risk(z_init, RHS_Data[:, 0]).detach().cpu()
        else:
            risk = torch.empty(0)

        total_REC_LONG = list()
        total_REC_PROBS = list()
        for nrun in range(val_runs):
            print('Begin---- REC with NSDE sampling. Nrun', nrun + 1, 'of', val_runs)
            if nrun==0:
                # We can define it only once cause we are using Means_RHS_Long
                self.DL.set_data(RHS_Data, self.T, rhs_long)
            pred_z = self.run_SDE(z_init)

            # probs here is an empty list
            REC_LONG, REC_LONG_probs = self.L_Dec.dec_samplings(pred_z, L_Mask, L_Mask_DE)
            REC_LONG = self.combine_long_preds(REC_LONG, REC_LONG_probs, dist=True)

            total_REC_LONG.append(REC_LONG)
            total_REC_PROBS.append(REC_LONG_probs)

        N, K = val_runs, len(L_Data)
        total_real_long = []
        total_mask_long = []
        total_sim_long = []
        total_sim_probs = []
        total_rec_long = []
        total_rec_probs = []

        # In this way because each output can have different time steps
        for k in range(K):
            out_real = [total_real_long_[n][k] for n in range(N)]
            out_real = torch.stack(out_real, dim=1).detach().cpu()
            total_real_long.append(out_real)

            out_mask = [total_mask_long_[n][k] for n in range(N)]
            out_mask = torch.stack(out_mask, dim=1).detach().cpu()
            total_mask_long.append(out_mask)

            out_pred = [total_sim_long_[n][k] for n in range(N)]
            out_pred = torch.stack(out_pred, dim=1).detach().cpu()
            total_sim_long.append(out_pred)

            out_probs = [total_sim_probs_[n][k] for n in range(N)]
            out_probs = torch.stack(out_probs, dim=1).detach().cpu()
            total_sim_probs.append(out_probs)

            out_pred = [total_REC_LONG[n][k] for n in range(N)]
            out_pred = torch.stack(out_pred, dim=1).detach().cpu()
            total_rec_long.append(out_pred)

            out_pred = [total_REC_PROBS[n][k] for n in range(N)]
            out_pred = torch.stack(out_pred, dim=1).detach().cpu()
            total_rec_probs.append(out_pred)

        total_real_stat = torch.cat(total_real_stat, dim=1).detach().cpu()
        total_mask_stat = torch.cat(total_mask_stat, dim=1).detach().cpu()
        total_rec_stat = torch.cat(total_rec_stat, dim=1).detach().cpu()
        total_zinit = torch.cat(total_zinit, dim=1).detach().cpu()
        total_RHS_Long = torch.cat(total_RHS_Long, dim=1).detach().cpu()
        total_Z0_Long = torch.cat(total_Z0_Long, dim=1).detach().cpu()
        total_Z0_Stat = torch.cat(total_Z0_Stat, dim=1).detach().cpu()
        z_init = z_init.detach().cpu()
        REC_STAT = REC_STAT.detach().cpu()
        s_samples = s_samples.detach().cpu()
        if self.config.time_to_event:
            total_risk = torch.cat(total_risk, dim=1).detach().cpu()

        return (total_zinit, z_init,
                Means_RHS_Long, Std_RHS_Long, total_RHS_Long,
                Means_Long, Std_Long, total_Z0_Long,
                s_samples, Means_Stat, Std_Stat, total_Z0_Stat,
                total_real_long, total_mask_long,
                total_sim_long, total_rec_long,
                total_risk, risk,
                total_real_stat, total_mask_stat,
                total_rec_stat, REC_STAT,
                total_sim_probs, total_rec_probs)

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
        var_names_long, var_names_static, var_names_TE = self.dataloader.dataset.get_var_names()

        if 'Sampling_PPD' in self.config.val_data_type:

            best = 'Best_' if self.config.from_best else ''
            self.IC_data_path = os.path.join(
                os.path.join(self.config.save_path_samples, 
                             'Val_Imgs_Sampling_PPD'),
                'PPD_Sampling_%sEp%d.pth'%(best, self.config.epoch_init))

            if os.path.isfile(self.IC_data_path):
                ppd_data = torch.load(self.IC_data_path, weights_only=False)
                self.z0_long = ppd_data['Z0_Long']
                self.z0_stat = ppd_data['Z0_Stat']
                if self.config.type_hivae == 'IC_BL_HIVAE':
                    self.rhs_long = ppd_data['RHS_Long']
            else:
                VI_data_path = os.path.join(
                    os.path.join(self.config.save_path_samples,
                                'Val_Imgs_Sampling_PSD'),
                    'Latent_Space_%s%sEp%d.pth'%(best, '', self.config.epoch_init))
                params_data = torch.load(VI_data_path, weights_only=False)
                z0_long = params_data['Z0_Long']
                z0_long = z0_long.view(-1, z0_long.size(-1)) # (B x REPI) x Features
                means, stds = z0_long.mean(0), z0_long.std(0)
                self.Long_Dists = dist.normal.Normal(means, stds)

                if self.config.type_hivae == 'IC_BL_HIVAE':
                    rhs_long = params_data['RHS_Long']
                    rhs_long = rhs_long.view(-1, rhs_long.size(-1)) # (B x REPI) x Features
                    means, stds = rhs_long.mean(0), rhs_long.std(0)
                    self.RHS_Long_Dists = dist.normal.Normal(means, stds)
                

                z0_stat = params_data['Z0_Stat']
                s_samples_static = params_data['GMM_Component']
                K = s_samples_static.shape[1] 
                self.Stat_Dists = [None] * K

                s_idx = s_samples_static.argmax(dim=-1) # B
                z_by_component = [[] for _ in range(K)]

                for comp in range(K):
                    mask = (s_idx == comp) # B x 1
                    z_by_component[comp] = z0_stat[mask] # b x Features

                for i in range(K):
                    z_flat = z_by_component[i].view(-1, z_by_component[i].size(-1)) # (B x REPI) x Features
                    means, stds = z_flat.mean(0), z_flat.std(0)
                    self.Stat_Dists[i] = (dist.normal.Normal(means, stds))

        for iter, data in progress_bar_val:

            preds = self.get_pred_val(data)

            (total_zinit, z_init,
            Means_RHS_Long, Std_RHS_Long, total_RHS_Long,
            Means_Long, Std_Long, total_Z0_Long,
            s_samples, Means_Stat, Std_Stat, total_Z0_Stat,
            total_real_long, total_mask_long,
            total_sim_long, total_rec_long,
            total_risk, risk,
            total_real_stat, total_mask_stat,
            total_rec_stat, REC_STAT,
            total_sim_probs, total_rec_probs) = preds

            VI = {}
            VI['Means_RHS_Long'] = Means_RHS_Long
            VI['Std_RHS_Long'] = Std_RHS_Long
            VI['RHS_Long'] = total_RHS_Long

            VI['Means_Long'] = Means_Long
            VI['Std_Long'] = Std_Long
            VI['Z0_Long'] = total_Z0_Long

            VI['Means_Stat'] = Means_Stat
            VI['Std_Stat'] = Std_Stat
            VI['GMM_Component'] = s_samples
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
            DATA['SIMS_Prob_Long'] = total_sim_probs
            DATA['REC_Long'] = total_rec_long
            DATA['REC_Prob'] = total_rec_probs
            DATA['VarNames_Long'] = var_names_long
            DATA['VarTypes_Long'] = self.long_types
            lt_norm_list, tmax_list = self.dataloader.dataset.get_TEncs()
            DATA['T'] = [(lt_norm * tmax).round() for lt_norm, tmax in zip(lt_norm_list, tmax_list)]
            DATA['T_DE'] = self.T.detach().cpu() * self.Tmax.detach().cpu()

            DATA['Obs_Stat'] = total_real_stat
            DATA['Mask_Stat'] = total_mask_stat
            DATA['SIMS_Stat'] = total_rec_stat
            DATA['REC_Stat'] = REC_STAT
            DATA['VarNames_Stat'] = var_names_static
            # In this way for the case when one uses one o the following ['IC_HIVAE', 'SLR_HIVAE']
            DATA['VarTypes_Stat'] = self.static_types[:self.config.s_vals_dim]
            DATA['PTNO'] = data[-1].cpu()

            if self.config.dataset == 'A4':
                real_long_types, real_static_types = load_only_A4_types(self.config)
                DATA['Real_VarTypes_Long'] = real_long_types
                DATA['Real_VarTypes_Stat'] = real_static_types
            elif self.config.dataset == 'PROACT':
                real_long_types, real_static_types = load_only_PROACT_types(self.config)
                DATA['Real_VarTypes_Long'] = real_long_types
                DATA['Real_VarTypes_Stat'] = real_static_types
            elif self.config.dataset == 'DATATOP':
                real_long_types, real_static_types = load_only_DATATOP_types(self.config)
                DATA['Real_VarTypes_Long'] = real_long_types
                DATA['Real_VarTypes_Stat'] = real_static_types

            if self.config.time_to_event:
                DATA['VarNames_TE'] = var_names_TE
                DATA['Risk'] = risk
                DATA['SIMS_Risk'] = total_risk
                DATA['Time_TE'] = data[5].cpu()
                DATA['Event_TE'] = data[6].cpu()
                
        best = 'Best_' if self.config.from_best else ''
        Val_Scenario = '' if self.config.Val_Scenario == 0 else 'Val%s_'%(self.config.Val_Scenario)
        if "Sampling_PSD" in self.config.val_data_type:
            torch.save(VI, 
                os.path.join(save_path, 'Latent_Space_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))
        elif "Sampling_PPD" in self.config.val_data_type and not os.path.isfile(self.IC_data_path):
            PPD = {}
            PPD['Z0_Stat'] = total_Z0_Stat
            PPD['Z0_Long'] = total_Z0_Long
            PPD['RHS_Long'] = total_RHS_Long
            torch.save(PPD, 
                os.path.join(save_path, 'PPD_Sampling_%sEp%d.pth'%(best,epoch)))

        torch.save(DATA, 
            os.path.join(save_path, 'Results_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))