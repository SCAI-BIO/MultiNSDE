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


def resolve_ate_target(dataloader, requested_name=''):
    var_names_long = dataloader.dataset.var_names_long
    requested_name = str(requested_name or '').strip()

    if requested_name == '':
        return 0, 0, str(var_names_long[0][0])

    requested_name_lower = requested_name.lower()
    for enc_idx, names in enumerate(var_names_long):
        for feat_idx, feat_name in enumerate(names):
            if str(feat_name).strip().lower() == requested_name_lower:
                return enc_idx, feat_idx, str(feat_name)

    available = [str(name) for names in var_names_long for name in names]


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
        risk_arm_mean = None
        factual_risk_mean = None

        def flatten_long_preds(long_preds):
            if isinstance(long_preds, (list, tuple)):
                if len(long_preds) == 1:
                    return long_preds[0]
                return torch.cat(long_preds, dim=-1)
            return long_preds

        total_real = []
        total_mask = []
        total_pbo_sim = []
        total_trt_sim = []
        total_trt_sim_arm = {1: [], 2: [], 3: []}
        total_risk_sim_arm = {0: [], 1: [], 2: [], 3: []}
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
        if self.config.time_to_event:
            Time_TE = data[5].to(self.device)
            Event_TE = data[6].to(self.device)
        else:
            Time_TE = Event_TE = None
        b = L_Data[0].size(0)
        n_drug = getattr(self.config, 'n_drug_var', 1)
        treat_start = RHS_Data.size(-1) - n_drug

        def build_arm_rhs(rhs_source, arm_idx):
            arm_rhs = rhs_source.clone()
            if self.config.dataset == 'DATATOP_Causal' and n_drug > 1:
                arm_rhs_templates = self.dataloader.dataset.arm_rhs_templates.to(self.device)
                if treat_start > 0:
                    arm_template = arm_rhs_templates[arm_idx].unsqueeze(0).expand(b, -1, -1)
                    arm_rhs[:, :, :treat_start] = arm_template
                arm_rhs[:, :, treat_start:] = 0.0
                arm_rhs[:, :, treat_start + arm_idx] = 1.0
            else:
                arm_rhs[:, :, treat_start:] = 0.0
                if arm_idx != 0:
                    arm_rhs[:, :, treat_start:] = 1.0
            return arm_rhs

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
        if self.config.dataset == 'DATATOP_Causal':
            S_Data = data[7].to(self.device)
            S_Mask = data[8].to(self.device)
        else:
            S_Data = data[5].to(self.device)
            S_Mask = data[6].to(self.device)
        _, _, Means_Stat, Std_Stat = self.Encoder.encode_stat(
            S_Data, S_Mask, self.tau, val=True)
        IPRED_Stat_Dists = dist.normal.Normal(Means_Stat, Std_Stat)

        if self.config.dataset == 'DATATOP_Causal':
            risk_arm_mean = {}
            factual_risk_mean = None
            if self.config.time_to_event:
                z_init_mean = torch.cat((Means_Long, Means_Stat), dim=1)
                z_init_mean = self.Encoder.project_zinit(z_init_mean)
                arm_ids = [0, 1, 2, 3]
                risk_stack = []
                for arm in arm_ids:
                    arm_rhs = build_arm_rhs(RHS_Data, arm)
                    arm_risk = self.Encoder.get_risk(z_init_mean, arm_rhs[:, 0])
                    risk_arm_mean[arm] = arm_risk.detach().cpu()
                    risk_stack.append(arm_risk.unsqueeze(1))

                if self.config.dataset == 'DATATOP_Causal' and n_drug > 1:
                    factual_treat = torch.argmax(
                        RHS_Data[:, 0, treat_start:treat_start + n_drug], dim=-1
                    ).long()
                    risk_stack = torch.cat(risk_stack, dim=1)
                    factual_risk_mean = torch.gather(
                        risk_stack,
                        1,
                        factual_treat.view(b, 1, 1).expand(-1, 1, risk_stack.size(-1))
                    ).squeeze(1)
                else:
                    factual_risk_mean = risk_stack[:, 1].squeeze(1)

                factual_risk_mean = factual_risk_mean.detach().cpu()

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

            if self.config.dataset == 'DATATOP_Causal':
                PBO_RHS_Data = build_arm_rhs(RHS_Data, 0)
                self.DL.set_data(PBO_RHS_Data, self.T, rhs_long)
                pred_z = self.run_SDE(z_init)
                sim_pbo, _ = self.L_Dec.dec_samplings(pred_z, L_Mask, L_Mask_DE)
                sim_pbo = self.combine_long_preds(sim_pbo)
                total_pbo_sim.append(flatten_long_preds(sim_pbo).unsqueeze(1))
                sim_risk_by_arm = {0: self.Encoder.get_risk(z_init, PBO_RHS_Data[:, 0])}

                # Multi-arm DATATOP: produce one simulation per active treatment arm.
                sim_trt_by_arm = {}
                for arm in [1, 2, 3]:
                    ARM_RHS_Data = build_arm_rhs(RHS_Data, arm)

                    self.DL.set_data(ARM_RHS_Data, self.T, rhs_long)
                    pred_z = self.run_SDE(z_init)
                    sim_arm, _ = self.L_Dec.dec_samplings(pred_z, L_Mask, L_Mask_DE)
                    sim_trt_by_arm[arm] = self.combine_long_preds(sim_arm)
                    total_trt_sim_arm[arm].append(flatten_long_preds(sim_trt_by_arm[arm]).unsqueeze(1))
                    sim_risk_by_arm[arm] = self.Encoder.get_risk(z_init, ARM_RHS_Data[:, 0])

                # Keep a compatibility channel for older tooling that expects TRT_SIMS.
                sim_trt = sim_trt_by_arm[1]
            else:
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

            if self.config.dataset == 'DATATOP_Causal' and self.config.time_to_event:
                for arm, arm_risk in sim_risk_by_arm.items():
                    total_risk_sim_arm[arm].append(arm_risk.unsqueeze(1))

        total_Z0_Stat = torch.cat(total_Z0_Stat, dim=1).detach().cpu()
        total_Z0_Long = torch.cat(total_Z0_Long, dim=1).detach().cpu()
        total_zinit = torch.cat(total_zinit, dim=1).detach().cpu()

        if self.config.dataset == 'A4_Causal':
            total_real = torch.cat(total_real, dim=1).detach().cpu()
            total_mask = torch.cat(total_mask, dim=1).detach().cpu()
            total_IPRED = torch.cat(total_IPRED, dim=1).detach().cpu()
            total_trt_sim = torch.cat(total_trt_sim, dim=1).detach().cpu()
            total_pbo_sim = torch.cat(total_pbo_sim, dim=1).detach().cpu()
        elif self.config.dataset == 'DATATOP_Causal' and len(total_pbo_sim) > 0:
            total_real = None
            total_mask = None
            total_IPRED = None
            total_trt_sim = None
            total_pbo_sim = torch.cat(total_pbo_sim, dim=1).detach().cpu()
        else:
            total_real = None
            total_mask = None
            total_IPRED = None
            total_trt_sim = None
            total_pbo_sim = None

        total_trt_sim_arms = None
        if self.config.dataset == 'DATATOP_Causal' and len(total_trt_sim_arm[1]) > 0:
            total_trt_sim_arms = {
                arm: torch.cat(total_trt_sim_arm[arm], dim=1).detach().cpu()
                for arm in [1, 2, 3]
            }

        total_risk_sim_arms = None
        if self.config.time_to_event and len(total_risk_sim_arm[0]) > 0:
            total_risk_sim_arms = {
                arm: torch.cat(total_risk_sim_arm[arm], dim=1).detach().cpu()
                for arm in total_risk_sim_arm
                if len(total_risk_sim_arm[arm]) > 0
            }

        total_Z0_Stat_IPRED = torch.cat(total_Z0_Stat_IPRED, dim=1).detach().cpu()
        total_Z0_Long_IPRED = torch.cat(total_Z0_Long_IPRED, dim=1).detach().cpu()
        total_zinit_IPRED = torch.cat(total_zinit_IPRED, dim=1).detach().cpu()

        return(total_real, total_mask, total_IPRED,
               total_trt_sim, total_pbo_sim,
               total_Z0_Stat_IPRED, total_Z0_Long_IPRED,
               total_zinit_IPRED, total_Z0_Stat,
               total_Z0_Long, total_zinit, total_trt_sim_arms,
               factual_risk_mean, risk_arm_mean, total_risk_sim_arms)


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

        def cat_batches(tensors):
            return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

        vi_batches = {
            'Z0_Long': [],
            'Z0_Stat': [],
            'Z0_Long_IPRED': [],
            'Z0_Stat_IPRED': [],
            'PBO': [],
            'PTNO': [],
        }
        data_batches = {
            'Z_init_IPRED': [],
            'Z_init': [],
            'PTNO': [],
            'PBO': [],
            'Obs_Scores': [],
            'PS_Scores': [],
        }
        if self.config.dataset == 'A4_Causal':
            data_batches.update({
                'OBS': [],
                'MASK': [],
                'IPRED': [],
                'TRT_SIMS': [],
                'PBO_SIMS': [],
            })
        if self.config.time_to_event:
            data_batches.update({
                'Time_TE': [],
                'Event_TE': [],
            })
            risk_batches = []
            risk_arm_batches = {0: [], 1: [], 2: [], 3: []}
            sim_risk_arm_batches = {0: [], 1: [], 2: [], 3: []}
        else:
            risk_batches = None
            risk_arm_batches = None
            sim_risk_arm_batches = None
        if self.config.dataset == 'DATATOP_Causal':
            trt_sim_arm_batches = {1: [], 2: [], 3: []}
            treat_batches = []
            pbo_sim_batches = []
        else:
            trt_sim_arm_batches = None
            treat_batches = None
            pbo_sim_batches = None

        self.tau = 1e-3
        for iter, data in progress_bar_val:

            preds = self.get_pred_val(data)
            (total_real, total_mask, total_IPRED,
               total_trt_sim, total_pbo_sim,
               total_Z0_Stat_IPRED, total_Z0_Long_IPRED,
               total_zinit_IPRED, total_Z0_Stat,
                    total_Z0_Long, total_zinit, total_trt_sim_arms,
                    factual_risk_mean, risk_arm_mean, total_risk_sim_arms) = preds

            vi_batches['Z0_Long'].append(total_Z0_Long)
            vi_batches['Z0_Stat'].append(total_Z0_Stat)
            vi_batches['Z0_Long_IPRED'].append(total_Z0_Long_IPRED)
            vi_batches['Z0_Stat_IPRED'].append(total_Z0_Stat_IPRED)
            vi_batches['PBO'].append(data[-2].cpu())
            vi_batches['PTNO'].append(data[-1].cpu())

            data_batches['Z_init_IPRED'].append(total_zinit_IPRED)
            data_batches['Z_init'].append(total_zinit)
            data_batches['PTNO'].append(data[-1].cpu())
            data_batches['PBO'].append(data[-2].cpu())
            data_batches['Obs_Scores'].append(data[-3].cpu())
            data_batches['PS_Scores'].append(data[-4].cpu())
            if self.config.dataset == 'A4_Causal':
                data_batches['OBS'].append(total_real)
                data_batches['MASK'].append(total_mask)
                data_batches['IPRED'].append(total_IPRED)
                data_batches['TRT_SIMS'].append(total_trt_sim)
                data_batches['PBO_SIMS'].append(total_pbo_sim)
            if self.config.time_to_event:
                data_batches['Time_TE'].append(data[5].cpu())
                data_batches['Event_TE'].append(data[6].cpu())
                if factual_risk_mean is not None:
                    risk_batches.append(factual_risk_mean)
                if risk_arm_mean is not None:
                    for arm, risk in risk_arm_mean.items():
                        risk_arm_batches[arm].append(risk)
                if total_risk_sim_arms is not None:
                    for arm, risk in total_risk_sim_arms.items():
                        sim_risk_arm_batches[arm].append(risk)
            if self.config.dataset == 'DATATOP_Causal':
                if total_pbo_sim is not None:
                    pbo_sim_batches.append(total_pbo_sim)
                if total_trt_sim_arms is not None:
                    for arm, sims in total_trt_sim_arms.items():
                        trt_sim_arm_batches[arm].append(sims)
                n_drug = getattr(self.config, 'n_drug_var', 4)
                treat_start = data[4].shape[-1] - n_drug
                treat_obs = torch.argmax(data[4][:, 0, treat_start:treat_start + n_drug], dim=-1).long()
                treat_batches.append(treat_obs.cpu())

        VI = {
            key: cat_batches(values)
            for key, values in vi_batches.items()
        }

        DATA = {
            key: cat_batches(values)
            for key, values in data_batches.items()
        }
        if self.config.time_to_event:
            DATA['VarNames_TE'] = list(self.dataloader.dataset.time_event_names)
            if risk_batches:
                DATA['Risk'] = cat_batches(risk_batches)
            for arm, values in risk_arm_batches.items():
                if values:
                    DATA[f'Risk_ARM{arm}'] = cat_batches(values)
            for arm, values in sim_risk_arm_batches.items():
                if values:
                    DATA[f'SIMS_Risk_ARM{arm}'] = cat_batches(values)
        if self.config.dataset == 'DATATOP_Causal':
            if pbo_sim_batches:
                DATA['PBO_SIMS'] = cat_batches(pbo_sim_batches)
            if trt_sim_arm_batches is not None:
                for arm, values in trt_sim_arm_batches.items():
                    if values:
                        DATA[f'TRT_SIMS_ARM{arm}'] = cat_batches(values)
                if trt_sim_arm_batches[1]:
                    DATA['TRT_SIMS'] = DATA['TRT_SIMS_ARM1']
            if treat_batches:
                DATA['TREAT'] = cat_batches(treat_batches)

        best = 'Best_' if self.config.from_best else ''
        Val_Scenario = '' if self.config.Val_Scenario == 0 else 'Val%s_'%(self.config.Val_Scenario)
        torch.save(VI, 
                os.path.join(save_path, 'Latent_Space_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))
        torch.save(DATA, 
            os.path.join(save_path, 'Results_%s%sEp%d.pth'%(best,Val_Scenario,epoch)))