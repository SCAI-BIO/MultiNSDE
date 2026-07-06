import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from collections import OrderedDict
import sys
sys.path.append('../../')
from Common_Functions.models.utils import print_current_losses
from Common_Functions.models.utils_pbo import split_by_pbo
from ATE_helpers import get_batch_rmst_or_components
from solver import Solver
from validation import *
import warnings
warnings.filterwarnings('ignore')

class Train(Solver):
    def __init__(self, config, dataloader):
        super(Train, self).__init__(config, dataloader)

        self.run()

    def run(self):

        global_steps = 0
        total_time = time.time()
        no_improvement = 0
        if self.config.inv_ic_loss:
            ic_criterion = SamplesLoss("sinkhorn", p=2, blur=0.05)

        # Epoch_init begins in 1
        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):

            avg_loss = 0
            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

            tau = np.max([1.0 - (0.999/(self.config.num_epochs - 50)) *
                         (epoch), 1e-3])

            epoch_time_init = time.time()
            if not self.config.time_to_event:   
                Time_TE = Event_TE = None # To avoid modifying the encoder
            factor_OR_DO = 0 if self.config.Only_ObsEndpoints else 1

            # Training along dataset            
            for iter, data in progress_bar:

                global_steps += 1
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
                    aux_idx = 2
                else:
                    Time_TE = Event_TE = None
                    aux_idx = 0

                S_Data = data[5 + aux_idx].to(self.device)
                S_Mask = data[6 + aux_idx].to(self.device)
                Pi_Scores = data[7 + aux_idx].to(self.device)
                Obs_Scores = data[8 + aux_idx].to(self.device)
                Placebo_Flag = data[-2].to(self.device)

                z_init, KL_Long, RHS_Long, te_log_loss, log_prob_stat, KL_Sstatic, KL_Zstatic = self.Encoder(
                    L_Data, L_Mask, L_T, Time_TE, Event_TE, RHS_Data[:, 0], S_Data, S_Mask, tau)

                self.DL.set_data(RHS_Data, self.T, RHS_Long)
                pred_z = self.run_SDE(z_init)

                pred_x, log_prob = self.L_Dec(
                    L_Data, L_Mask, L_Mask_DE, pred_z)
                full_rec_loss = - log_prob.mean() * self.config.lambda_RecLong
                KL_Long = self.config.lambda_KLLong * KL_Long.mean()
                ELBO_NODE = KL_Long + full_rec_loss
                loss = ELBO_NODE

                if self.config.dataset == 'A4_Causal':
                    # PACC at week 240
                    y_a = L_Data[0][:, -1, 0]
                    u_a = pred_x[:, -1, 0]
                    delta = L_Mask[0][:, -1, 0]

                elif self.config.dataset == 'DATATOP_Causal':
                    if self.config.time_to_event:
                        factual_risk = self.Encoder.get_risk(z_init, RHS_Data[:, 0])
                        y_a, u_a, delta = get_batch_rmst_or_components(
                            factual_risk,
                            Time_TE,
                            Event_TE,
                            horizon=getattr(self.config, 'survival_target_time', None),
                        )

                if self.config.or_losses:
                    if self.config.dataset == 'A4_Causal':
                        r_i = delta / Obs_Scores * (y_a - u_a)
                        loss_treatment = (r_i * (((1-Placebo_Flag)/Pi_Scores) - (Placebo_Flag/(1 - Pi_Scores)))).pow(2).mean()
                        loss_treatment = self.config.lambda_OR_TRT * loss_treatment
                        loss_dropout   = (r_i * (delta - Obs_Scores)).pow(2).mean()
                        loss_dropout = factor_OR_DO * self.config.lambda_OR_DO * loss_dropout
                        loss = loss + loss_treatment + loss_dropout

                    if self.config.dataset == 'DATATOP_Causal':
                        r_i = delta / Obs_Scores * (y_a - u_a)  # DATATOP OR residual is theobserved RMST target minus predicted RMST, reweighted for observation.
                        # Multi-arm setting: Pi_Scores already matches each patient's realized arm.
                        treatment_term = r_i / Pi_Scores  # Reweight the DATATOP residual by the realized treatment propensity
                        loss_treatment = treatment_term.pow(2).mean()
                        loss_treatment = self.config.lambda_OR_TRT * loss_treatment
                        loss_dropout   = (r_i * (delta - Obs_Scores)).pow(2).mean()
                        loss_dropout = factor_OR_DO * self.config.lambda_OR_DO * loss_dropout
                        loss = loss + loss_treatment + loss_dropout

                if self.config.time_to_event:
                    # te_log_loss has a dimension per TE variable in case one would need it in the future
                    te_log_loss = - te_log_loss.sum() * self.config.lambda_TE
                    loss = loss + te_log_loss

                if self.config.inv_ic_loss:
                    z_init_pbo, z_init_non_pbo, _, _ = split_by_pbo(z_init, Placebo_Flag,static=True)
                    ic_loss = self.config.lambda_IC * ic_criterion(z_init_non_pbo, z_init_pbo)
                    loss = loss + ic_loss

                ELBO_HIVAE = -torch.mean(log_prob_stat - KL_Zstatic - KL_Sstatic, 0)
                ELBO_HIVAE = self.config.lambda_RecStat * ELBO_HIVAE
                loss = loss + ELBO_HIVAE

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                if (iter + 1) % self.config.print_freq == 0 or (iter + 1) == len(self.dataloader):
                    losses = OrderedDict()

                    losses['ELBO_NODE'] = ELBO_NODE.item()
                    losses['ELBO_Stat'] = ELBO_HIVAE.item()
                    if self.config.or_losses:
                        losses['OR_TRT'] = loss_treatment.item()
                        losses['OR_DO'] = loss_dropout.item()
                    if self.config.time_to_event:
                        losses['TE_Loss'] = te_log_loss.item()
                    if self.config.inv_ic_loss:
                        losses['IC_Loss'] = ic_loss.item()
                    losses['Batch_Loss'] = loss.item()
                    losses['Avg_Loss'] = avg_loss / (iter + 1)
                    progress_bar.set_postfix(**losses)

            t_epoch = time.time() - epoch_time_init
            t_total = time.time() - total_time
            print_current_losses(epoch, global_steps, losses,
                t_epoch, t_total, self.config.save_path_losses,
                s_excel=True)

            avg_loss = avg_loss / (iter + 1)
            if (epoch) % self.config.save_freq == 0:
                self.save(epoch, avg_loss)
            if avg_loss < self.best_loss:
                self.save(epoch, avg_loss, best=True)
                self.best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > self.config.patience:
                self.config.num_epochs = epoch
                print('Early Stoped in epoch ', epoch)
                break

        self.save(epoch, avg_loss)