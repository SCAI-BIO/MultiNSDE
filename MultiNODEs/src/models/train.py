import numpy as np
from tqdm import tqdm
import time
import torch
from collections import OrderedDict
from utils import print_current_losses
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

            # Training along dataset            
            for iter, data in progress_bar:

                global_steps += 1
                X, W = data[0].to(self.device), data[1].to(self.device)

                # Get Z0_long
                z0_long, KL_Long = self.L_Enc(X, W, self.T)

                # Static Data 
                if self.config.static_data:
                    S_Data = data[2].to(self.device)
                    S_Mask = data[3].to(self.device)

                    z0_stat, _, log_prob, KL_S, KL_Zstatic = self.HIVAE(
                        S_Data, S_Mask, tau)

                    if self.config.ode_static_data == 'ADD_NN':
                        self.ODE.add_static_data(z0_stat)
                        z_init = z0_long
                    elif self.config.ode_static_data == 'ADD':
                        z_init = z0_long + z0_stat
                    else:
                        z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
                else:
                    z_init = z0_long

                # ODE Solver
                pred_z = self.run_ODE(z_init)
                pred_x = self.L_Dec(pred_z)

                # Implement weighted mean squared error for Vader
                rec_loss = W*(X - pred_x)**2
                rec_loss = (rec_loss.sum()*X.size(1)*X.size(2) / (W.sum())).mean()
                KL_Long = KL_Long.mean()
                ELBO_NODE = KL_Long + rec_loss

                if self.config.static_data:

                    ELBO_HIVAE = -torch.mean(log_prob - KL_Zstatic - KL_S, 0)

                    ELBO_NODE_SCALED = (ELBO_HIVAE/(ELBO_HIVAE + ELBO_NODE))*ELBO_NODE
                    ELBO_HIVAE_SCALED = (ELBO_HIVAE/(ELBO_HIVAE + ELBO_NODE))*ELBO_NODE


                    long = ELBO_NODE / (ELBO_NODE + ELBO_HIVAE)
                    stat = ELBO_HIVAE / (ELBO_NODE + ELBO_HIVAE)

                    # equations 6 and 7 in the paper
                    ELBO_NODE_SCALED = (stat / (long + stat)) *(ELBO_NODE)
                    ELBO_HIVAE_SCALED = self.config.lamba_ELBO * ((long / (long + stat)) * ELBO_HIVAE)
                    loss = ELBO_NODE_SCALED + ELBO_HIVAE_SCALED
                else:
                    loss = ELBO_NODE

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                if (iter + 1) % self.config.print_freq == 0 or (iter + 1) == len(self.dataloader):
                    losses = OrderedDict()
                    if self.config.static_data:
                        losses['L_Scaled'] = ELBO_NODE_SCALED.item()
                        losses['S_Scaled'] = ELBO_HIVAE_SCALED.item()
                        losses['Total'] = loss.item()
                    else:
                        losses['L_Rec'] = rec_loss.item()
                        losses['L_KL'] = KL_Long.item()
                        losses['Total'] = loss.item()

                    losses['Avg_ELBO'] = avg_loss / (iter + 1)
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
