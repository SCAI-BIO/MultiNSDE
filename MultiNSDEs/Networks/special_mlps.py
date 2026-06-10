import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import DistH
from .norm_act import (
    get_act, get_norm)
from .imputation import ImpLayer
from .define_mlps import make_bl_mean_var_mlp

class MLP_BL_Gaussian(nn.Module):
    def __init__(self, config, mod_num=0, data_implayer=None):
        super(MLP_BL_Gaussian, self).__init__()

        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(config, data_implayer)

        self.mod_num = mod_num
        i_size = config.n_long_var[mod_num]
        h_size = config.nhidden_lenc[mod_num]
        target_size = config.long_ldim[mod_num]
        nlayers = config.nlayers_mlp_lenc[mod_num]
        self.act_mean = get_act(config.act_mean[mod_num])
        self.act_var = get_act(config.act_var[mod_num])
        norm_mean = config.norm_mean[mod_num]
        norm_var = config.norm_var[mod_num]
    
        self.mean, self.var  = make_bl_mean_var_mlp(
            i_size, h_size, target_size, nlayers,
            norm_mean, norm_var,
            self.act_mean, self.act_var)

        self.learn_mean = config.learn_mean
        self.DistH = DistH(config)

    def forward(self, inputs, mask, val=False):

        if self.imp_layer:
            if ~torch.all(mask == 1.0):
                inputs = self.Imp_Layer(inputs, mask)

        mean = self.mean(inputs)
        var = self.var(inputs)
        if self.learn_mean:
            std = F.softplus(var) + 1e-6
        else:
            std = torch.exp(0.5 * var)

        z = self.DistH.sampling_z(mean, std, val=val)
        KL = self.DistH.kl_z_prior(mean, std)

        if val:
            return z, mean, std
        else:
            return z, KL

class DeepSurvHead(nn.Module):
    def __init__(self, inp_dim):
        super(DeepSurvHead, self).__init__()
        self.risk_net = nn.Sequential(
            nn.Linear(inp_dim, inp_dim//2),
            nn.ReLU(),
            nn.Linear(inp_dim//2, 1))

    def get_loss(self, log_risk, durations, events):
        """
        cox_ph_loss
        log_risk: tensor (N,) = model output (unscaled log-risk, real numbers)
        durations: tensor (N,) floats
        events: tensor (N,) 0/1 ints (1 if event observed)
        Returns negative partial log-likelihood (scalar)
        Implementation uses sorting-by-duration trick to compute risk sets efficiently.
        """

        # sort by descending durations (so risk set for time t is prefix)
        order = torch.argsort(durations, descending=True)
        lr_sorted = log_risk[order]
        events_sorted = events[order]

        # compute cumulative log-sum-exp of the risk (numerically stable)
        # but easier: compute cumulative sum of exp(log_risk) for risk sums
        exp_lr = torch.exp(lr_sorted)
        # cumulative sum over sorted exp_lr gives denominator for each observed event
        # risk_set_sum_at_i = sum_{j <= i} exp_lr[j] because descending sort
        risk_set_cumsum = torch.cumsum(exp_lr, dim=0)

        # for positions where event==1, partial log-likelihood term:
        # sum_i events_i * (lr_i - log(risk_set_sum_at_i))
        # select only event positions
        observed_idx = (events_sorted == 1)
        if observed_idx.sum() == 0:
            # no events -> loss 0 (or tiny)
            return torch.tensor(0., device=log_risk.device, requires_grad=True)

        lr_obs = lr_sorted[observed_idx]
        denom_obs = risk_set_cumsum[observed_idx]

        pll = torch.sum(lr_obs - torch.log(denom_obs + 1e-12))
        return pll / (observed_idx.sum().float())
 
    def forward(self, x):
        return self.risk_net(x)