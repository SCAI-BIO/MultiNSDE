from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import constraints as const
from torch.distributions import kl_divergence
from .distributions import DistH, ReparameterizedCategorical
from .imputation import ImpLayer
from .hivae_norm import HIVAE_NORM

class HIVAE(nn.Module):
    def __init__(self, config, static_types, data_implayer=None):
        super(HIVAE, self).__init__()

        self.var_types = static_types
        self.norm = config.hivae_norm
        if self.norm:
            self.Norm_Layer = HIVAE_NORM()

            # Normalization parameters
            self.register_buffer(
                "_mean_data",
                torch.zeros(len(self.var_types), requires_grad=False),
            )
            self.register_buffer(
                "_std_data",
                torch.ones(len(self.var_types), requires_grad=False),
            )

            self.momentum = 0.01
            self._batch_mean_data = self._batch_std_data = None

        self.imp_layer = config.hivae_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(data_implayer)

        self.Encoder = HIVAE_Encoder(
            config)
        self.Decoder = HIVAE_Decoder(
            config, static_types)

    @property
    def normalization_parameters(self):
        if self._batch_std_data is not None:
            return self._batch_mean_data, self._batch_std_data
        else:
            return self._mean_data, self._std_data

    def update_params(self, params):
        new_mean = (1 - self.momentum) * self._mean_data + self.momentum * params[0]
        new_std = (1 - self.momentum) * self._std_data + self.momentum * params[1]
        self._mean_data = new_mean
        self._std_data = new_std
        # self._batch_mean_data = params[0]
        # self._batch_std_data = params[1]

    def dec_samplings(self, static_data, mask, z0_stat, tau=1e-3):

        if self.norm:
            new_static_data, new_mask, norm_params = self.Norm_Layer(
                static_data, mask, self.var_types,
                self.normalization_parameters)
            self.update_params(norm_params)
        else:
            new_static_data = static_data

        if self.imp_layer:
            if ~torch.all(new_mask == 1.0):
                new_static_data = self.Imp_Layer(new_static_data, new_mask)
        s_samples = self.Encoder.dec_samplings(new_static_data, tau)

        samples = self.Decoder.dec_samplings(
            z0_stat, s_samples,
            self.normalization_parameters)
        
        return samples


    def forward(self, static_data, mask, tau=1, val=False):
        # self._batch_mean_data = self._batch_std_data = None
        if self.norm:
            new_static_data, new_mask, norm_params = self.Norm_Layer(
                static_data, mask, self.var_types,
                self.normalization_parameters)
            self.update_params(norm_params)
        else:
            new_static_data = static_data

        if self.imp_layer:
            if ~torch.all(new_mask == 1.0):
                new_static_data = self.Imp_Layer(new_static_data, new_mask)

        enc_output = self.Encoder(new_static_data, tau, val=val)
        (s_samples, logits_s,
            z0_stat, mean_qz_stat, std_qz_stat 
            )= enc_output

        samples, log_prob, KL_S, KL_Z = self.Decoder(
            static_data, mask, z0_stat,
            s_samples, logits_s,
            mean_qz_stat, std_qz_stat,
            self.normalization_parameters)

        if val:
            return z0_stat, mean_qz_stat, std_qz_stat
        else:
            return z0_stat, samples, log_prob, KL_S, KL_Z

class HIVAE_Encoder(nn.Module):
    def __init__(self, config):
        super(HIVAE_Encoder, self).__init__()

        zinp_dim = config.s_data_dim + config.s_dim_static
        z_dim = config.stat_ldim

        self.z_mean = nn.Linear(zinp_dim, z_dim)
        self.z_logvar = nn.Linear(zinp_dim, z_dim)
        self.s_logits = nn.Linear(config.s_data_dim, config.s_dim_static)

        self.DistH = DistH(config)
        self.learn_mean = config.learn_mean

    def dec_samplings(self, static_data, tau=1e-3, val=True):
        logits_s = self.s_logits(static_data)
        probs_s = F.softmax(logits_s, dim=-1).clamp(1e-6, 1 - 1e-6)

        s_samples = self.DistH.sampling_q(probs_s, tau, val=val)
        return s_samples

    def forward(self, static_data, tau=1, val=False):

        # Using GMM Prior after running modules through NN
        logits_s = self.s_logits(static_data)
        probs_s = F.softmax(logits_s, dim=-1).clamp(1e-6, 1 - 1e-6)

        s_samples = self.DistH.sampling_q(probs_s, tau, val=val)
        x_s = torch.cat((static_data, s_samples), dim=1)

        mean = self.z_mean(x_s)
        var = self.z_logvar(x_s)
        if self.learn_mean:
            std = F.softplus(var) + 1e-6
        else:
            std = torch.exp(0.5 * var)
        z = self.DistH.sampling_z(mean, std, val=val)
        return s_samples, logits_s, z, mean, std

class HIVAE_Decoder(nn.Module):
    def __init__(self, config, static_types):
        super(HIVAE_Decoder, self).__init__()

        self.norm = config.hivae_norm
        if self.norm:
            self.Norm_Layer = HIVAE_NORM()

        ydim = config.s_vals_dim
        sdim = config.s_dim_static
        zdim = config.stat_ldim

        self.prior_s_val = nn.Parameter(
            torch.ones(sdim) / sdim, requires_grad=False)
        self.prior_loc_z = nn.Linear(sdim, zdim, bias=True)

        learn_mean = config.learn_mean
        self.ylayer = nn.Linear(zdim, ydim)
        self.var_types = static_types
        self.mode = config.mode

        self.heads = nn.ModuleList()
        for i in range(ydim):
            if static_types[i, 0] == 'real':
                self.heads.append(RealHead(sdim, ydim, learn_mean=learn_mean))
            elif static_types[i, 0] in ['cat', 'ord']:
                self.heads.append(CatHead(sdim, ydim, static_types[i, 1]))
            elif static_types[i, 0] == 'count':
                self.heads.append(CountHead(sdim, ydim))
            elif static_types[i, 0] == 'pos':
                self.heads.append(PosHead(sdim, ydim))
            elif static_types[i, 0] == 'gamma':
                self.heads.append(GammaHead(sdim, ydim))
            elif static_types[i, 0] == 'bernoulli':
                self.heads.append(BernoulliHead(sdim, ydim))

    @property
    def prior_s(self):
        return dist.OneHotCategorical(
            probs=self.prior_s_val, validate_args=False)

    def prior_z(self, loc):
        return dist.normal.Normal(loc, torch.ones_like(loc))

    def kl_s(self, logits_s):
        return kl_divergence(
                dist.OneHotCategorical(
                    logits=logits_s, validate_args=False),
                self.prior_s)

    def kl_z(self, mean_qz, std_qz, mean_pz, std_pz):
        return kl_divergence(
                dist.normal.Normal(mean_qz, std_qz),
                dist.normal.Normal(mean_pz, std_pz)).sum(1)

    def dec_samplings(self, samples_z, samples_s,
                      norm_parameters=None):
        y = self.ylayer(samples_z)
        x_params = []
        for head in self.heads:
            x_params.append(head(samples_s, y))

        if self.norm:
            x_params = self.Norm_Layer.denormalize_params(
                x_params, self.var_types, norm_parameters)

        samples = [torch.Tensor([-1])] * len(self.var_types)
        
        for i, (head_i, params_i) in enumerate(
            zip(self.heads, x_params)):
            head_i.init_dist(params_i)
            samples[i] = (head_i.sample())

        samples = torch.cat(samples, dim=1)
        return samples

    def forward(self, data, mask, samples_z, samples_s,
                logits_s, mean_qz_stat, std_qz_stat,
                norm_parameters=None):

        y = self.ylayer(samples_z)
        x_params = []
        for head in self.heads:
            x_params.append(head(samples_s, y))

        if self.norm:
            x_params = self.Norm_Layer.denormalize_params(
                x_params, self.var_types, norm_parameters)

        log_probs = [torch.Tensor([-1])] * len(self.var_types)
        samples = [torch.Tensor([-1])] * len(self.var_types)
        
        for i, (x_i, m_i, head_i, params_i) in enumerate(
            zip(data.T, mask.T, self.heads, x_params)):
            head_i.init_dist(params_i)
            if m_i.ndim == 1:
                m_i = m_i.unsqueeze(1)
            log_probs[i] = head_i.log_prob(x_i) * m_i
            # During training we do not need the samples instead the
            # log prob. To train the model correctly we would need to
            # use rsample to get the gradients. However, the Poisson
            # distribution does not provide rsample function. So far
            # we have not found a better option
            samples[i] = head_i.sample()

            # The correct implementation would look something like the 
            # following lines. However, as mentioned we do not need 
            # the samples to train the model
            # samples[i] = (head_i.rsample() if not self.val
            #                 else head_i.sample())

        log_prob = torch.cat(log_probs, dim=1)  # batch, features
        log_prob = log_prob.sum(dim=1)  # batch

        KL_S = self.kl_s(logits_s)

        mean_pz = self.prior_loc_z(samples_s)
        std_pz = torch.ones_like(mean_pz)
        KL_Z = self.kl_z(mean_qz_stat, std_qz_stat, mean_pz, std_pz)

        samples = torch.cat(samples, dim=1)  # batch, features
        return samples, log_prob, KL_S, KL_Z

class BasicHead(nn.Module, ABC):
    def __init__(self):
        super(BasicHead, self).__init__()
        self._dist = None

    @abstractmethod
    def init_dist(self, params):
        raise NotImplementedError

    def sample(self):
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")

        gen_sample = self._dist.sample()
        if gen_sample.ndim == 1:
            return gen_sample.unsqueeze(1)
        elif gen_sample.ndim == 0:
            # ensure 2 dim output
            return gen_sample.unsqueeze(0).unsqueeze(1)
        return gen_sample

    def rsample(self):
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")

        gen_sample = self._dist.rsample()
        if gen_sample.ndim == 1:
            return gen_sample.unsqueeze(1)
        return gen_sample

    @property
    def mode(self):
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")
        res = self._dist.mode
        if res.ndim == 1:
            return res.unsqueeze(1)
        return res


class RealHead(BasicHead):
    def __init__(self, sdim, ydim, learn_mean=False):
        super(RealHead, self).__init__()

        self.mean = nn.Linear(ydim + sdim, 1)
        self.var = nn.Linear(sdim, 1)
        const_ = const.greater_than(0)
        self.lower_bound = const_.lower_bound
        self._distclass = dist.Normal
        self.learn_mean=learn_mean

    def init_dist(self, params):
        self._dist = self._distclass(params[0], params[1])

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data)

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        mean = self.mean(s_and_y)
        var = self.var(samples_s)

        if self.learn_mean:
            std = F.softplus(var)
        else:
            std = torch.exp(0.5 * var)
        std = self.lower_bound + 1e-15 + std
        return (mean, std)


class GammaHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(GammaHead, self).__init__()

        self.concentration_layer = nn.Linear(ydim + sdim, 1)
        self.rate_layer = nn.Linear(sdim, 1)
        self._distclass = dist.Gamma
        const_ = const.greater_than(0)
        self.lower_bound = const_.lower_bound

    def init_dist(self, params):
        self._dist = self._distclass(params[0], params[1])

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data.clamp(min=1e-3))

    def forward(self, samples_s, y):
        y = F.relu(y)
        s_and_y = torch.cat([y, samples_s], dim=-1)

        concentration = F.softplus(
            self.concentration_layer(s_and_y))        
        concentration = self.lower_bound + 1e-15 + concentration

        # rate = self.rate_layer(samples_s).min(torch.tensor(1e-3))
        rate = F.softplus(self.rate_layer(samples_s).min(torch.tensor(1e-3)))
        rate = self.lower_bound + 1e-15 + rate
        return (concentration, rate)


class PosHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(PosHead, self).__init__()

        self.mean = nn.Linear(ydim + sdim, 1)
        self.std = nn.Linear(sdim, 1)
        const_ = const.greater_than(0)
        self.lower_bound = const_.lower_bound
        self._distclass = dist.LogNormal

    def init_dist(self, params):
        self._dist = self._distclass(params[0], params[1])

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data.clamp(min=1e-3))

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        mean = self.mean(s_and_y)
        std = F.softplus(self.std(samples_s))
        std = self.lower_bound + 1e-15 + std
        return (mean, std)


class CountHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(CountHead, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.lambda_layer = nn.Linear(ydim + sdim, 1)
        const_ = const.greater_than(0)
        self.lower_bound = const_.lower_bound
        self._distclass = dist.Poisson

    def init_dist(self, params):
        self._dist = self._distclass(params)

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data)

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        rate = F.softplus(self.lambda_layer(s_and_y))
        rate = self.lower_bound + 1e-15 + rate
        return (rate)


class BernoulliHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(BernoulliHead, self).__init__()

        self.logit_layer = nn.Linear(ydim + sdim, 1)
        self._distclass = dist.Bernoulli

    def init_dist(self, params):
        self._dist = self._distclass(logits=params)

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data)

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        logits = self.logit_layer(s_and_y)
        return (logits)


class CatHead(BasicHead):
    def __init__(self, sdim, ydim, num_classes):
        super(CatHead, self).__init__()

        self.logit_layer = nn.Linear(ydim + sdim, num_classes)
        self._distclass = ReparameterizedCategorical

    def init_dist(self, params):
        self._dist = self._distclass(logits=params)

    def log_prob(self, data):
        log_prob_ = self._dist.log_prob(data)
        if data.ndim == 1:
            log_prob_ = log_prob_.unsqueeze(1)
        return log_prob_

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        logits = self.logit_layer(s_and_y)
        return (logits)
