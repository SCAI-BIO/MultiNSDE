from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import constraints as const
from torch.distributions import kl_divergence
from .distributions import (
    DistH, ReparameterizedCategorical)
from .imputation import ImpLayer
from .hivae_norm import HIVAE_NORM

class HIVAE(nn.Module):
    def __init__(self, config, static_types, data_implayer=None, scaling_static_stats=None):
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
            self._batch_std_data = None

        self.imp_layer = config.hivae_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(data_implayer)

        self.Encoder = HIVAE_Encoder(
            config)
        self.Decoder = HIVAE_Decoder(
            config, static_types,
            scaling_static_stats=scaling_static_stats)

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

    def dec_samplings(self, static_data, mask, z0, tau=1e-3):

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
            z0, s_samples,
            self.normalization_parameters)
        
        return samples

    def enc_data(self, static_data, mask, tau=1, val=False, full_outputs=False):

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
        if full_outputs:
            return enc_output
        else:
            (s_samples, _,
                z0_stat, mean_qz_stat, std_qz_stat 
                )= enc_output
            return s_samples, z0_stat, mean_qz_stat, std_qz_stat

    def forward(self, static_data, mask, tau=1, val=False):
        enc_output = self.enc_data(static_data, mask, tau, val=val, full_outputs=True)
        (s_samples, logits_s, z0_stat,
         mean_qz_stat, std_qz_stat) = enc_output

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
    def __init__(self, config, mod_num=0, long=False):
        super(HIVAE_Encoder, self).__init__()

        if long:
            s_data_dim = config.nhidden_lenc[mod_num]
            s_dim = config.s_dim_long[mod_num]
            zinp_dim = s_data_dim + s_dim
            z_dim = config.long_ldim[mod_num]
        else:
            s_data_dim = config.s_data_dim
            s_dim = config.s_dim_static
            zinp_dim = s_data_dim + s_dim
            z_dim = config.stat_ldim

        self.z_mean = nn.Linear(zinp_dim, z_dim)
        self.z_logvar = nn.Linear(zinp_dim, z_dim)
        self.s_logits = nn.Linear(s_data_dim, s_dim)

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
    def __init__(self, config, static_types, scaling_static_stats=None):
        super(HIVAE_Decoder, self).__init__()

        self.norm = config.hivae_norm
        if self.norm:
            self.Norm_Layer = HIVAE_NORM()

        ydim = config.s_vals_dim
        sdim = config.s_dim_static
        zdim = config.stat_ldim
        zydim = config.stat_ldim if 'SLR' in config.type_hivae else config.IC_size

        self.prior_s_val = nn.Parameter(
            torch.ones(sdim) / sdim, requires_grad=False)
        self.prior_loc_z = nn.Linear(sdim, zdim, bias=True)

        learn_mean = config.learn_mean
        nlayers=config.hivae_nl_mlp
        if nlayers > 1:
            # Geometric "average"
            layers = [round(zydim**((nlayers-i-1)/nlayers) * ydim**((i+1)/nlayers)) for i in range(nlayers-1)]
            layers_sizes = [zydim] + layers + [ydim]
            layers_seq = []
            for i in range(len(layers_sizes)-1):
                layers_seq.append(nn.Linear(layers_sizes[i], layers_sizes[i+1]))
                if i < len(layers_sizes)-2:
                    layers_seq.append(nn.ReLU())
            self.ylayer = nn.Sequential(*layers_seq)
        else:
            self.ylayer = nn.Linear(zydim, ydim)
        self.var_types = static_types
        self.mode = config.mode

        # it only have the names for the continuos variables
        names = np.array(scaling_static_stats['columns'])
        count = len(names)

        if config.clipping:
            lower_bounds = np.full(count, -float("inf"))
            upper_bounds = np.full(count,  float("inf"))

            if config.dataset == 'PROACT':
                lower_bounds[names == 'Age'] = 0
                upper_bounds[names == 'Age'] = 100

            lower_bounds = torch.tensor(lower_bounds, dtype=torch.float32)
            upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32)

            self.clipping_static_lower = lower_bounds
            self.clipping_static_upper = upper_bounds
        else:
            self.clipping_static_lower = [None] * count
            self.clipping_static_upper = [None] * count

        self._val = False if config.mode == 'train' else True
        self._MAP_sampling = config.HIVAE_MAP_sampling
        self.ydim = ydim
        self.heads = nn.ModuleList()
        clip_idx=0
        for i in range(ydim):
            if static_types[i, 0] == 'real':
                self.heads.append(RealHead(sdim, ydim, learn_mean=learn_mean))
                clip_idx += 1
            elif static_types[i, 0] == 'cat':
                self.heads.append(CatHead(sdim, ydim, static_types[i, 1]))
            elif static_types[i, 0] == 'ord':
                self.heads.append(OrdHead(sdim, ydim, static_types[i, 1]))
            elif static_types[i, 0] == 'bce':
                self.heads.append(BCEHead(sdim, ydim))
            elif static_types[i, 0] == 'count':
                self.heads.append(CountHead(sdim, ydim))
            elif static_types[i, 0] == 'pos':
                self.heads.append(PosHead(sdim, ydim))
                clip_idx += 1
            elif static_types[i, 0] == 'mse':
                self.heads.append(MSEHead(sdim, ydim,
                                          self.clipping_static_lower[clip_idx],
                                          self.clipping_static_upper[clip_idx]))
                clip_idx += 1
            elif static_types[i, 0] == 'gamma':
                self.heads.append(GammaHead(sdim, ydim))
                clip_idx += 1
            elif static_types[i, 0] == 'bernoulli':
                self.heads.append(BernoulliHead(sdim, ydim))
            else:
                raise ValueError(f"Unsupported head type: {static_types[i, 0]}")
        
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
                x_params, self.var_types[:self.ydim], norm_parameters)

        samples = [torch.Tensor([-1])] * len(self.var_types)
        
        for i, (head_i, params_i) in enumerate(
            zip(self.heads, x_params)):
            head_i.init_dist(params_i,
                             MAP_sampling = self._MAP_sampling,
                             val = self._val)
            if isinstance(head_i, MSEHead):
                samples[i] = params_i
            elif isinstance(head_i, BCEHead):
                samples[i] = (params_i > 0.5).float()
            else:
                samples[i] = (head_i.sample())

        samples = torch.cat(samples, dim=1)
        return samples

    def forward(self, data, mask, samples_z, samples_s,
                logits_s, mean_qz_stat, std_qz_stat,
                norm_parameters=None):

        y = self.ylayer(samples_z)
        x_params = []
        # Get the distribution parameters using samples_s and „y“
        for head in self.heads:
            x_params.append(head(samples_s, y))

        if self.norm:
            x_params = self.Norm_Layer.denormalize_params(
                x_params, self.var_types[:self.ydim], norm_parameters)

        # [:self.ydim] is used to be able to only use some variables as input
        # but without the need to decode them
        # Code assumes the variables that are not decoded are always at the end
        log_probs = [torch.Tensor([-1])] * len(self.var_types[:self.ydim])
        samples = [torch.Tensor([-1])] * len(self.var_types[:self.ydim])

        for i, (x_i, m_i, head_i, params_i) in enumerate(
            zip(data.T, mask.T, self.heads, x_params)):
            head_i.init_dist(params_i,
                             MAP_sampling = self._MAP_sampling,
                             val = self._val)
            if m_i.ndim == 1:
                m_i = m_i.unsqueeze(1)

            # During training we do not need the samples instead the
            # log prob. To train the model correctly we could also
            # use rsample to get the gradients. However, the Poisson
            # distribution does not provide rsample function. So far
            # we have not found a better option
            if isinstance(head_i, MSEHead):
                log_probs[i] = head_i.get_loss(x_i) * m_i
                samples[i] = params_i
            elif isinstance(head_i, BCEHead):
                log_probs[i] = head_i.get_loss(x_i) * m_i
                samples[i] = (params_i > 0.5).float()
            else:
                log_probs[i] = head_i.log_prob(x_i) * m_i
                samples[i] = head_i.sample()

            # The other implementation would look something like the 
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
        self._MAP_sampling = False
        self._val = False

    @abstractmethod
    def init_dist(self, params):
        raise NotImplementedError

    def sample(self, max_V=100):
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")

        if self._val and self._MAP_sampling:
            if self._distclass in [dist.Normal, dist.LogNormal, dist.Gamma]:
                gen_sample = self._dist.mean.detach()
            elif self._distclass in [dist.Poisson, dist.Bernoulli, ReparameterizedCategorical]:
                gen_sample = self._dist.mode.detach()
            else:
                raise ValueError(f"Unsupported MAP sampling for head type: {self._distclass}")
        else:
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


class MSEHead(BasicHead):
    def __init__(self, sdim, ydim, lower_bound=None, upper_bound=None):
        super(MSEHead, self).__init__()
        self.mean = nn.Linear(ydim + sdim, 1)
        if lower_bound is not None:
            const_ = const.greater_than(lower_bound)
            self.register_buffer("lower_bound", torch.tensor(const_.lower_bound))
        else:
            self.lower_bound = None

        if upper_bound is not None:
            const_ = const.less_than(upper_bound)
            self.register_buffer("upper_bound", torch.tensor(const_.upper_bound))
        else:
            self.upper_bound = None

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = params
        self._MAP_sampling = MAP_sampling
        self._val = val

    def get_loss(self, data):
        # We are calculating here the MSE and returning its negative
        # to be compatible with the other heads
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return -(self._dist - data)**2

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        mean = self.mean(s_and_y)
        if self.lower_bound is not None or self.upper_bound is not None:
            mean = torch.clamp(mean, min=self.lower_bound, max=self.upper_bound)
        return (mean)


class RealHead(BasicHead):
    def __init__(self, sdim, ydim, learn_mean=False):
        super(RealHead, self).__init__()

        self.mean = nn.Linear(ydim + sdim, 1)
        self.var = nn.Linear(sdim, 1) if sdim != 0 else nn.Linear(ydim, 1)
        const_ = const.greater_than(0)
        self.register_buffer("lower_bound", torch.tensor(const_.lower_bound))
        self._distclass = dist.Normal
        self.learn_mean=learn_mean

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(params[0], params[1])
        self._MAP_sampling = MAP_sampling
        self._val = val

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data)

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        mean = self.mean(s_and_y)
        var = self.var(samples_s) if samples_s.numel() > 0 else self.var(y)
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
        self.rate_layer = nn.Linear(sdim, 1) if sdim != 0 else nn.Linear(ydim, 1)
        self._distclass = dist.Gamma
        const_ = const.greater_than(0)
        self.register_buffer("lower_bound", torch.tensor(const_.lower_bound))

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(params[0], params[1])
        self._MAP_sampling = MAP_sampling
        self._val = val

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

        rate = F.softplus(self.rate_layer(samples_s).min(torch.tensor(1e-3))) if samples_s.numel() > 0 else (
            F.softplus(self.rate_layer(y).min(torch.tensor(1e-3))))
        rate = self.lower_bound + 1e-15 + rate
        return (concentration, rate)


class PosHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(PosHead, self).__init__()

        self.mean = nn.Linear(ydim + sdim, 1)
        self.std = nn.Linear(sdim, 1) if sdim != 0 else nn.Linear(ydim, 1)
        const_ = const.greater_than(0)
        self.register_buffer("lower_bound", torch.tensor(const_.lower_bound))
        self._distclass = dist.LogNormal

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(params[0], params[1])
        self._MAP_sampling = MAP_sampling
        self._val = val

    def log_prob(self, data):
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return self._dist.log_prob(data.clamp(min=1e-3))

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        mean = self.mean(s_and_y)
        std = F.softplus(self.std(samples_s)) if samples_s.numel() > 0 else F.softplus(self.std(y))
        std = self.lower_bound + 1e-15 + std
        return (mean, std)


class CountHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(CountHead, self).__init__()
        self.lambda_layer = nn.Linear(ydim + sdim, 1)
        const_ = const.greater_than(0)
        self.register_buffer("lower_bound", torch.tensor(const_.lower_bound))
        self._distclass = dist.Poisson

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(params)
        self._MAP_sampling = MAP_sampling
        self._val = val

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

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(logits=params)
        self._MAP_sampling = MAP_sampling
        self._val = val

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

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(logits=params)
        self._MAP_sampling = MAP_sampling
        self._val = val

    def log_prob(self, data):
        log_prob_ = self._dist.log_prob(data)
        if data.ndim == 1:
            log_prob_ = log_prob_.unsqueeze(1)
        return log_prob_

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        logits = self.logit_layer(s_and_y)
        return (logits)


class OrdHead(BasicHead):
    def __init__(self, sdim, ydim, num_classes):
        super(OrdHead, self).__init__()

        self.threshold_layer = nn.Linear(ydim + sdim, num_classes - 1)
        self._distclass = ReparameterizedCategorical

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = self._distclass(probs=params)
        self._MAP_sampling = MAP_sampling
        self._val = val

    def log_prob(self, data):
        log_prob_ = self._dist.log_prob(data)
        if data.ndim == 1:
            log_prob_ = log_prob_.unsqueeze(1)
        return log_prob_

    def forward(self, samples_s, y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        thresholds = self.threshold_layer(s_and_y)
        # To be sure they are ordered
        thresholds = torch.cumsum(torch.exp(thresholds), dim=-1)
        cum_probs = torch.sigmoid(thresholds)
        cum_probs = torch.cat([torch.zeros_like(cum_probs[..., :1]),
                               cum_probs, torch.ones_like(cum_probs[..., :1])],
                               dim=-1)
        # Probabilities per class = difference of cumulative probs
        probs = cum_probs[..., 1:] - cum_probs[..., :-1]
        return (probs)


class BCEHead(BasicHead):
    def __init__(self, sdim, ydim):
        super(BCEHead, self).__init__()
        self.logit_layer = nn.Linear(ydim + sdim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def init_dist(self, params, MAP_sampling=False, val=False):
        self._dist = params
        self._MAP_sampling = MAP_sampling
        self._val = val

    def get_loss(self, data):
        # We are calculating here the BCE and returning its negative
        # to be compatible with the other heads
        if data.ndim == 1:
            data = data.unsqueeze(1)
        return -self.bce_loss(self._dist, data)
 
    def forward(self, samples_s ,y):
        s_and_y = torch.cat([y, samples_s], dim=-1)
        logits = self.logit_layer(s_and_y)        
        return (logits)