import numpy as np
import torch
import torch.nn as nn
from .hivae import (
    RealHead, CatHead,
    OrdHead, CountHead,
    PosHead, MSEHead,
    GammaHead, BernoulliHead)

# ==============================================
# ========= LONGITUDINAL DIST ENCODERS =========
# ==============================================
# This version does not use a GMM prior on the latent space
class Decoder_Dist(nn.Module):
    def __init__(self, config, long_types, scaling_long_stats=None):
        super(Decoder_Dist, self).__init__()

        long_types = np.vstack(long_types)
        ydim = sum(config.n_long_var)
        sdim = 0

        zydim = config.IC_size
        if config.ANDE:
            zydim += config.ANDE_dim

        learn_mean = config.learn_mean
        nlayers=config.nlayers_mlp_ldec
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
        self.var_types = long_types

        names = np.array(scaling_long_stats['columns'])
        count = len(names)

        if config.clipping:
            lower_bounds = np.full(count, -float("inf"))
            upper_bounds = np.full(count,  float("inf"))

            if config.scaler != 'none' and config.dataset == 'A4':
                names = np.array(scaling_long_stats['columns'])
                lower_bounds[names == 'NPI'] = 0
                lower_bounds[names == 'BMI'] = 15
                lower_bounds[(names == 'Heart Rate') | (names == 'Weight')] = 40
            lower_bounds = torch.tensor(lower_bounds, dtype=torch.float32)

            if config.scaler == 'robust':
                long_var_medians = torch.from_numpy(np.array(scaling_long_stats['median'])).to(torch.float)
                long_var_iqrs =  torch.from_numpy(np.array(scaling_long_stats['iqr'])).to(torch.float)
                self.clipping_values = (lower_bounds - long_var_medians) / long_var_iqrs
            elif config.scaler == 'minmax':
                scale = torch.from_numpy(np.array(scaling_long_stats['scale'])).to(torch.float)
                datamin = torch.from_numpy(np.array(scaling_long_stats['datamin'])).to(torch.float)
                min_ = torch.from_numpy(np.array(scaling_long_stats['min_'])).to(torch.float)
                self.clipping_values = (lower_bounds- min_) * (1/scale) + datamin
            else:
                self.clipping_values = lower_bounds
            upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32)
            self.clipping_static_lower = lower_bounds
            self.clipping_static_upper = upper_bounds
        else:
            self.clipping_static_lower = [None] * count
            self.clipping_static_upper = [None] * count

        clip_idx=0

        self._val = False if config.mode == 'train' else True
        self._MAP_sampling = config.HIVAE_MAP_sampling
        self.heads = nn.ModuleList()
        for i in range(ydim):
            if long_types[i, 0] == 'real':
                self.heads.append(RealHead(sdim, ydim, learn_mean=learn_mean))
                clip_idx += 1
            elif long_types[i, 0] == 'cat':
                self.heads.append(CatHead(sdim, ydim, long_types[i, 1]))
            elif long_types[i, 0] == 'ord':
                self.heads.append(OrdHead(sdim, ydim, long_types[i, 1]))
            elif long_types[i, 0] == 'count':
                self.heads.append(CountHead(sdim, ydim))
            elif long_types[i, 0] == 'pos':
                self.heads.append(PosHead(sdim, ydim))
                clip_idx += 1
            elif long_types[i, 0] == 'mse':
                self.heads.append(MSEHead(sdim, ydim,
                                          self.clipping_static_lower[clip_idx],
                                          self.clipping_static_upper[clip_idx]))
                clip_idx += 1
            elif long_types[i, 0] == 'gamma':
                self.heads.append(GammaHead(sdim, ydim))
                clip_idx += 1
            elif long_types[i, 0] == 'bernoulli':
                self.heads.append(BernoulliHead(sdim, ydim))
            else:
                raise ValueError(f"Unsupported head type: {long_types[i, 0]}")

    def map_data(self, data, mask):
        output = torch.zeros_like(mask, dtype=torch.float)

        d_offset = 0
        for sub_data in data:
            _, _, D_i = sub_data.shape

            submask = mask[..., d_offset: d_offset + D_i]
            output[:, :, d_offset: d_offset + D_i][submask.bool()] = sub_data.view(-1)
            d_offset += D_i
        return output

    def dec_samplings(self, samples_z, mask, mask_de, tau=1e-3):

        y = self.ylayer(samples_z)
        num_timepoints = y.size(1)
        empty_ss = torch.empty(0).to(y.device)

        samples = [None] * num_timepoints
        for t in range(num_timepoints):
            y_ = y[:, t, :]
            samples_ = [torch.Tensor([-1])] * len(self.var_types)

            x_params = []
            for head in self.heads:
                x_params.append(head(empty_ss, y_))

            for i, (head_i, params_i) in enumerate(
                zip(self.heads, x_params)):
                head_i.init_dist(params_i,
                                 MAP_sampling = self._MAP_sampling,
                                 val = self._val)
                if isinstance(head_i, MSEHead):
                    samples_[i] = params_i
                else:
                    samples_[i] = head_i.sample()
            samples[t] = torch.stack(samples_, dim=1)[..., 0] # Batch x Dim

        samples = torch.stack(samples, 1) # B x T x Dim

        # This needs to be done because the input dimension of each
        # module can have different number of measurements
        out_list, probs_list = [], []
        i_id, end_id = 0, 0
        for i, mask_ in enumerate(mask):
            end_id += mask_.shape[-1]
            mask_de_ = mask_de[..., i_id:end_id]
            out_ = samples[..., i_id:end_id]

            B, _, D = mask_de_.shape
            K = mask_de_[0, :, 0].sum()
            out_ = out_[mask_de_.bool()].view(B, K, D)
            out_list.append(out_)

            # In this way to avoid editing the full code
            probs_list.append(torch.empty(0))
            i_id = end_id

        return out_list, probs_list

    def forward(self, data, mask, mask_de, samples_z):

        # To get the data of each specific encoder accordingly
        data = self.map_data(data, mask_de)
        mask = self.map_data(mask, mask_de)
        y = self.ylayer(samples_z)
        empty_ss = torch.empty(0).to(y.device)

        num_timepoints = y.size(1)
        log_probs = [None] * num_timepoints
        samples = [None] * num_timepoints
        for t in range(num_timepoints):
            sub_data = data[:, t, :]
            sub_mask = mask[:, t, :]
            y_ = y[:, t, :]

            log_probs_ = [torch.Tensor([-1])] * len(self.var_types)
            samples_ = [torch.Tensor([-1])] * len(self.var_types)

            x_params = []
            for head in self.heads:
                x_params.append(head(empty_ss, y_))

            for i, (x_i, m_i, head_i, params_i) in enumerate(
                zip(sub_data.T, sub_mask.T, self.heads, x_params)):

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
                    log_probs_[i] = head_i.get_loss(x_i) * m_i
                    samples_[i] = params_i
                else:
                    log_probs_[i] = head_i.log_prob(x_i) * m_i
                    samples_[i] = head_i.sample()

                # The other implementation would look something like the 
                # following lines. However, as mentioned we do not need 
                # the samples to train the model
                # samples[i] = (head_i.rsample() if not self.val
                #                 else head_i.sample())

            log_probs[t] = torch.stack(log_probs_, dim=1).sum(-1).sum(-1) # Batch
            samples[t] = torch.stack(samples_, dim=1)[..., 0] # Batch x Dim

        # Stacking to get B x T and then add across time
        log_probs = torch.stack(log_probs, 1).sum(-1) # Batch
        samples = torch.stack(samples, 1) # B x T x Dim

        return samples, log_probs