import torch
import torch.nn as nn
import torchsde
from .norm_act import get_act
from .define_mlps import get_bottleneck_architecture

class NSDE_DRHS_Mix_Ito(torchsde.SDEIto):
    def __init__(self, config):
        super (NSDE_DRHS_Mix_Ito, self).__init__(noise_type="diagonal")

        act = get_act(config.act_de)
        norm_name = config.norm_de
        num_de_layers = config.nlayers_DE
        latent_dim = config.IC_size

        if config.ANDE:
            latent_dim += config.ANDE_dim
        out_dim = latent_dim

        if 'Causal' in config.type_hivae:
            latent_dim += sum(config.long_ldim)
        else:
            latent_dim += sum(config.rhs_ldim)

        latent_dim_placebo = latent_dim
        latent_dim = latent_dim + config.n_RHS_Feat

        h_size = config.nhidden_de
        diff = h_size - latent_dim
        diff_placebo = h_size - latent_dim_placebo

        layers = get_bottleneck_architecture(
            num_de_layers, latent_dim, h_size,
            out_dim, diff, act, norm_name)

        layers_placebo = get_bottleneck_architecture(
            num_de_layers, latent_dim_placebo, h_size,
            out_dim, diff_placebo, act, norm_name)

        self.model_drug = nn.Sequential(*layers)
        self.model_placebo = nn.Sequential(*layers_placebo)
        if config.sde_noise_type == 'fixed':
            self.noise_scale = config.sde_noise_init
        elif config.sde_noise_type == 'learned_all':
            self.noise_scale = nn.Parameter(torch.tensor(config.sde_noise_init))
        elif config.sde_noise_type == 'learned_ind':
            self.noise_scale = nn.Parameter(torch.full((out_dim,), config.sde_noise_init))

        self.DRHS_PBO_slide = config.DRHS_PBO_slide

    def set_data(self, dose_data, T, drhs_data, static_data=None):
        self.dose_data = dose_data
        self.T = T.cpu().numpy()
        self.drhs_data = drhs_data

    def f(self, t, x):        
        idx = self._get_index(t)
        dose_data = self.dose_data[:, idx]
        inp_ = torch.cat((x, dose_data, self.drhs_data), dim=-1)
        inp_pbo = torch.cat((x, self.drhs_data), dim=-1)

        out1 = self.model_drug(inp_)
        out2 = self.model_placebo(inp_pbo)
        Drug = (dose_data[self.DRHS_PBO_slide] > 0).any(dim=1, keepdim=True).float()
        out = Drug * out1 + out2
        return out

    def g(self, t, x):
        return self.noise_scale * torch.ones_like(x)

    def _get_index(self, t):
        t_scalar = t.item()
        idx = (self.T <= t_scalar).sum() - 1
        return min(max(idx, 0), len(self.T) - 1)

class NSDE_DRHS_Mix_Stra(torchsde.SDEStratonovich):
    def __init__(self, config):
        super (NSDE_DRHS_Mix_Stra, self).__init__(noise_type="diagonal")

        act = get_act(config.act_de)
        norm_name = config.norm_de
        num_de_layers = config.nlayers_DE
        latent_dim = config.IC_size

        if config.ANDE:
            latent_dim += config.ANDE_dim
        out_dim = latent_dim

        if 'Causal' in config.type_hivae:
            latent_dim += sum(config.long_ldim)
        else:
            latent_dim += sum(config.rhs_ldim)

        latent_dim_placebo = latent_dim
        latent_dim = latent_dim + config.n_RHS_Feat
        h_size = config.nhidden_de
        diff = h_size - latent_dim
        diff_placebo = h_size - latent_dim_placebo

        layers = get_bottleneck_architecture(
            num_de_layers, latent_dim, h_size,
            out_dim, diff, act, norm_name)

        layers_placebo = get_bottleneck_architecture(
            num_de_layers, latent_dim_placebo, h_size,
            out_dim, diff_placebo, act, norm_name)

        self.model_drug = nn.Sequential(*layers)
        self.model_placebo = nn.Sequential(*layers_placebo)
        if config.sde_noise_type == 'fixed':
            self.noise_scale = config.sde_noise_init
        elif config.sde_noise_type == 'learned_all':
            self.noise_scale = nn.Parameter(torch.tensor(config.sde_noise_init))
        elif config.sde_noise_type == 'learned_ind':
            self.noise_scale = nn.Parameter(torch.full((out_dim,), config.sde_noise_init))

        self.DRHS_PBO_slide = config.DRHS_PBO_slide

    def set_data(self, dose_data, T, drhs_data):
        self.dose_data = dose_data
        self.T = T.cpu().numpy()
        self.drhs_data = drhs_data

    def f(self, t, x):        
        idx = self._get_index(t)
        dose_data = self.dose_data[:, idx]
        inp_ = torch.cat((x, dose_data, self.drhs_data), dim=-1)
        inp_pbo = torch.cat((x, self.drhs_data), dim=-1)
        out1 = self.model_drug(inp_)
        out2 = self.model_placebo(inp_pbo)
        Drug = (dose_data[self.DRHS_PBO_slide] > 0).any(dim=1, keepdim=True).float()
        out = Drug * out1 + out2
        return out

    def g(self, t, x):
        return self.noise_scale * torch.ones_like(x)

    def _get_index(self, t):
        t_scalar = t.item()
        idx = (self.T <= t_scalar).sum() - 1
        return min(max(idx, 0), len(self.T) - 1)

def get_nsde_class(config):
    if config.method_solver in ['euler', 'midpoint', 'log_ode', 'milstein_ito',  'srk']:
        if config.method_solver == 'milstein_ito':
            config.method_solver = 'milstein'
        return NSDE_DRHS_Mix_Ito(config)
    elif config.method_solver in ['heun', 'euler_heun', 'adjoint_reversible_heun',
                                  'reversible_heun', 'milstein']:
        return NSDE_DRHS_Mix_Stra(config)
    else:
        raise ValueError(f"Unknown method {config.method_solver}")