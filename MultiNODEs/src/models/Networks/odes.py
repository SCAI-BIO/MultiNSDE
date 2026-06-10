import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from .norm_act import (
    get_act, get_norm)
from .distributions import DistH
from .imputation import ImpLayer

# ========================================
# ================= ODEs =================
# ========================================
class InternalODE(nn.Module):
    # Latent ODE with bottleneck structure
    # num_odelayers specifies the depth of the Neural ODE
    def __init__(self, h_size):
        super (InternalODE, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.Tanh(),
            nn.Linear(h_size, h_size))

    def forward(self, t, x):
        out = self.model(x)
        return out
    
class RNNODEEncoder(nn.Module):
    def __init__(self, config, data_implayer=None):
        super(RNNODEEncoder, self).__init__()

        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(data_implayer)

        # The LSTM is done in a Reversed way to learn the initial conditions
        # of the ODE System. 
        i_size = config.n_long_var
        h_size = config.nhidden_lenc
        target_size = config.long_ldim
        rnn_nlayers = config.nlayers_rnn_enc
        nlayers = config.nlayers_mlp_lenc

        self.act_mean = get_act(config.act_mean)
        self.act_var = get_act(config.act_var)
        norm_mean = config.norm_mean
        norm_var = config.norm_var

        self.Enc = config.type_lenc
        self.reversed = config.rev_lenc
        if 'LSTM' in self.Enc:
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                               batch_first=True, dropout=0, bidirectional=False)

        self.F_node = InternalODE(h_size)

        self.method = config.method_solver
        self.rtol = config.rtol
        self.atol = config.atol

        mean, var = [], []
        for i in range(nlayers):

            # The last layer does not have an activation
            if nlayers == 1 or i == (nlayers - 1):
                mean.append(nn.Linear(h_size, target_size))
                var.append(nn.Linear(h_size, target_size))
            else:
                mean.append(nn.Linear(h_size, h_size))
                mean.append(get_norm(norm_mean, h_size))
                if config.act_mean != 'none':
                    mean.append(self.act_mean)
                var.append(nn.Linear(h_size, h_size))
                var.append(get_norm(norm_var, h_size))
                if config.act_var != 'none':
                    var.append(self.act_var)

        self.mean = nn.Sequential(*mean)
        self.var = nn.Sequential(*var)
        self.h_size = h_size
        self.rnn_layers = rnn_nlayers

        self.learn_mean = config.learn_mean
        self.DistH = DistH(config)

    def forward(self, inputs, mask, Time_FNode, val=False):

        if self.imp_layer:
            if ~torch.all(mask == 1.0):
                inputs = self.Imp_Layer(inputs, mask)

        h = torch.zeros(self.rnn_layers , inputs.size(0), self.h_size).to(inputs.device)
        if 'LSTM' in self.Enc:
            c = torch.zeros(self.rnn_layers , inputs.size(0), self.h_size).to(inputs.device)

        if self.reversed:
            inputs = torch.flip(inputs, [1])
            Time_FNode = torch.flip(Time_FNode, [0])

        for t, (t0, t1) in enumerate(zip(Time_FNode[:-1], Time_FNode[1:])):
            long_v = inputs[:, t:t+1, :]  # longitudinal variable in time t
            time_interval = torch.Tensor([t0 - t0, abs(t1 - t0)]).to(long_v.device)

            if 'LSTM' in self.Enc:
                _, (h, c) = self.rnn(long_v, (h, c))
            else:
                _, h = self.rnn(long_v, h)

            h = odeint(self.F_node, h[0], time_interval, rtol=self.rtol,
                       atol=self.atol, method=self.method)
            h = h[-1:]

        h = h[0, :]
        mean = self.mean(h)
        var = self.var(h)

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


class LatentODE(nn.Module):
    # Latent ODE with bottleneck structure
    # num_odelayers specifies the depth of the Neural ODE
    def __init__(self, config):
        super (LatentODE, self).__init__()

        act = get_act(config.act_ode)
        norm_name = config.norm_ode
        num_ode_layers = config.nlayers_ODE

        latent_dim = config.long_ldim 
        self.ode_static_data = False
        if config.static_data:

            self.static_data = 0
            if config.ode_static_data == 'IC':
                latent_dim = latent_dim + config.stat_ldim
            elif config.ode_static_data == 'ADD_NN':
                self.ode_static_data = True
                nlayers = config.nlayers_ODE_staticdata
                act_name = config.act_ode_staticdata
                norm_name = config.norm_ode_staticdata
                act = get_act(act_name)
                layers = []

                for i in range(nlayers):

                    if i == 0:
                        layers.append(nn.Linear(config.stat_ldim + 1, latent_dim))
                    else:
                        layers.append(nn.Linear(latent_dim, latent_dim))

                    if nlayers > 2 and (i < (nlayers -1)):
                        layers.append(get_norm(norm_name, latent_dim))
                        layers.append(act)
                        
                self.static = nn.Sequential(*layers)
            else:
                assert (config.long_ldim == config.stat_ldim), 'Longitudinal and '\
                    'Static latent dimensions needs to be the same'

        h_size = config.nhidden_ode
        diff = h_size - latent_dim

        layers_enc = []
        layers_dec = []

        # Encoder and decoder parts of the ODE function
        # input and output are just mirrored
        # at the end we reverse the dec list to be aligned
        # with the encoder one
        for i in range(num_ode_layers):

            if num_ode_layers == 1:
                layers_enc.append(nn.Linear(latent_dim, h_size))
                layers_dec.append(nn.Linear(h_size, latent_dim))
            else:
                fact_i = i/num_ode_layers
                fact_o = (i+1)/num_ode_layers

                c_i = latent_dim + int(np.round(diff*fact_i))
                c_o = latent_dim + int(np.round(diff*fact_o))

                if fact_o == 1:
                    layers_enc.append(nn.Linear(c_i, h_size))
                    layers_dec.append(nn.Linear(h_size, c_i))
                else:
                    layers_enc.append(nn.Linear(c_i, c_o))
                    layers_enc.append(get_norm(norm_name, c_o))
                    layers_dec.append(nn.Linear(c_o, c_i))
                    layers_dec.append(get_norm(norm_name, c_i))

            layers_enc.append(act)
            layers_dec.append(act)

        layers_enc.append(nn.Linear(h_size, h_size))
        layers_enc.append(act)
        layers_dec = layers_dec[:-1]  #remove the last act layer
        layers_dec.reverse()
        layers = layers_enc + layers_dec  # concat enc + dec

        self.model = nn.Sequential(*layers)

    def add_static_data(self, static_data):
        self.static_data = static_data

    def forward(self, t, x):

        if self.ode_static_data:
            aux = t.repeat(self.static_data.shape[0], 1)
            inp_s = torch.cat((self.static_data, aux), 1)
            x = x + self.static(inp_s)
        out = self.model(x)

        return out