import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_act import (
    get_act, get_norm)
from .distributions import DistH
from .imputation import ImpLayer

# ========================================
# ========= LONGITUDINAL MODELS ==========
# ========================================
class RevRNNEncoder(nn.Module):
    def __init__(self, config, data_implayer=None):
        super(RevRNNEncoder, self).__init__()

        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(config, data_implayer)

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
        if self.Enc == 'LSTM':
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                               batch_first=True, dropout=0, bidirectional=False)

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

    def forward(self, inputs, mask, time=None, val=False):

        if self.imp_layer:
            if ~torch.all(mask == 1.0):
                inputs = self.Imp_Layer(inputs, mask)

        h = torch.zeros(self.rnn_layers , inputs.size(0), self.h_size).to(inputs.device)
        if self.Enc == 'LSTM':
            c = torch.zeros(self.rnn_layers , inputs.size(0), self.h_size).to(inputs.device)

        for t in reversed(range(inputs.size(1))):
            long_v = inputs[:, t:t+1, :]  # longitudinal variable in time t
            if self.Enc == 'LSTM':
                out, (h, c) = self.rnn(long_v, (h, c))
            else:
                out, h = self.rnn(long_v, h)

        out = out[:, 0, :]
        mean = self.mean(out)
        var = self.var(out)
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


class RNNEncoder(nn.Module):
    def __init__(self, config, data_implayer=None):
        super(RNNEncoder, self).__init__()
        
        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(data_implayer)

        # Forward LSTM encoder
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
        if self.Enc == 'LSTM':
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                               batch_first=True, dropout=0, bidirectional=False)

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

        self.learn_mean = config.learn_mean
        self.DistH = DistH(config)

    def forward(self, inputs, mask, time=None, val=False):

        if self.imp_layer:
            if ~torch.all(mask == 1.0):
                inputs = self.Imp_Layer(inputs, mask)

        out, _ = self.rnn(inputs)
        out = out[:, -1, :]
        mean = self.mean(out)
        var = self.var(out)

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


class RNNDecoder(nn.Module):
    def __init__(self, config):
        super(RNNDecoder, self).__init__()

        # Forward LSTM decoder
        i_size = config.long_ldim 
        if config.static_data and config.ode_static_data == 'IC':
            i_size = i_size + config.stat_ldim

        h_size = config.nhidden_ldec
        target_size = config.n_long_var
        rnn_nlayers = config.nlayers_rnn_dec
        nlayers = config.nlayers_mlp_ldec
        act_name = config.act_dec
        norm_name = config.norm_dec

        self.Dec = config.type_lenc
        if self.Dec == 'LSTM':
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)

        act = get_act(act_name)
        layers = []
        for i in range(nlayers):

            if nlayers == 1 or i == nlayers -1:
                layers.append(nn.Linear(h_size, target_size))
            else:
                layers.append(nn.Linear(h_size, h_size))
                layers.append(get_norm(norm_name, h_size))

            if i < nlayers -1:
                layers.append(act)
        self.lin = nn.Sequential(*layers)

    def forward(self, z):
        out, _ = self.rnn(z)
        # Here the linear layers are being run over the
        # features dimension for all the time steps
        out = self.lin(out)  # out is pred_x
        return out


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        act_name = config.act_dec
        act = get_act(act_name)
        norm_name = config.norm_dec
        nlayers = config.nlayers_mlp_ldec

        i_size = config.long_ldim 
        if config.static_data and config.ode_static_data == 'IC':
            i_size = i_size + config.stat_ldim
        out_dim = config.n_long_var
        h_size = config.nhidden_ldec

        layers = []
        for i in range(nlayers):

            if nlayers == 1:
                layers.append(nn.Linear(i_size, out_dim))
            elif i == 0:
                layers.append(nn.Linear(i_size, h_size))
                if nlayers > 2:
                    layers.append(get_norm(norm_name, h_size))
            elif i == nlayers -1:
                # Last layer does not have norm nor activation
                layers.append(nn.Linear(h_size, out_dim))
            else:
                layers.append(nn.Linear(h_size, h_size))
                layers.append(get_norm(norm_name, h_size))

            if i < nlayers -1:
                if i == nlayers - 2:
                    # It's convenient to use dropout after activation, but 
                    # in case of Relu before activation
                    if act_name == 'relu':
                        layers.append(nn.Dropout(config.drop_dec))
                        layers.append(act)
                    else:
                        layers.append(act)
                        layers.append(nn.Dropout(config.drop_dec))
                else:
                    layers.append(act)
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)