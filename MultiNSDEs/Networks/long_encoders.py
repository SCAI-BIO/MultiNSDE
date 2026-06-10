import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_act import get_act
from .distributions import DistH
from .imputation import ImpLayer
from .define_mlps import make_mean_var_mlp


# ========================================
# =========== RNN/LSTM ENCODERS ==========
# ========================================
class RevRNNEncoder(nn.Module):
    def __init__(self, config, mod_num=0, data_implayer=None):
        super(RevRNNEncoder, self).__init__()

        self.extrapolation = config.extrapolation
        self.mode = config.mode
        self.type_mod_enc = config.type_hivae
        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(config, data_implayer)

        # The LSTM is done in a Reversed way to learn the initial conditions
        # of the ODE System. 
        i_size = config.n_long_var[mod_num]
        h_size = config.nhidden_lenc[mod_num]
        target_size = config.rhs_ldim[mod_num]
        rnn_nlayers = config.nlayers_rnn_enc[mod_num]
        nlayers = config.nlayers_mlp_lenc[mod_num]

        self.act_mean = get_act(config.act_mean[mod_num])
        self.act_var = get_act(config.act_var[mod_num])
        norm_mean = config.norm_mean[mod_num]
        norm_var = config.norm_var[mod_num]

        self.Enc = config.type_lenc[mod_num]
        if self.Enc == 'LSTM':
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                               batch_first=True, dropout=0, bidirectional=False)

        out_size = target_size
        num_nns = 1

        self.mean = nn.ModuleList()
        self.var = nn.ModuleList()
        for _ in range(num_nns):
            mean_net, var_net = make_mean_var_mlp(
                h_size, out_size, nlayers,
                norm_mean, norm_var,
                config.act_mean, config.act_var)
            self.mean.append(mean_net)
            self.var.append(var_net)

        self.h_size = h_size
        self.rnn_layers = rnn_nlayers
        self.learn_mean = config.learn_mean
        self.DistH = DistH(config)

    def forward(self, inputs, mask, time=None, tau=None, val=False):

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
        mean = torch.cat([net(out) for net in self.mean], dim=-1)
        var  = torch.cat([net(out) for net in self.var], dim=-1)

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
    def __init__(self, config, mod_num=0, data_implayer=None):
        super(RNNEncoder, self).__init__()

        self.extrapolation = config.extrapolation
        self.mode = config.mode
        self.type_mod_enc = config.type_hivae
        self.imp_layer = config.long_implayer
        if self.imp_layer:
            self.Imp_Layer = ImpLayer(config, data_implayer)

        # Forward LSTM encoder
        i_size = config.n_long_var[mod_num]
        h_size = config.nhidden_lenc[mod_num]
        target_size = config.rhs_ldim[mod_num]
        rnn_nlayers = config.nlayers_rnn_enc[mod_num]
        nlayers = config.nlayers_mlp_lenc[mod_num]

        self.act_mean = get_act(config.act_mean[mod_num])
        self.act_var = get_act(config.act_var[mod_num])
        norm_mean = config.norm_mean[mod_num]
        norm_var = config.norm_var[mod_num]

        self.Enc = config.type_lenc[mod_num]
        if self.Enc == 'LSTM':
            self.rnn = nn.LSTM(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        else:
            self.rnn = nn.RNN(i_size, h_size, num_layers=rnn_nlayers, bias=True,
                               batch_first=True, dropout=0, bidirectional=False)

        out_size = target_size
        num_nns = 1

        self.mean = nn.ModuleList()
        self.var = nn.ModuleList()
        for _ in range(num_nns):
            mean_net, var_net = make_mean_var_mlp(
                h_size, out_size, nlayers,
                norm_mean, norm_var,
                config.act_mean, config.act_var)
            self.mean.append(mean_net)
            self.var.append(var_net)

        self.learn_mean = config.learn_mean
        self.DistH = DistH(config)

    def forward(self, inputs, mask, time=None, tau=None, val=False):
       
        if self.imp_layer:
            if ~torch.all(mask == 1.0):
                inputs = self.Imp_Layer(inputs, mask)

        out, _ = self.rnn(inputs)
        out = out[:, -1, :]
        mean = torch.cat([net(out) for net in self.mean], dim=-1)
        var  = torch.cat([net(out) for net in self.var], dim=-1)

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