import torch
import torch.nn as nn
from .norm_act import get_act
from .long_encoders import *
from .hivae import HIVAE
from .special_mlps import (
    MLP_BL_Gaussian, DeepSurvHead)
from .define_mlps import make_ic_mlp

class Mod_Encoder_BL_HIVAE(nn.Module):
    def __init__(self, config, static_types,
                 ldata_implayer=None,
                 sdata_implayer=None,
                 scaling_static_stats=None):
        super(Mod_Encoder_BL_HIVAE, self).__init__()

        self.config = config
        self.num_modules = config.num_lenc
        self.modules_ = nn.ModuleList()
        self.modules_BL = nn.ModuleList()
        self.prognostic_model = True if 'Causal' not in config.type_hivae else False
        for i in range(self.num_modules):
            
            if self.prognostic_model:
                # Longitudinal Encoder
                if self.config.rev_lenc:
                    L_Enc = RevRNNEncoder(
                        self.config, mod_num=i,
                        data_implayer=[ldata_implayer[0][i],
                                       ldata_implayer[1][i],
                                       ldata_implayer[2][i]])
                else:
                    L_Enc = RNNEncoder(
                        self.config, mod_num=i,
                        data_implayer=[ldata_implayer[0][i],
                                       ldata_implayer[1][i],
                                       ldata_implayer[2][i]])
                self.modules_.append(L_Enc)

            BL_Enc = MLP_BL_Gaussian(
                self.config, mod_num=i,
                data_implayer=[ldata_implayer[0][i][:,0],
                               ldata_implayer[1][i][:,0],
                               ldata_implayer[2][i]]) # So that it runs
            self.modules_BL.append(BL_Enc)

        z_inp = sum(config.long_ldim)
        self.HIVAE = HIVAE(self.config, static_types,
                        data_implayer=sdata_implayer,
                        scaling_static_stats=scaling_static_stats)
        z_inp += self.config.stat_ldim

        # In this modular encoder we are always using a projection layer to get the IC
        # either using only the long latent representations of combined with the static
        # latent representations
        nlayers = self.config.nlayers_projec
        hsize = self.config.nhidden_projec
        ic_size = self.config.IC_size
        act = get_act(self.config.act_proj)
        norm_name = self.config.norm_proj
        self.MLP = make_ic_mlp(z_inp, hsize, ic_size, nlayers, norm_name, act)

        self.time_to_event = self.config.time_to_event
        if self.time_to_event:
            self.te_heads = nn.ModuleList()
            for i in range(config.num_var_te):
                self.te_heads.append(DeepSurvHead(ic_size + config.n_RHS_Feat))

    def get_out(self, outs_):

        if isinstance(outs_[0], tuple):
            outs = []
            for no in range(len(outs_[0])):

                out_to_concat = [out[no] for out in outs_]
                if out_to_concat[0].ndim == 1:
                    # KL Div
                    conc_out = torch.stack(out_to_concat, dim=1)
                    conc_out = (conc_out.sum(1) if self.config.comb_long_loss == 'sum'
                                else conc_out.mean(1))
                elif out_to_concat[0].ndim == 2:
                    # Z0_Long, Mean, Std
                    conc_out = torch.cat(out_to_concat, dim=-1)
                else:
                    conc_out = out_to_concat
                outs.append(conc_out)
        elif isinstance(outs_[0], torch.Tensor):
            # DL decodings
            outs = outs_
        else:
            outs = torch.cat(outs_, dim=1)
        return outs

    def encode_long(self, L_Data, L_Mask, T, tau=1e-3, val=False):
        outs_BL = []
        for i, mod_ in enumerate(self.modules_BL):
            outs_BL.append(
                mod_(L_Data[i][:,0], L_Mask[i][:,0], val=val))

        if not self.prognostic_model:
            return self.get_out(outs_BL)

        outs_ = []
        for i, mod_ in enumerate(self.modules_):
            outs_.append(
                mod_(L_Data[i], L_Mask[i], T[i], tau=tau, val=val))
        
        return (self.get_out(outs_BL), self.get_out(outs_))

    def encode_stat(self, S_Data, S_Mask, tau=1e-3, val=False):
        return self.HIVAE.enc_data(S_Data, S_Mask, tau, val=val)

    def decode_stat(self, S_Data, S_Mask, z0, tau=1e-3):
        return self.HIVAE.dec_samplings(S_Data, S_Mask, z0, tau)

    def project_zinit(self, z_init):
        return self.MLP(z_init)

    def get_risk(self, z_init, Drug_TE):
        risk = []
        for i, head_i in enumerate(self.te_heads):
            risk_i = head_i(torch.cat((z_init, Drug_TE), dim=-1))
            risk.append(risk_i)
        return torch.cat(risk, -1)

    def forward(self, L_Data, L_Mask, T, Time_TE=None, Event_TE= None,
                Drug_TE= None, S_Data=None, S_Mask=None, tau=1e-3, val=False):

        outs_ = []
        for i, mod_ in enumerate(self.modules_BL):
            outs_.append(
                mod_(L_Data[i][:,0], L_Mask[i][:,0]))
        z0_long, KL_Long = self.get_out(outs_)

        if self.prognostic_model:
            outs_ = []
            for i, mod_ in enumerate(self.modules_):
                outs_.append(
                    mod_(L_Data[i], L_Mask[i], T[i], tau=tau))

            RHS_Long, KL_RHS_Long = self.get_out(outs_)
            KL_Long = KL_Long + KL_RHS_Long
        else:
            RHS_Long = z0_long

        z0_stat, _, log_prob, KL_S, KL_Zstatic = self.HIVAE(
            S_Data, S_Mask, tau=tau, val=val)
        z_init = torch.cat((z0_long, z0_stat), dim=1)
        z_init = self.MLP(z_init)

        if self.time_to_event:
            te_loss = []
            for i, head_i in enumerate(self.te_heads):
                risk_i = head_i(torch.cat((z_init, Drug_TE), dim=-1))
                te_loss_i = head_i.get_loss(risk_i[:, 0], Time_TE[:, i], Event_TE[:, i])
                te_loss.append(te_loss_i)
            te_loss = torch.stack(te_loss, -1)
        else:
            te_loss = None

        return (z_init, KL_Long, RHS_Long,
                te_loss, log_prob,
                KL_S, KL_Zstatic)
