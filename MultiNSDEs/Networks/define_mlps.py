import numpy as np
import torch.nn as nn
from .norm_act import get_norm

def make_ic_mlp(z_inp, hsize, ic_size, nlayers, norm_name, act):
    layers = []
    for i in range(nlayers):

        # The last layer does not have an activation
        if nlayers == 1:
            layers.append(nn.Linear(z_inp, ic_size))
        elif i == 0:
            layers.append(nn.Linear(z_inp, hsize))
            layers.append(get_norm(norm_name, hsize))
            if act != 'none':
                layers.append(act)
        elif i == (nlayers - 1):
            layers.append(nn.Linear(hsize, ic_size))
        else:
            layers.append(nn.Linear(hsize, hsize))
            layers.append(get_norm(norm_name, hsize))
            if act != 'none':
                layers.append(act)

    return nn.Sequential(*layers)


def make_mean_var_mlp(h_size, out_size, nlayers, norm_mean, norm_var, act_mean, act_var):
    mean_layers, var_layers = [], []
    for i in range(nlayers):
        if nlayers == 1 or i == (nlayers - 1):
            mean_layers.append(nn.Linear(h_size, out_size))
            var_layers.append(nn.Linear(h_size, out_size))
        else:
            mean_layers.append(nn.Linear(h_size, h_size))
            mean_layers.append(get_norm(norm_mean, h_size))
            if act_mean != 'none':
                mean_layers.append(act_mean)

            var_layers.append(nn.Linear(h_size, h_size))
            var_layers.append(get_norm(norm_var, h_size))
            if act_var != 'none':
                var_layers.append(act_var)

    return nn.Sequential(*mean_layers), nn.Sequential(*var_layers)


def make_bl_mean_var_mlp(i_size, h_size, out_size, nlayers, norm_mean, norm_var, act_mean, act_var):
    mean_layers, var_layers = [], []
    for i in range(nlayers):
        if nlayers == 1:
            mean_layers.append(nn.Linear(i_size, out_size))
            var_layers.append(nn.Linear(i_size, out_size))
        elif i == 0:
            mean_layers.append(nn.Linear(i_size, h_size))
            mean_layers.append(get_norm(norm_mean, h_size))
            if act_mean != 'none':
                mean_layers.append(act_mean)
            var_layers.append(nn.Linear(i_size, h_size))
            var_layers.append(get_norm(norm_var, h_size))
            if act_var != 'none':
                var_layers.append(act_var)
        elif nlayers - 1:
            mean_layers.append(nn.Linear(h_size, out_size))
            var_layers.append(nn.Linear(h_size, out_size))
        else:
            mean_layers.append(nn.Linear(h_size, h_size))
            mean_layers.append(get_norm(norm_mean, h_size))
            if act_mean != 'none':
                mean_layers.append(act_mean)
            var_layers.append(nn.Linear(h_size, h_size))
            var_layers.append(get_norm(norm_var, h_size))
            if act_var != 'none':
                var_layers.append(act_var)

    return nn.Sequential(*mean_layers), nn.Sequential(*var_layers)


def get_bottleneck_architecture(num_de_layers, latent_dim, h_size,
                                out_dim, diff, act, norm_name, ncde=False):
    layers_enc = []
    layers_dec = []

    # Encoder and decoder parts of the ODE function
    # input and output are just mirrored
    # at the end we reverse the dec list to be aligned
    # with the encoder one
    for i in range(num_de_layers):

        if num_de_layers == 1:
            layers_enc.append(nn.Linear(latent_dim, h_size))
            if ncde:
                layers_dec.append(nn.Linear(h_size, latent_dim*out_dim))
            else:
                layers_dec.append(nn.Linear(h_size, out_dim))
        else:
            fact_i = i/num_de_layers
            fact_o = (i+1)/num_de_layers

            c_i = latent_dim + int(np.round(diff*fact_i))
            c_o = latent_dim + int(np.round(diff*fact_o))

            if fact_o == 1:
                layers_enc.append(nn.Linear(c_i, h_size))                
                layers_dec.append(nn.Linear(h_size, c_i))
            else:
                layers_enc.append(nn.Linear(c_i, c_o))
                layers_enc.append(get_norm(norm_name, c_o))
                if i == 0:
                    if ncde:
                        layers_dec.append(nn.Linear(c_o, latent_dim*out_dim))
                    else:
                        layers_dec.append(nn.Linear(c_o, out_dim))
                else:
                    layers_dec.append(nn.Linear(c_o, c_i))
                layers_dec.append(get_norm(norm_name, c_o)) # use c_i for all models

        if i > -1:
            layers_enc.append(act)
            layers_dec.append(act)

    layers_enc.append(nn.Linear(h_size, h_size))
    layers_enc.append(act)
    layers_dec = layers_dec[:-1]  #remove the last act layer
    layers_dec.reverse()
    layers = layers_enc + layers_dec  # concat enc + dec
    return layers