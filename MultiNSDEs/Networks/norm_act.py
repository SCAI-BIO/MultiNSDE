import torch.nn as nn

def get_act(act):
    if act == 'tanh':
        act = nn.Tanh()
    elif act == 'relu':
        act = nn.ReLU()
    elif act == 'selu':
        act = nn.SELU()
    elif act == 'softplus':
        act = nn.Softplus()
    elif act == 'sigmoid':
        act = nn.Sigmoid()
    else:
        act = nn.Identity()
    return act

def get_norm(norm, num_feat):
    if norm == 'instance':
        norm = nn.InstanceNorm1d(num_feat)
    elif norm == 'batch':
        norm = nn.BatchNorm1d(num_feat)
    else:
        norm = nn.Identity()
    return norm