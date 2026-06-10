import torch
import torch.nn as nn

def initialize_imputation(data):
    
    if len(data) == 3:
        # Static data
        X, data_types, W = data

        W_A = W.sum(0)
        A = torch.sum(X * W, 0)
        # Initializing using a weighted average
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        cat_indices = [index for index, entry in enumerate(data_types)
                       if entry[0] == 'cat']
        A[cat_indices] = 0
    else:
        X, W = data
        W_A = W.sum(0)
        A = torch.sum(X * W, 0)
        # Initializing using a weighted average
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if W_A[i, j] == 0:
                    A[i, j] = torch.sum(X[:, :, j]) / torch.sum(W[:, :, j])
                    W_A[i, j] = 1
    
    return A

# ========================================
# =========== VADER IMPUTATION ===========
# ========================================
class ImpLayer(nn.Module):
    def __init__(self, config, data):
        super(ImpLayer, self).__init__()

        self.extrapolation = config.extrapolation
        self.mode = config.mode
        data, self.time_Encs = data[:2], data[-1]

        data = list(data)
        if self.extrapolation and self.mode != 'train':
            for i in range(len(data)):
                if data[i].ndim == 3 and self.time_Encs[-1] > 1:
                    data[i] = data[i][:, :-1, :]
        data = tuple(data)

        A_init = initialize_imputation(data)
        self.b = nn.Parameter(A_init)

    def forward(self, X, W):
        if self.time_Encs[-1] > 1 and X.ndim == 3:
            X = X[:, :-1, :] 
            W = W[:, :-1, :] 

        # X is the data and w is the indicator function
        # Handle missing values section of the main text
        return (1 - W) * self.b + X * W 
