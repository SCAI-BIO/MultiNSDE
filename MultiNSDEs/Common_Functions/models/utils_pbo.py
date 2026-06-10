import torch

# ==================================================================#
# ==================================================================#
def split_by_pbo(Data, PBO_flag, static=False):

    # Flatten PBO_flag if needed
    if PBO_flag.ndim == 2 and PBO_flag.shape[1] == 1:
        PBO_flag = PBO_flag.squeeze(1)

    pbo_mask = PBO_flag == 1
    non_pbo_mask = PBO_flag == 0

    # To come back to the original order
    pbo_indices = torch.where(pbo_mask)[0]
    non_pbo_indices = torch.where(non_pbo_mask)[0]
    
    if static:
        Data_pbo = Data[pbo_mask]
        Data_non_pbo = Data[non_pbo_mask]
    else:
        Data_pbo = [ld[pbo_mask] for ld in Data]
        Data_non_pbo = [ld[non_pbo_mask] for ld in Data]

    return Data_pbo, Data_non_pbo, pbo_indices, non_pbo_indices

def reassing_data(data_pbo,data_npbo,idx_pbo,idx_non_pbo,PBO_flag):
    data_combined = torch.empty((PBO_flag.shape[0], data_pbo.shape[1]),
                                device=data_pbo.device)
    data_combined[idx_non_pbo] = data_npbo
    data_combined[idx_pbo]     = data_pbo
    return data_combined

def get_data_npbo_pbo(L_Data,L_Mask,PBO_flag,S_Data=None,S_Mask=None):
    L_Data_pbo, L_Data_non_pbo, idx_pbo, idx_non_pbo = split_by_pbo(L_Data,PBO_flag)
    L_Mask_pbo, L_Mask_non_pbo,_,_ = split_by_pbo(L_Mask,PBO_flag)
    if S_Data is not None and S_Mask is not None:
        S_Data_pbo, S_Data_non_pbo,_,_ = split_by_pbo(S_Data,PBO_flag,static=True)
        S_Mask_pbo, S_Mask_non_pbo,_,_ = split_by_pbo(S_Mask,PBO_flag,static=True)
    else:
        S_Data_pbo, S_Data_non_pbo = None, None
        S_Mask_pbo, S_Mask_non_pbo = None, None
    
    return (L_Data_pbo, L_Data_non_pbo, idx_pbo, idx_non_pbo,
            L_Mask_pbo, L_Mask_non_pbo, S_Data_pbo,
            S_Data_non_pbo, S_Mask_pbo, S_Mask_non_pbo)