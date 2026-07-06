#!/usr/bin/ipython
import os
import numpy as np
import torch
import pandas as pd
from glob import glob
from parser import base_parser
import sys
sys.path.append('../../')
from Common_Functions.data.load_data_helpers import *
from Common_Functions.models.val_utils import breslow_baseline_from_risk
import warnings
warnings.filterwarnings('ignore')

def main(config):

    print('Getting raw data')
    val_runs = config.nruns_ppd
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)
    epoch = config.epoch_init

    # ===================================
    # ======== Load the data ============
    # ===================================
    data = config.save_path
    data = torch.load(data)

    dfs_path = config.folder_path
    raw_path = os.path.join(dfs_path, 'Raw_Output')
    os.makedirs(raw_path, exist_ok=True)

    Total_Z_init = data['Z_init'][:, :val_runs]
    Total_Z_init = Total_Z_init.numpy()

    B, R, N = Total_Z_init.shape  # For Total_Z_init (B, R, N)
    Total_Z_init = Total_Z_init.reshape(B * R, N)  # Flatten Total_Z_init to (B*R) x N

    PTNOs = data['PTNO'].repeat_interleave(R).numpy()
    PBO = data['PBO'].repeat_interleave(R).numpy()

    Zinit_df = pd.DataFrame(Total_Z_init, columns=[f'Feature_{i+1}' for i in range(N)])
    Zinit_df['PTNO'] = PTNOs
    Zinit_df['PBO'] = PBO
    Zinit_df['REPI'] = [r + 1 for _ in range(B) for r in range(R)]  # REPI > 0 for replicates
    Zinit_df = Zinit_df.sort_values(by=['PTNO', 'REPI']).reset_index(drop=True)

    # Reorder columns to have 'Batch' and 'REPI' first
    Zinit_df = Zinit_df[['PTNO', 'REPI', 'PBO'] + [f'Feature_{i+1}' for i in range(N)]]
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)
    path = os.path.join(raw_path, 'Zinit_%sEP%d.csv'%(Val_Scenario,epoch))
    Zinit_df.to_csv(path, index=False)
    if config.dataset == 'A4_Causal':
        TRT = 1 - PBO
        PS_SCORES = data['PS_Scores'].repeat_interleave(R).numpy()
        OBS_SCORES = data['Obs_Scores'].repeat_interleave(R).numpy()
        OBS = np.round(data['OBS'].numpy().reshape(B * R, 2), 3)
        MASK = data['MASK'].numpy().reshape(B * R, 2)
        IPRED = data['IPRED'].numpy().reshape(B * R, 2)
        PBO_SIMS = np.round(data['PBO_SIMS'].numpy().reshape(B * R, 2), 3)
        TRT_SIMS = np.round(data['TRT_SIMS'].numpy().reshape(B * R, 2), 3)

        df = pd.DataFrame({
            "PTNO": PTNOs,
            "TRT": TRT.squeeze(),
            "REPI": [r + 1 for _ in range(B) for r in range(R)],
            "OBS_BL": OBS[:, 0].squeeze(),
            "MASK_BL": MASK[:, 0].squeeze(),
            "IPRED_BL": IPRED[:, 0].squeeze(),
            "PBO_SIMS_BL": PBO_SIMS[:, 0].squeeze(),
            "TRT_SIMS_BL": TRT_SIMS[:, 0].squeeze(),
            "OBS_END": OBS[:, 1].squeeze(),
            "MASK_END": MASK[:, 1].squeeze(),
            "IPRED_END": IPRED[:, 1].squeeze(),
            "PBO_SIMS_END": PBO_SIMS[:, 1].squeeze(),
            "TRT_SIMS_END": TRT_SIMS[:, 1].squeeze(),
            "PS_SCORES": PS_SCORES.squeeze(),
            "OBS_SCORES": OBS_SCORES.squeeze()})

        path = os.path.join(raw_path, 'Output_%sEP%d.csv'%(Val_Scenario,epoch))
        df.to_csv(path, index=False)

    if config.dataset == 'DATATOP_Causal' and config.time_to_event and 'Risk' in data and 'Time_TE' in data and 'Event_TE' in data:
        time = data['Time_TE']
        event = data['Event_TE']
        te_names = list(data['VarNames_TE'])
        treat = data['TREAT'] if 'TREAT' in data else (1 - data['PBO']).long()

        arm_ids = sorted([
            int(key.replace('Risk_ARM', ''))
            for key in data.keys()
            if key.startswith('Risk_ARM')
        ])
        sim_arm_ids = sorted([
            int(key.replace('SIMS_Risk_ARM', ''))
            for key in data.keys()
            if key.startswith('SIMS_Risk_ARM')
        ])

        risk_records = []
        for arm in arm_ids:
            risk_arm = data[f'Risk_ARM{arm}']
            for b in range(risk_arm.shape[0]):
                factual = int(int(treat[b].item()) == arm)
                for d, var_name in enumerate(te_names):
                    risk_records.append({
                        'PTNO': int(data['PTNO'][b].item()),
                        'TREAT': int(treat[b].item()),
                        'PBO': int(data['PBO'][b].item()),
                        'REPI': 0,
                        'Arm': arm,
                        'Factual': factual,
                        'Var': var_name,
                        'Time': float(time[b, d]),
                        'Event': float(event[b, d]),
                        'Risk': float(risk_arm[b, d]),
                    })

        for arm in sim_arm_ids:
            sim_risk_arm = data[f'SIMS_Risk_ARM{arm}'][:, :val_runs]
            for b in range(sim_risk_arm.shape[0]):
                factual = int(int(treat[b].item()) == arm)
                for n in range(sim_risk_arm.shape[1]):
                    for d, var_name in enumerate(te_names):
                        risk_records.append({
                            'PTNO': int(data['PTNO'][b].item()),
                            'TREAT': int(treat[b].item()),
                            'PBO': int(data['PBO'][b].item()),
                            'REPI': n + 1,
                            'Arm': arm,
                            'Factual': factual,
                            'Var': var_name,
                            'Time': float(time[b, d]),
                            'Event': float(event[b, d]),
                            'Risk': float(sim_risk_arm[b, n, d]),
                        })

        risk_df = pd.DataFrame(risk_records)
        risk_df = risk_df.sort_values(by=['PTNO', 'REPI', 'Arm', 'Var']).reset_index(drop=True)
        path = os.path.join(raw_path, 'Risk_TE_%sEP%d.csv' % (Val_Scenario, epoch))
        risk_df.to_csv(path, index=False)

        factual_risk = risk_df[(risk_df['REPI'] == 0) & (risk_df['Factual'] == 1)].copy()
        if not factual_risk.empty:
            baseline_risk = breslow_baseline_from_risk(factual_risk)
            path = os.path.join(raw_path, 'Baseline_Risk_TE_EP%d.csv' % (epoch))
            baseline_risk.to_csv(path, index=False)

    print('Raw outputs were saved')

if __name__ == '__main__':

    config = base_parser()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.or_model_path = os.path.join(config.save_path, config.dataset, 'OR_ATEmodels')
    config.save_path = os.path.join(config.save_path, config.dataset, 
                                    config.exp_name, 'Fold%d'%(config.train_fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')

    name_ = 'Val_Imgs_%s'%(config.val_data_type)
    config.folder_path = os.path.join(config.save_path_samples, name_)
    save_path = os.path.join(config.save_path_samples, name_)
    Val_Scenario = '' if config.Val_Scenario == 0 else 'Val%s_'%(config.Val_Scenario)

    # If there are more than 1 checkpoint, the model will do the plots
    # for all the saved checkpoints
    if config.from_best:
        opcs = glob(save_path + '/*.pth')
        epoch = 1
        for opc in opcs:
            if 'Best' in opc:
                epoch_opc = opc.split('/')[-1].split('Ep')[-1].split('.pth')[0]
    
                if int(epoch_opc) > epoch:
                    epoch = int(epoch_opc)
        config.epoch_init = epoch
        config.save_path = os.path.join(
            save_path, 'Results_Best_%sEp%d.pth'%(Val_Scenario,config.epoch_init))
    else:
        config.save_path = os.path.join(
            save_path, 'Results_%sEp%d.pth'%(Val_Scenario,config.epoch_init))
    main(config)