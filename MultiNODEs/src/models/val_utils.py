import torch
import numpy as np
import pandas as pd

@torch.no_grad()
def unscale(data, ids, i_c, e_c, scaling_long_stats, scaler='robust'):
    if scaler == 'robust':
        long_var_medians = torch.from_numpy(np.array(scaling_long_stats['median'])).to(torch.float)
        long_var_iqrs =  torch.from_numpy(np.array(scaling_long_stats['iqr'])).to(torch.float)
        data[...,ids] = (data[..., ids] * long_var_iqrs[i_c:e_c]) + long_var_medians[i_c:e_c]
    else:
        scale = torch.from_numpy(np.array(scaling_long_stats['scale'])).to(torch.float)
        datamin = torch.from_numpy(np.array(scaling_long_stats['datamin'])).to(torch.float)
        min_ = torch.from_numpy(np.array(scaling_long_stats['min_'])).to(torch.float)
        data[...,ids] = (data[..., ids] -  min_[i_c:e_c]) * (1 / scale[i_c:e_c]) + datamin[i_c:e_c]

    return data

@torch.no_grad()
def transform_back(data, c_long_ids, log_scaler):
    if log_scaler != 'none' and len(c_long_ids) > 0:
        if log_scaler == 'log1p':
            data[..., c_long_ids] =  torch.expm1(data[..., c_long_ids])
        elif log_scaler == 'log':
            data[..., c_long_ids] =  torch.exp(data[..., c_long_ids]) - 1e-12
    return data

def breslow_baseline_from_risk(df, time_col='Time', event_col='Event',
                               risk_col='Risk', var_col='Var'):
    baselines = []
    for var_name, group in df.groupby(var_col):
        g = group.copy()
        g['Risk_exp'] = np.exp(g[risk_col].astype(float))
        g = g.sort_values(by=time_col).reset_index(drop=True)
        event_times = np.sort(g.loc[g[event_col] == 1, time_col].unique())

        cumhaz_list, d_list, denom_list, incr_list = [], [], [], []
        Lambda = 0.0

        for t in event_times:
            d_j = int(((g[time_col] == t) & (g[event_col] == 1)).sum())
            denom = g.loc[g[time_col] >= t, 'Risk_exp'].sum()
            increment = 0.0 if denom == 0 else d_j / denom
            Lambda += increment

            d_list.append(d_j)
            denom_list.append(denom)
            incr_list.append(increment)
            cumhaz_list.append(Lambda)

        baseline = pd.DataFrame({
            var_col: var_name,
            'Time': event_times,
            'd_j': d_list,
            'Denom': denom_list,
            'Increment': incr_list,
            'Cumhaz': cumhaz_list
        })

        baselines.append(baseline)
    baseline_df = pd.concat(baselines, ignore_index=True)

    return baseline_df