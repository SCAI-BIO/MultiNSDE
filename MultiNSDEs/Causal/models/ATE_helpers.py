import numpy as np
import pandas as pd
import torch


def _torch_breslow_baseline_from_risk(log_risk, durations, events, horizon=None):
    log_risk = torch.as_tensor(log_risk, dtype=torch.float32).reshape(-1)
    durations = torch.as_tensor(durations, dtype=torch.float32).reshape(-1)
    events = torch.as_tensor(events, dtype=torch.float32).reshape(-1)

    if horizon is None:
        horizon = float(durations.max().item()) if durations.numel() > 0 else 0.0
    else:
        horizon = float(horizon)

    event_mask = (events > 0.5) & (durations <= horizon)
    event_times = torch.unique(durations[event_mask], sorted=True)
    if event_times.numel() == 0:
        return durations.new_empty((0,)), durations.new_empty((0,))

    exp_risk = torch.exp(log_risk)
    cumhaz = []
    cumulative = log_risk.new_tensor(0.0)
    for event_time in event_times:
        d_j = (((durations == event_time) & (events > 0.5)).float()).sum()
        denom = exp_risk[durations >= event_time].sum()
        increment = d_j / denom if float(denom.item()) > 0 else log_risk.new_tensor(0.0)
        cumulative = cumulative + increment
        cumhaz.append(cumulative)

    return event_times, torch.stack(cumhaz)


def _torch_stepwise_rmst_from_risk(log_risk, baseline_times, baseline_cumhaz, horizon):
    log_risk = torch.as_tensor(log_risk, dtype=torch.float32)
    if log_risk.ndim == 0:
        log_risk = log_risk.unsqueeze(0)

    baseline_times = torch.as_tensor(
        baseline_times, dtype=log_risk.dtype, device=log_risk.device
    )
    baseline_cumhaz = torch.as_tensor(
        baseline_cumhaz, dtype=log_risk.dtype, device=log_risk.device
    )
    horizon = float(horizon)

    if horizon <= 0:
        return torch.zeros_like(log_risk)

    keep = baseline_times <= horizon
    event_times = baseline_times[keep]
    event_cumhaz = baseline_cumhaz[keep]

    grid = torch.cat((
        log_risk.new_tensor([0.0]),
        event_times,
        log_risk.new_tensor([horizon]),
    ))
    interval_lengths = grid[1:] - grid[:-1]
    interval_cumhaz = torch.cat((log_risk.new_tensor([0.0]), event_cumhaz))
    surv = torch.exp(-torch.exp(log_risk).unsqueeze(-1) * interval_cumhaz.unsqueeze(0))
    return torch.sum(surv * interval_lengths.unsqueeze(0), dim=1)


def get_batch_rmst_or_components(log_risk, durations, events, horizon=None):
    log_risk = torch.as_tensor(log_risk, dtype=torch.float32)
    durations = torch.as_tensor(durations, dtype=torch.float32, device=log_risk.device)
    events = torch.as_tensor(events, dtype=torch.float32, device=log_risk.device)

    squeeze_output = log_risk.ndim == 1
    if squeeze_output:
        log_risk = log_risk.unsqueeze(-1)
        durations = durations.unsqueeze(-1)
        events = events.unsqueeze(-1)

    observed_rmst = []
    predicted_rmst = []
    observed_mask = []
    for te_idx in range(log_risk.shape[1]):
        te_horizon = (
            float(horizon)
            if horizon is not None
            else float(durations[:, te_idx].max().item())
        )

        te_duration = durations[:, te_idx]
        te_event = events[:, te_idx]
        baseline_times, baseline_cumhaz = _torch_breslow_baseline_from_risk(
            log_risk[:, te_idx].detach(),
            te_duration.detach(),
            te_event.detach(),
            horizon=te_horizon,
        )

        observed_rmst.append(torch.minimum(te_duration, te_duration.new_full(te_duration.shape, te_horizon)))
        observed_mask.append(
            (((te_event > 0.5) & (te_duration <= te_horizon)) | (te_duration >= te_horizon)).float()
        )
        predicted_rmst.append(
            _torch_stepwise_rmst_from_risk(
                log_risk[:, te_idx], baseline_times, baseline_cumhaz, te_horizon
            )
        )

    observed_rmst = torch.stack(observed_rmst, dim=1)
    predicted_rmst = torch.stack(predicted_rmst, dim=1)
    observed_mask = torch.stack(observed_mask, dim=1)

    if squeeze_output:
        return observed_rmst[:, 0], predicted_rmst[:, 0], observed_mask[:, 0]
    return observed_rmst, predicted_rmst, observed_mask


def _stepwise_rmst_from_risk(log_risk, baseline_times, baseline_cumhaz, horizon):
    baseline_times = np.asarray(baseline_times, dtype=float)
    baseline_cumhaz = np.asarray(baseline_cumhaz, dtype=float)
    horizon = float(horizon)

    if horizon <= 0:
        return np.zeros_like(np.asarray(log_risk, dtype=float))

    keep = baseline_times <= horizon
    event_times = baseline_times[keep]
    event_cumhaz = baseline_cumhaz[keep]

    grid = np.concatenate(([0.0], event_times, [horizon]))
    interval_lengths = np.diff(grid)
    interval_cumhaz = np.concatenate(([0.0], event_cumhaz))
    surv = np.exp(-np.exp(np.asarray(log_risk, dtype=float))[:, None] * interval_cumhaz[None, :])
    return np.sum(surv * interval_lengths[None, :], axis=1)


def get_survival_tau_datatop(risk_df, baseline_df, horizon=None):
    risk_df = risk_df.copy()
    risk_df['Arm'] = pd.to_numeric(risk_df['Arm'], errors='coerce')
    risk_df['REPI'] = pd.to_numeric(risk_df['REPI'], errors='coerce')
    risk_df = risk_df[(risk_df['REPI'] == 0) & (risk_df['Arm'].isin([0, 1, 2, 3]))]

    patient_rows = []
    for var_name, risk_var in risk_df.groupby('Var'):
        baseline_var = baseline_df[baseline_df['Var'] == var_name].sort_values('Time')
        if baseline_var.empty:
            continue

        horizon_var = float(horizon) if horizon is not None else float(baseline_var['Time'].max())
        base_times = baseline_var['Time'].to_numpy(dtype=float)
        base_cumhaz = baseline_var['Cumhaz'].to_numpy(dtype=float)

        arm0 = risk_var[risk_var['Arm'] == 0][['PTNO', 'Risk']].rename(columns={'Risk': 'Risk_PBO'})
        if arm0.empty:
            continue

        for arm in [1, 2, 3]:
            arm_df = risk_var[risk_var['Arm'] == arm][['PTNO', 'Risk']].rename(columns={'Risk': 'Risk_ARM'})
            if arm_df.empty:
                continue

            merged = arm0.merge(arm_df, on='PTNO', how='inner')
            if merged.empty:
                continue

            rmst_pbo = _stepwise_rmst_from_risk(merged['Risk_PBO'].to_numpy(), base_times, base_cumhaz, horizon_var)
            rmst_arm = _stepwise_rmst_from_risk(merged['Risk_ARM'].to_numpy(), base_times, base_cumhaz, horizon_var)
            tau = rmst_arm - rmst_pbo

            out = merged[['PTNO']].copy()
            out['Variable'] = var_name
            out['TREAT_ARM'] = arm
            out['HORIZON'] = horizon_var
            out['RMST_PBO'] = np.round(rmst_pbo, 6)
            out['RMST_ARM'] = np.round(rmst_arm, 6)
            out['Tau_hat_i'] = np.round(tau, 6)
            patient_rows.append(out)

    if len(patient_rows) == 0:
        raise ValueError('Could not build DATATOP survival ATE from Risk_TE/Baseline_Risk_TE files.')

    return pd.concat(patient_rows, ignore_index=True)


def cumhaz_at_horizon(baseline_var, horizon=None):
    baseline_var = baseline_var.sort_values('Time')
    if baseline_var.empty:
        return np.nan, np.nan

    horizon_var = float(horizon) if horizon is not None else float(baseline_var['Time'].max())
    cumhaz_up_to_horizon = baseline_var.loc[
        baseline_var['Time'] <= horizon_var, 'Cumhaz'
    ]
    cumhaz_value = float(cumhaz_up_to_horizon.iloc[-1]) if len(cumhaz_up_to_horizon) > 0 else 0.0
    return horizon_var, cumhaz_value


def summarize_survival_tau(df):
    patient_tau = (
        df.groupby(['PTNO', 'Variable', 'TREAT_ARM'], as_index=False)
        [['Tau_hat_i', 'RMST_PBO', 'RMST_ARM', 'HORIZON']]
        .mean()
    )

    outputs = []
    for (var_name, arm), group in patient_tau.groupby(['Variable', 'TREAT_ARM']):
        psi_hat = float(group['Tau_hat_i'].mean())
        n = group.shape[0]
        var_psi = float(((group['Tau_hat_i'] - psi_hat) ** 2).sum() / (n ** 2)) if n > 0 else np.nan
        std_psi = float(np.sqrt(var_psi)) if n > 0 else np.nan
        ci_lower = psi_hat - 1.96 * std_psi if n > 0 else np.nan
        ci_upper = psi_hat + 1.96 * std_psi if n > 0 else np.nan

        group = group.copy()
        group['ATE_RMST'] = np.round(psi_hat, 6)
        group['ATE_STD'] = np.round(std_psi, 6)
        group['ATE_LB'] = np.round(ci_lower, 6)
        group['ATE_UB'] = np.round(ci_upper, 6)
        outputs.append(group)

        print(f'{var_name} TREAT {arm} vs placebo - ATE RMST: {psi_hat:.6f}')
        print('CI: (%.6f , %.6f)' % (ci_lower, ci_upper))

    return pd.concat(outputs, ignore_index=True)


def summarize_log_hr(log_hr_df):
    outputs = []
    group_cols = ['Variable', 'Comparison']
    if 'HORIZON' in log_hr_df.columns:
        group_cols.append('HORIZON')

    for group_key, group in log_hr_df.groupby(group_cols):
        if len(group_cols) == 3:
            var_name, comparison, horizon = group_key
        else:
            var_name, comparison = group_key
            horizon = np.nan

        log_hr_i = group['LOG_HR_i'].astype(float)
        log_hr_hat = float(log_hr_i.mean())
        n = log_hr_i.shape[0]
        var_log_hr = float(((log_hr_i - log_hr_hat) ** 2).sum() / (n ** 2)) if n > 0 else np.nan
        std_log_hr = float(np.sqrt(var_log_hr)) if n > 0 else np.nan

        hr_hat = float(np.exp(log_hr_hat)) if n > 0 else np.nan
        ci_lower = float(np.exp(log_hr_hat - 1.96 * std_log_hr)) if n > 0 else np.nan
        ci_upper = float(np.exp(log_hr_hat + 1.96 * std_log_hr)) if n > 0 else np.nan

        out = group.copy()
        if 'HORIZON' in out.columns:
            out['HORIZON'] = np.round(horizon, 6)
        out['LOG_HR'] = np.round(log_hr_hat, 6)
        out['HR'] = np.round(hr_hat, 6)
        out['HR_STD_LOG'] = np.round(std_log_hr, 6)
        out['HR_LB'] = np.round(ci_lower, 6)
        out['HR_UB'] = np.round(ci_upper, 6)
        outputs.append(out)

        if np.isnan(horizon):
            print(f'{var_name} {comparison} - HR: {hr_hat:.6f}')
        else:
            print(f'{var_name} {comparison} at horizon {horizon:.6f} - HR: {hr_hat:.6f}')
        print('CI: (%.6f , %.6f)' % (ci_lower, ci_upper))

    if len(outputs) == 0:
        raise ValueError('Could not build DATATOP hazard ratios from Risk_TE files.')

    return pd.concat(outputs, ignore_index=True)
