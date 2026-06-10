import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_ATE():
    path = '/home/valderramanino/SYNTHIA/Development_SDG/data/A4/Causal_scores_test.csv'
    doses_path = '/home/valderramanino/SYNTHIA/Development_SDG/data/A4/Causal_doses_data.csv'
    
    df = pd.read_csv(path, index_col=False, na_values=".")
    df = df.loc[
        df["TIME"].isin([0, 240]),
        ["BID", "NEW_ID", "SUBSTUDY", "TIME", "PACC", "CHPACC"]]
    
    doses = pd.read_csv(doses_path, index_col=False, na_values=".")
    doses = doses.loc[
        doses["TIME"].isin([0]),
        ["NEW_ID", "PB"]]
    df = df.merge(doses, on="NEW_ID", how="left")
    # Before cleaning df has 1120
    # After cleaning df has 793

    bad_ids = df.loc[
      (df["TIME"] == 240) & (df["CHPACC"].isna()), "BID"]
    df = df[~df["BID"].isin(bad_ids)]

    # Pivot de PACC
    pacc_pivot = df.pivot(index="BID", columns="TIME", values="PACC")
    chpacc_240 = df.loc[df["TIME"] == 240, ["BID", "CHPACC"]]
    meta = df[["BID", "NEW_ID", "SUBSTUDY", "PB"]].drop_duplicates()

    # Merge final
    df = (
        meta
        .merge(pacc_pivot, on="BID")
        .merge(chpacc_240, on="BID")
        .rename(columns={
            0: "PACC_0",
            240: "PACC_240",
            "CHPACC": "CHPACC_240"}))
    df["DIFF_PACC"] = df["PACC_240"] - df["PACC_0"]
    df_trt = df[df.PB==0]
    df_pbo = df[df.PB==1]

    diff_trt = np.round(df_trt.DIFF_PACC, 3)
    diff_pbo = np.round(df_pbo.DIFF_PACC, 3)
    mean_trt = np.round(diff_trt.mean(), 3)
    mean_pbo = np.round(diff_pbo.mean(), 3)

    n = diff_trt.shape[0]
    Var_trt = ((mean_trt - diff_trt) ** 2).sum() / (n ** 2)
    Std_trt = np.round(np.sqrt(Var_trt), 3)
    ci_lower_trt = np.round(mean_trt - 1.96 * Std_trt, 3)
    ci_upper_trt = np.round(mean_trt + 1.96 * Std_trt, 3)

    df["Mean_Diff_TRT"] = mean_trt
    df["Mean_LB_TRT"] = ci_lower_trt
    df["Mean_UB_TRT"] = ci_upper_trt

    n = diff_pbo.shape[0]
    Var_pbo = ((mean_pbo - diff_pbo) ** 2).sum() / (n ** 2)
    Std_pbo = np.round(np.sqrt(Var_pbo), 3)
    ci_lower_pbo = np.round(mean_pbo - 1.96 * Std_pbo, 3)
    ci_upper_pbo = np.round(mean_pbo + 1.96 * Std_pbo, 3)

    df["Mean_Diff_PBO"] = mean_pbo
    df["Mean_LB_PBO"] = ci_lower_pbo
    df["Mean_UB_PBO"] = ci_upper_pbo

    df["Mean_Effect"] = mean_trt - mean_pbo
    df["Mean_LB_Effect"] = ci_lower_trt - ci_lower_pbo
    df["Mean_UB_Effect"] = ci_upper_trt - ci_upper_pbo

    print('Using PACC')
    print('Mean Data Effect:', mean_trt - mean_pbo)
    print('CI: (%.3f , %.3f)'%(ci_lower_trt - ci_lower_pbo, ci_upper_trt - ci_upper_pbo))


    diff_trt = np.round(df_trt.CHPACC_240, 3)
    diff_pbo = np.round(df_pbo.CHPACC_240, 3)
    mean_trt = np.round(diff_trt.mean(), 3)
    mean_pbo = np.round(diff_pbo.mean(), 3)

    n = diff_trt.shape[0]
    Var_trt = ((mean_trt - diff_trt) ** 2).sum() / (n ** 2)
    Std_trt = np.round(np.sqrt(Var_trt), 3)
    ci_lower_trt = np.round(mean_trt - 1.96 * Std_trt, 3)
    ci_upper_trt = np.round(mean_trt + 1.96 * Std_trt, 3)

    df["Mean_Diff_TRT"] = mean_trt
    df["Mean_LB_TRT"] = ci_lower_trt
    df["Mean_UB_TRT"] = ci_upper_trt

    n = diff_pbo.shape[0]
    Var_pbo = ((mean_pbo - diff_pbo) ** 2).sum() / (n ** 2)
    Std_pbo = np.round(np.sqrt(Var_pbo), 3)
    ci_lower_pbo = np.round(mean_pbo - 1.96 * Std_pbo, 3)
    ci_upper_pbo = np.round(mean_pbo + 1.96 * Std_pbo, 3)

    df["Mean_Diff_PBO"] = mean_pbo
    df["Mean_LB_PBO"] = ci_lower_pbo
    df["Mean_UB_PBO"] = ci_upper_pbo

    df["Mean_Effect"] = mean_trt - mean_pbo
    df["Mean_LB_Effect"] = ci_lower_trt - ci_lower_pbo
    df["Mean_UB_Effect"] = ci_upper_trt - ci_upper_pbo


    print('Using CHPACC')
    print('Mean Data Effect:', mean_trt - mean_pbo)
    print('CI: (%.3f , %.3f)'%(ci_lower_trt - ci_lower_pbo, ci_upper_trt - ci_upper_pbo))

    return df

if __name__ == '__main__':

    df = get_ATE()
