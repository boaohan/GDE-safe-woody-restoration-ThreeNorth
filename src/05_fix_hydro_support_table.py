"""
05_fix_hydro_support_table.py

Checks and harmonizes HydroSupport variables, including PET/AET scale diagnostics and AI/WUE recalculation.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 0. 参数区
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

CLASSYEAR_CSV = DATA_DIR / "ThreeNorth_ClassYear_FullMetrics_2005_2024.csv"
HYDRO_OLD_CSV = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024.csv"

# 输出
OUT_HYDRO_FIXED = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024.csv"
OUT_HYDRO_BACKUP = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024_raw_backup.csv"
OUT_QA = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024_fixQA.txt"

# 自动修正 PET / AET 尺度
AUTO_FIX_ET_SCALE = True

# 阈值：根据你当前结果设置
PET_SCALE_THRESHOLD = 3000.0
AET_SCALE_THRESHOLD = 1000.0

# GWSA proxy 的 unitScale(-50,50)
GW_PROXY_MIN = -50.0
GW_PROXY_MAX = 50.0

# 是否覆盖旧 HydroSupport
OVERWRITE_OUTPUT = True


# ============================================================
# 1. 工具函数
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")


def clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    d = denom.copy().astype(float)
    d = d.where(np.abs(d) > 1e-12, np.nan)
    return numer.astype(float) / d


# ============================================================
# 2. 主程序
# ============================================================

def main():
    require_file(CLASSYEAR_CSV)
    require_file(HYDRO_OLD_CSV)

    classyear = pd.read_csv(CLASSYEAR_CSV, encoding="utf-8-sig")
    hydro_old = pd.read_csv(HYDRO_OLD_CSV, encoding="utf-8-sig")

    # 备份旧版 HydroSupport
    if OVERWRITE_OUTPUT and HYDRO_OLD_CSV.exists():
        hydro_old.to_csv(OUT_HYDRO_BACKUP, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2.1 从 ClassYear 重新聚合年度动态变量
    # --------------------------------------------------------
    needed_cols = [
        "ClassCode", "P", "PET", "AET", "Runoff", "Soil", "Tmean",
        "AI_prelim_noGW", "AI_with_GWSA_proxy", "GWSA_mean_period",
        "LAI_mean", "LAI_max", "LAI_nat_mean", "LAI_nat_max"
    ]
    missing_cols = [c for c in needed_cols if c not in classyear.columns]
    if missing_cols:
        raise RuntimeError(f"ClassYear 缺少必要列: {missing_cols}")

    dyn = (
        classyear.groupby("ClassCode", as_index=False)
        .agg({
            "P": "mean",
            "PET": "mean",
            "AET": "mean",
            "Runoff": "mean",
            "Soil": "mean",
            "Tmean": "mean",
            "AI_prelim_noGW": "mean",
            "AI_with_GWSA_proxy": "mean",
            "GWSA_mean_period": "mean",
            "LAI_mean": "mean",
            "LAI_max": "mean",
            "LAI_nat_mean": "mean",
            "LAI_nat_max": "mean",
        })
        .rename(columns={
            "P": "P_mean",
            "PET": "PET_raw_mean",
            "AET": "AET_raw_mean",
            "Runoff": "Runoff_mean",
            "Soil": "Soil_mean",
            "Tmean": "Tmean_mean",
            "AI_prelim_noGW": "AI_prelim_noGW_raw_mean",
            "AI_with_GWSA_proxy": "AI_with_GWSA_proxy_raw_mean",
            "GWSA_mean_period": "GWSA_mean_period_mean",
            "LAI_mean": "LAI_mean_period",
            "LAI_max": "LAI_max_period",
            "LAI_nat_mean": "LAI_nat_mean_period",
            "LAI_nat_max": "LAI_nat_max_period",
        })
    )

    # --------------------------------------------------------
    # 2.2 自动判断 PET / AET 是否需要缩放
    # --------------------------------------------------------
    pet_scale_factor = 1.0
    aet_scale_factor = 1.0

    pet_median = float(dyn["PET_raw_mean"].median(skipna=True))
    aet_median = float(dyn["AET_raw_mean"].median(skipna=True))

    if AUTO_FIX_ET_SCALE:
        if pet_median > PET_SCALE_THRESHOLD:
            pet_scale_factor = 10.0
        if aet_median > AET_SCALE_THRESHOLD:
            aet_scale_factor = 10.0

    dyn["PET_scale_factor"] = pet_scale_factor
    dyn["AET_scale_factor"] = aet_scale_factor

    dyn["PET_mean"] = dyn["PET_raw_mean"] / dyn["PET_scale_factor"]
    dyn["AET_mean"] = dyn["AET_raw_mean"] / dyn["AET_scale_factor"]

    # --------------------------------------------------------
    # 2.3 重算 AI
    # --------------------------------------------------------
    # 近似重建 GWSA proxy：对应 GEE 的 unitScale(-50,50) 再截断到 [0,1]
    dyn["GW_proxy_mean"] = clip01(
        (dyn["GWSA_mean_period_mean"] - GW_PROXY_MIN) / (GW_PROXY_MAX - GW_PROXY_MIN)
    )

    dyn["AI_prelim_noGW_mean"] = safe_div(
        dyn["P_mean"] + dyn["Soil_mean"] + dyn["Runoff_mean"],
        dyn["PET_mean"]
    )

    dyn["AI_with_GWSA_proxy_mean"] = safe_div(
        dyn["P_mean"] + dyn["Soil_mean"] + dyn["Runoff_mean"] + dyn["GW_proxy_mean"],
        dyn["PET_mean"]
    )

    # 正式给 Step5 使用的 AI 列
    dyn["AI_full"] = dyn["AI_with_GWSA_proxy_mean"]

    # --------------------------------------------------------
    # 2.4 从旧 HydroSupport 中提取静态背景列
    # --------------------------------------------------------
    static_cols = [
        "ClassCode", "Pixel_n", "Area_km2", "BaseZone", "GDE_Level",
        "GDE_frac_mean", "GDE_stability_mean", "GDE_persistence_count_mean",
        "GDE_trajectory_code_mean", "GWSA_trend_mean",
        "Elevation_mean", "Slope_mean",
        "SW_occurrence_mean", "SW_access_mean",
        "NaturalFrac_mean", "NaturalMask_1km_mean",
        "LAI_current_3yr_mean"
    ]
    missing_static = [c for c in static_cols if c not in hydro_old.columns]
    if missing_static:
        raise RuntimeError(f"旧 HydroSupport 缺少静态背景列: {missing_static}")

    static_df = hydro_old[static_cols].drop_duplicates(subset=["ClassCode"]).copy()

    # --------------------------------------------------------
    # 2.5 修 SW_access / 缺失
    # --------------------------------------------------------
    static_df["SW_access_raw_mean"] = static_df["SW_access_mean"]

    sw_access_unique = static_df["SW_access_mean"].dropna().round(10).nunique()
    if sw_access_unique <= 1:
        static_df["SW_access_use_flag"] = 0
        static_df["SW_access_mean"] = np.nan
    else:
        static_df["SW_access_use_flag"] = 1

    static_df["SW_occurrence_mean"] = static_df["SW_occurrence_mean"].fillna(0.0)
    static_df["GDE_trajectory_code_mean"] = static_df["GDE_trajectory_code_mean"].fillna(-1.0)

    # --------------------------------------------------------
    # 2.6 合并动态与静态
    # --------------------------------------------------------
    hydro_new = static_df.merge(dyn, on="ClassCode", how="left")

    # BaseZone / GDE_Level 修正成整数
    hydro_new["BaseZone"] = np.rint(hydro_new["BaseZone"]).astype("Int64")
    hydro_new["GDE_Level"] = np.rint(hydro_new["GDE_Level"]).astype("Int64")

    # LAI_nat 缺失填补
    hydro_new["LAI_nat_mean_period"] = hydro_new["LAI_nat_mean_period"].fillna(hydro_new["LAI_mean_period"])
    hydro_new["LAI_nat_max_period"] = hydro_new["LAI_nat_max_period"].fillna(hydro_new["LAI_max_period"])

    # --------------------------------------------------------
    # 2.7 输出列顺序
    # --------------------------------------------------------
    final_cols = [
        "ClassCode", "Pixel_n", "Area_km2", "BaseZone", "GDE_Level",

        # 正式建议给 Step5 用的列
        "P_mean", "PET_mean", "AET_mean", "Runoff_mean", "Soil_mean", "Tmean_mean",
        "AI_prelim_noGW_mean", "AI_with_GWSA_proxy_mean", "AI_full",
        "GWSA_mean_period_mean",
        "LAI_mean_period", "LAI_max_period", "LAI_nat_mean_period", "LAI_nat_max_period",

        # 静态背景
        "GDE_frac_mean", "GDE_stability_mean", "GDE_persistence_count_mean",
        "GDE_trajectory_code_mean", "GWSA_trend_mean",
        "Elevation_mean", "Slope_mean",
        "SW_occurrence_mean", "SW_access_mean", "SW_access_use_flag",
        "NaturalFrac_mean", "NaturalMask_1km_mean",
        "LAI_current_3yr_mean",

        # 诊断列：保留原值与缩放信息
        "PET_raw_mean", "AET_raw_mean",
        "PET_scale_factor", "AET_scale_factor",
        "AI_prelim_noGW_raw_mean", "AI_with_GWSA_proxy_raw_mean",
        "GW_proxy_mean", "SW_access_raw_mean",
    ]

    hydro_new = hydro_new[final_cols].sort_values("ClassCode").reset_index(drop=True)

    # --------------------------------------------------------
    # 2.8 写出
    # --------------------------------------------------------
    out_path = OUT_HYDRO_FIXED if OVERWRITE_OUTPUT else DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024_fixed.csv"
    hydro_new.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2.9 QA 文本
    # --------------------------------------------------------
    qa_lines = []
    qa_lines.append("========== HydroSupport Fix QA ==========")
    qa_lines.append(f"ClassYear source: {CLASSYEAR_CSV}")
    qa_lines.append(f"Old Hydro source: {HYDRO_OLD_CSV}")
    qa_lines.append(f"Output: {out_path}")
    qa_lines.append("")
    qa_lines.append(f"Rows: {len(hydro_new)}")
    qa_lines.append(f"PET_raw_median: {pet_median:.4f}")
    qa_lines.append(f"AET_raw_median: {aet_median:.4f}")
    qa_lines.append(f"PET_scale_factor: {pet_scale_factor}")
    qa_lines.append(f"AET_scale_factor: {aet_scale_factor}")
    qa_lines.append(f"SW_access_unique_non_na: {sw_access_unique}")
    qa_lines.append("")
    qa_lines.append("Missing counts after fix:")
    for c in final_cols:
        qa_lines.append(f"{c}: {int(hydro_new[c].isna().sum())}")

    OUT_QA.write_text("\n".join(qa_lines), encoding="utf-8")

    print("完成。")
    print(f"输出文件: {out_path}")
    print(f"备份文件: {OUT_HYDRO_BACKUP}")
    print(f"QA 文件: {OUT_QA}")


if __name__ == "__main__":
    main()
