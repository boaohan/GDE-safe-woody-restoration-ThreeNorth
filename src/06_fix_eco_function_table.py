"""
06_fix_eco_function_table.py

Checks and harmonizes EcoFunction variables before multi-objective optimization.

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
HYDRO_FIXED_CSV = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024.csv"
ECO_OLD_CSV = DATA_DIR / "ThreeNorth_Class_EcoFunction.csv"

# 输出
OUT_ECO_FIXED = DATA_DIR / "ThreeNorth_Class_EcoFunction.csv"
OUT_ECO_BACKUP = DATA_DIR / "ThreeNorth_Class_EcoFunction_raw_backup.csv"
OUT_QA = DATA_DIR / "ThreeNorth_Class_EcoFunction_fixQA.txt"

OVERWRITE_OUTPUT = True


# ============================================================
# 1. 工具函数
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    d = denom.copy().astype(float)
    d = d.where(np.abs(d) > 1e-12, np.nan)
    return numer.astype(float) / d


# ============================================================
# 2. 主程序
# ============================================================

def main():
    require_file(CLASSYEAR_CSV)
    require_file(HYDRO_FIXED_CSV)
    require_file(ECO_OLD_CSV)

    classyear = pd.read_csv(CLASSYEAR_CSV, encoding="utf-8-sig")
    hydro = pd.read_csv(HYDRO_FIXED_CSV, encoding="utf-8-sig")
    eco_old = pd.read_csv(ECO_OLD_CSV, encoding="utf-8-sig")

    # 备份旧版 EcoFunction
    if OVERWRITE_OUTPUT and ECO_OLD_CSV.exists():
        eco_old.to_csv(OUT_ECO_BACKUP, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2.1 基本列检查
    # --------------------------------------------------------
    need_classyear = [
        "ClassCode", "Year", "NDVI_gs", "FVC", "NPP", "AET",
        "WUE", "BareSoilFrac", "SandConnectivity",
        "Wind10_spring", "WindErosionRisk"
    ]
    missing_cy = [c for c in need_classyear if c not in classyear.columns]
    if missing_cy:
        raise RuntimeError(f"ClassYear 缺少必要列: {missing_cy}")

    need_hydro = [
        "ClassCode", "AET_scale_factor", "BaseZone", "GDE_Level", "Pixel_n", "Area_km2"
    ]
    missing_h = [c for c in need_hydro if c not in hydro.columns]
    if missing_h:
        raise RuntimeError(f"HydroSupport 缺少必要列: {missing_h}")

    # --------------------------------------------------------
    # 2.2 合并 AET_scale_factor
    # --------------------------------------------------------
    work = classyear.merge(
        hydro[["ClassCode", "AET_scale_factor"]],
        on="ClassCode",
        how="left"
    )

    if work["AET_scale_factor"].isna().any():
        missing_n = int(work["AET_scale_factor"].isna().sum())
        raise RuntimeError(f"AET_scale_factor 合并后仍有缺失: {missing_n} 行")

    # --------------------------------------------------------
    # 2.3 修正 AET，并重算 WUE
    # --------------------------------------------------------
    work["AET_raw"] = pd.to_numeric(work["AET"], errors="coerce")
    work["AET_corr"] = work["AET_raw"] / pd.to_numeric(work["AET_scale_factor"], errors="coerce")

    # 旧版 WUE 作为诊断
    work["WUE_raw"] = pd.to_numeric(work["WUE"], errors="coerce")

    # 修正版 WUE
    work["WUE_corr"] = safe_div(
        pd.to_numeric(work["NPP"], errors="coerce"),
        work["AET_corr"]
    )

    # --------------------------------------------------------
    # 2.4 按 ClassCode 聚合新版 EcoFunction
    # --------------------------------------------------------
    eco_new = (
        work.groupby("ClassCode", as_index=False)
        .agg({
            "NDVI_gs": ["mean", "std"],
            "FVC": ["mean", "std"],
            "NPP": ["mean", "std"],
            "WUE_corr": ["mean", "std"],
            "WUE_raw": ["mean", "std"],
            "BareSoilFrac": ["mean", "std"],
            "SandConnectivity": ["mean", "std"],
            "Wind10_spring": ["mean", "std"],
            "WindErosionRisk": ["mean", "std"],
        })
    )

    eco_new.columns = [
        "ClassCode",
        "NDVI_mean", "NDVI_std",
        "FVC_mean", "FVC_std",
        "NPP_mean", "NPP_std",
        "WUE_mean", "WUE_std",
        "WUE_raw_mean", "WUE_raw_std",
        "BareSoilFrac_mean", "BareSoilFrac_std",
        "SandConnectivity_mean", "SandConnectivity_std",
        "Wind10_spring_mean", "Wind10_spring_std",
        "WindErosion_mean", "WindErosion_std",
    ]

    # --------------------------------------------------------
    # 2.5 合并 class 背景
    # --------------------------------------------------------
    eco_new = eco_new.merge(
        hydro[["ClassCode", "Pixel_n", "Area_km2", "BaseZone", "GDE_Level"]],
        on="ClassCode",
        how="left"
    )

    eco_new["BaseZone"] = np.rint(eco_new["BaseZone"]).astype("Int64")
    eco_new["GDE_Level"] = np.rint(eco_new["GDE_Level"]).astype("Int64")

    # 列顺序
    final_cols = [
        "ClassCode", "Pixel_n", "Area_km2", "BaseZone", "GDE_Level",
        "NDVI_mean", "NDVI_std",
        "FVC_mean", "FVC_std",
        "NPP_mean", "NPP_std",
        "WUE_mean", "WUE_std",
        "BareSoilFrac_mean", "BareSoilFrac_std",
        "SandConnectivity_mean", "SandConnectivity_std",
        "Wind10_spring_mean", "Wind10_spring_std",
        "WindErosion_mean", "WindErosion_std",

        # 诊断列
        "WUE_raw_mean", "WUE_raw_std",
    ]
    eco_new = eco_new[final_cols].sort_values("ClassCode").reset_index(drop=True)

    # --------------------------------------------------------
    # 2.6 写出
    # --------------------------------------------------------
    out_path = OUT_ECO_FIXED if OVERWRITE_OUTPUT else DATA_DIR / "ThreeNorth_Class_EcoFunction_fixed.csv"
    eco_new.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2.7 QA
    # --------------------------------------------------------
    qa_lines = []
    qa_lines.append("========== EcoFunction Fix QA ==========")
    qa_lines.append(f"ClassYear source: {CLASSYEAR_CSV}")
    qa_lines.append(f"Hydro source: {HYDRO_FIXED_CSV}")
    qa_lines.append(f"Old Eco source: {ECO_OLD_CSV}")
    qa_lines.append(f"Output: {out_path}")
    qa_lines.append("")
    qa_lines.append(f"Rows: {len(eco_new)}")
    qa_lines.append("")
    qa_lines.append("WUE summary:")
    qa_lines.append(f"WUE_raw_mean_median: {float(eco_new['WUE_raw_mean'].median(skipna=True)):.8f}")
    qa_lines.append(f"WUE_new_mean_median: {float(eco_new['WUE_mean'].median(skipna=True)):.8f}")
    qa_lines.append("")
    qa_lines.append("Missing counts after fix:")
    for c in final_cols:
        qa_lines.append(f"{c}: {int(eco_new[c].isna().sum())}")

    OUT_QA.write_text("\n".join(qa_lines), encoding="utf-8")

    print("完成。")
    print(f"输出文件: {out_path}")
    print(f"备份文件: {OUT_ECO_BACKUP}")
    print(f"QA 文件: {OUT_QA}")


if __name__ == "__main__":
    main()
