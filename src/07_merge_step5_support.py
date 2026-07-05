"""
07_merge_step5_support.py

Merges candidate, LAI safety, hydro-support, eco-function, and observed-structure tables into the Step 5 support table.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径区：务必使用修正版 (1)
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

CANDIDATE_CSV = DATA_DIR / "ThreeNorth_Class_CandidateLibrary.csv"
SAFE_CSV = DATA_DIR / "ThreeNorth_Class_LAI_safe_max.csv"
CLASS_SUPPORT_CSV = DATA_DIR / "ThreeNorth_Class_LAImaxTotal_q25.csv"

# 正式版：使用修正版
HYDRO_SUPPORT_CSV = DATA_DIR / "ThreeNorth_Class_HydroSupport_2005_2024.csv"
ECO_FUNCTION_CSV = DATA_DIR / "ThreeNorth_Class_EcoFunction.csv"

# ObservedStructure 先不进主打分流程
OBS_SUPPORT_CSV = DATA_DIR / "ThreeNorth_Class_ObservedStructure.csv"

OUT_STEP5_SUPPORT = DATA_DIR / "ThreeNorth_Class_Step5Support_RealMetrics.csv"
OUT_QA_TXT = DATA_DIR / "ThreeNorth_Class_MOO_QA.txt"

# ============================================================
# 1. 工具函数
# ============================================================

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def merge_support(left: pd.DataFrame, right: pd.DataFrame, key: str = "ClassCode") -> pd.DataFrame:
    if right is None or right.empty:
        return left
    dup_cols = [c for c in right.columns if c in left.columns and c != key]
    if dup_cols:
        right = right.drop(columns=dup_cols)
    return left.merge(right, on=key, how="left")

def to_numeric_if_exists(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def require_detected(name: str, value: str | None) -> None:
    if value is None:
        raise RuntimeError(f"Step5 缺少关键字段: {name}")

def fill_small_missing(df: pd.DataFrame, col: str, fallback: float | str | None = None):
    if col not in df.columns:
        return
    if fallback is None:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(method="ffill")
    elif isinstance(fallback, str):
        df[col] = df[col].fillna(df[fallback])
    else:
        df[col] = df[col].fillna(fallback)


# ============================================================
# 2. 读取
# ============================================================

cand = pd.read_csv(CANDIDATE_CSV, encoding="utf-8-sig")
safe_df = pd.read_csv(SAFE_CSV, encoding="utf-8-sig")
class_support_df = pd.read_csv(CLASS_SUPPORT_CSV, encoding="utf-8-sig")
hydro_df = pd.read_csv(HYDRO_SUPPORT_CSV, encoding="utf-8-sig")
eco_df = pd.read_csv(ECO_FUNCTION_CSV, encoding="utf-8-sig")

print("候选库:", cand.shape)
print("安全上限表:", safe_df.shape)
print("承载力上限表:", class_support_df.shape)
print("HydroSupport:", hydro_df.shape)
print("EcoFunction:", eco_df.shape)

# 可选：读 ObservedStructure，但本轮不进主排序
obs_df = None
if OBS_SUPPORT_CSV.exists():
    obs_df = pd.read_csv(OBS_SUPPORT_CSV, encoding="utf-8-sig")
    print("ObservedStructure(仅备用):", obs_df.shape)


# ============================================================
# 3. Hydro / Eco 清洗
# ============================================================

# ---- HydroSupport：按修正版逻辑应已完成修正，但再做一次保险填补 ----
fill_small_missing(hydro_df, "SW_occurrence_mean", 0.0)
fill_small_missing(hydro_df, "GDE_trajectory_code_mean", -1.0)
fill_small_missing(hydro_df, "LAI_nat_mean_period", "LAI_mean_period")
fill_small_missing(hydro_df, "LAI_nat_max_period", "LAI_max_period")

# 若 AI_full 不在列里，则回退构造
if "AI_full" not in hydro_df.columns and "AI_with_GWSA_proxy_mean" in hydro_df.columns:
    hydro_df["AI_full"] = hydro_df["AI_with_GWSA_proxy_mean"]

# SW_access 当前不进入模型
if "SW_access_use_flag" in hydro_df.columns:
    use_sw_access = int(pd.to_numeric(hydro_df["SW_access_use_flag"], errors="coerce").fillna(0).max()) == 1
else:
    use_sw_access = False

# ---- EcoFunction：修正版 WUE 已完成重算 ----
# 这里只做数值化与缺失保护
num_cols_eco = [c for c in eco_df.columns if c not in ["ClassCode", "BaseZone", "GDE_Level"]]
to_numeric_if_exists(eco_df, num_cols_eco)

for c in [
    "NDVI_mean", "NDVI_std",
    "FVC_mean", "FVC_std",
    "NPP_mean", "NPP_std",
    "WUE_mean", "WUE_std",
    "BareSoilFrac_mean", "BareSoilFrac_std",
    "SandConnectivity_mean", "SandConnectivity_std",
    "Wind10_spring_mean", "Wind10_spring_std",
    "WindErosion_mean", "WindErosion_std"
]:
    fill_small_missing(eco_df, c, None)


# ============================================================
# 4. 合并到 Step5 工作表
# ============================================================

work = cand.copy()
work = merge_support(work, safe_df, key="ClassCode")
work = merge_support(work, class_support_df, key="ClassCode")
work = merge_support(work, hydro_df, key="ClassCode")
work = merge_support(work, eco_df, key="ClassCode")

# 数值列统一转 numeric
possible_num_cols = [
    "ClassCode", "GDE_Level", "R_gde", "LAI_safe_max", "LAI_max_total", "LAI_current_3yr_mean",
    "target_LAI", "scheme_LAI_capacity", "LAI_margin",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",

    "P_mean", "PET_mean", "AET_mean", "Runoff_mean", "Soil_mean", "Tmean_mean",
    "GWSA_mean_period_mean", "GWSA_trend_mean",
    "AI_full", "AI_prelim_noGW_mean", "AI_with_GWSA_proxy_mean",

    "Elevation_mean", "Slope_mean", "GDE_frac_mean", "GDE_stability_mean", "GDE_persistence_count_mean",
    "NDVI_mean", "NDVI_std", "FVC_mean", "FVC_std",
    "NPP_mean", "NPP_std", "WUE_mean", "WUE_std",
    "WindErosion_mean", "BareSoilFrac_mean", "SandConnectivity_mean",
]
to_numeric_if_exists(work, possible_num_cols)


# ============================================================
# 5. 字段识别：正式版真实指标
# ============================================================

P_COL = first_existing(work, ["P_mean"])
PET_COL = first_existing(work, ["PET_mean"])
AET_COL = first_existing(work, ["AET_mean"])
SOIL_COL = first_existing(work, ["Soil_mean"])
RUNOFF_COL = first_existing(work, ["Runoff_mean"])
AI_COL = first_existing(work, ["AI_full", "AI_with_GWSA_proxy_mean", "AI_prelim_noGW_mean"])
GWSA_TREND_COL = first_existing(work, ["GWSA_trend_mean"])
LAI_CUR_COL = first_existing(work, ["LAI_current_3yr_mean"])

NDVI_MEAN_COL = first_existing(work, ["NDVI_mean"])
NDVI_STD_COL = first_existing(work, ["NDVI_std"])
NPP_MEAN_COL = first_existing(work, ["NPP_mean"])
NPP_STD_COL = first_existing(work, ["NPP_std"])
WUE_MEAN_COL = first_existing(work, ["WUE_mean"])
WUE_STD_COL = first_existing(work, ["WUE_std"])

WIND_EROSION_COL = first_existing(work, ["WindErosion_mean"])
BARE_SOIL_COL = first_existing(work, ["BareSoilFrac_mean"])
SAND_CONN_COL = first_existing(work, ["SandConnectivity_mean"])

# 强制存在
require_detected("P_COL", P_COL)
require_detected("PET_COL", PET_COL)
require_detected("AET_COL", AET_COL)
require_detected("SOIL_COL", SOIL_COL)
require_detected("RUNOFF_COL", RUNOFF_COL)
require_detected("AI_COL", AI_COL)
require_detected("GWSA_TREND_COL", GWSA_TREND_COL)
require_detected("LAI_CUR_COL", LAI_CUR_COL)

require_detected("NDVI_MEAN_COL", NDVI_MEAN_COL)
require_detected("NDVI_STD_COL", NDVI_STD_COL)
require_detected("NPP_MEAN_COL", NPP_MEAN_COL)
require_detected("NPP_STD_COL", NPP_STD_COL)
require_detected("WUE_MEAN_COL", WUE_MEAN_COL)
require_detected("WUE_STD_COL", WUE_STD_COL)

require_detected("WIND_EROSION_COL", WIND_EROSION_COL)
require_detected("BARE_SOIL_COL", BARE_SOIL_COL)
require_detected("SAND_CONN_COL", SAND_CONN_COL)

# SW_access 当前禁用
SW_ACCESS_COL = None
if use_sw_access and "SW_access_mean" in work.columns:
    # 若未来修好，可启用
    if pd.to_numeric(work["SW_access_mean"], errors="coerce").dropna().nunique() > 1:
        SW_ACCESS_COL = "SW_access_mean"


# ============================================================
# 6. 输出 Step5 支持表 + QA
# ============================================================

work.to_csv(OUT_STEP5_SUPPORT, index=False, encoding="utf-8-sig")

qa_lines = []
qa_lines.append("========== Step5 QA ==========")
qa_lines.append("HydroSupport = 修正版 (1)")
qa_lines.append("EcoFunction = 修正版 (1)")
qa_lines.append("")
qa_lines.append("真实稳定性输入是否齐全: True")
qa_lines.append("真实固沙输入是否存在: True")
qa_lines.append("")
qa_lines.append(f"P_COL={P_COL}")
qa_lines.append(f"PET_COL={PET_COL}")
qa_lines.append(f"AET_COL={AET_COL}")
qa_lines.append(f"SOIL_COL={SOIL_COL}")
qa_lines.append(f"RUNOFF_COL={RUNOFF_COL}")
qa_lines.append(f"AI_COL={AI_COL}")
qa_lines.append(f"GWSA_TREND_COL={GWSA_TREND_COL}")
qa_lines.append(f"LAI_CUR_COL={LAI_CUR_COL}")
qa_lines.append(f"NDVI_MEAN_COL={NDVI_MEAN_COL}")
qa_lines.append(f"NDVI_STD_COL={NDVI_STD_COL}")
qa_lines.append(f"NPP_MEAN_COL={NPP_MEAN_COL}")
qa_lines.append(f"NPP_STD_COL={NPP_STD_COL}")
qa_lines.append(f"WUE_MEAN_COL={WUE_MEAN_COL}")
qa_lines.append(f"WUE_STD_COL={WUE_STD_COL}")
qa_lines.append(f"WIND_EROSION_COL={WIND_EROSION_COL}")
qa_lines.append(f"BARE_SOIL_COL={BARE_SOIL_COL}")
qa_lines.append(f"SAND_CONN_COL={SAND_CONN_COL}")
qa_lines.append(f"SW_ACCESS_COL={SW_ACCESS_COL}")
qa_lines.append("")
qa_lines.append(f"Step5Support rows={len(work)}")

OUT_QA_TXT.write_text("\n".join(qa_lines), encoding="utf-8")

print(f"已输出 Step5 支持表: {OUT_STEP5_SUPPORT}")
print(f"已输出 QA: {OUT_QA_TXT}")
print("\n".join(qa_lines))
