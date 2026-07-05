"""
09_multiobjective_optimization.py

Scores candidate configurations, constructs Pareto fronts, and selects the best compromise restoration solution.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# 0. 参数区
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"
DATA_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path(__file__).resolve().parent

# ---- 主输入：已经合并好的正式支持表 ----
STEP5_SUPPORT_CSV = DATA_DIR / "ThreeNorth_Class_Step5Support_RealMetrics.csv"
SPECIES_POOL_CSV = DATA_DIR / "ThreeNorth_SpeciesPool_Seed.csv"

# ---- 输出 ----
OUT_SCORED_CSV = DATA_DIR / "ThreeNorth_Class_MOO_ScoredCandidates.csv"
OUT_PARETO_CSV = DATA_DIR / "ThreeNorth_Class_MOO_ParetoFront.csv"
OUT_BEST_CSV = DATA_DIR / "ThreeNorth_Class_MOO_BestCompromise.csv"
OUT_RANGE_CSV = DATA_DIR / "ThreeNorth_Class_MOO_ParetoRanges_byClass.csv"
OUT_ZONE_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_MOO_Summary_byZone.csv"
OUT_QA_TXT = DATA_DIR / "ThreeNorth_Class_MOO_QA.txt"

# ---- Best compromise 权重 ----
W_SAND = 0.25
W_STABILITY = 0.25
W_WATER_SECURITY = 0.25
W_GDE_SAFETY = 0.25

# ---- 稳健标准化分位数 ----
ROBUST_Q_LOW = 0.05
ROBUST_Q_HIGH = 0.95

EPS = 1e-9


# =============================================================================
# 1. 工具函数
# =============================================================================

def robust_minmax(
    s: pd.Series,
    q_low: float = ROBUST_Q_LOW,
    q_high: float = ROBUST_Q_HIGH,
    reverse: bool = False,
) -> pd.Series:
    """稳健标准化到 0-1，使用 5%-95% 分位数截断。"""
    x = pd.to_numeric(s, errors="coerce").astype(float).copy()
    lo = x.quantile(q_low)
    hi = x.quantile(q_high)

    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        out = pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    else:
        x_clip = x.clip(lower=lo, upper=hi)
        out = (x_clip - lo) / (hi - lo)

    if reverse:
        out = 1.0 - out

    return out.fillna(0.0)


def safe_div(a: pd.Series | float, b: pd.Series | float, default: float = 0.0):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    if np.isscalar(a) and np.isscalar(b):
        return float(out)
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def split_species(text: object) -> List[str]:
    if pd.isna(text):
        return []
    s = str(text).strip()
    if not s:
        return []
    for sep in ["，", ",", ";", "；", "/", "|"]:
        s = s.replace(sep, "、")
    return [x.strip() for x in s.split("、") if x.strip()]


def shannon_evenness(ratios: Sequence[float]) -> float:
    arr = np.asarray(ratios, dtype=float)
    arr = arr[arr > EPS]
    if len(arr) <= 1:
        return 0.0
    p = arr / arr.sum()
    h = -np.sum(p * np.log(p + EPS))
    return float(h / np.log(len(p)))


def present_layers_fraction(ratios: Sequence[float]) -> float:
    arr = np.asarray(ratios, dtype=float)
    return float((arr > EPS).sum() / 3.0)


def species_trait_weighted_mean(
    species_names: Sequence[str],
    species_df: pd.DataFrame,
    trait_col: str,
    default: float,
) -> float:
    if not species_names:
        return float(default)
    sub = species_df.loc[species_df["species_name"].isin(species_names)].copy()
    if sub.empty or trait_col not in sub.columns:
        return float(default)
    vals = pd.to_numeric(sub[trait_col], errors="coerce").dropna()
    if vals.empty:
        return float(default)
    return float(vals.mean())


def weighted_scheme_trait(
    row: pd.Series,
    species_df: pd.DataFrame,
    trait_col: str,
    default: float = 0.0,
) -> float:
    t_ratio = float(row.get("tree_ratio", 0.0) or 0.0)
    s_ratio = float(row.get("shrub_ratio", 0.0) or 0.0)
    g_ratio = float(row.get("grass_ratio", 0.0) or 0.0)

    t_names = split_species(row.get("tree_species", ""))
    s_names = split_species(row.get("shrub_species", ""))
    g_names = split_species(row.get("grass_species", ""))

    t_val = species_trait_weighted_mean(t_names, species_df, trait_col, default)
    s_val = species_trait_weighted_mean(s_names, species_df, trait_col, default)
    g_val = species_trait_weighted_mean(g_names, species_df, trait_col, default)

    return float(t_ratio * t_val + s_ratio * s_val + g_ratio * g_val)


def map_gde_level(level: object) -> float:
    try:
        lv = int(level)
    except Exception:
        return 0.0
    mp = {0: 0.00, 1: 0.30, 2: 0.60, 3: 1.00}
    return float(mp.get(lv, 0.0))


def zone_structure_score(row: pd.Series) -> float:
    """不同主恢复类型下，对乔灌草结构赋不同防风固沙权重。"""
    z = str(row.get("veg_zone_primary", ""))
    t = float(row.get("tree_ratio", 0.0) or 0.0)
    s = float(row.get("shrub_ratio", 0.0) or 0.0)
    g = float(row.get("grass_ratio", 0.0) or 0.0)

    if z == "低覆盖灌草":
        score = 0.10 * t + 0.40 * s + 0.50 * g
    elif z == "灌草主导":
        score = 0.15 * t + 0.45 * s + 0.40 * g
    elif z == "低密乔灌草":
        score = 0.30 * t + 0.40 * s + 0.30 * g
    elif z == "乔灌草":
        score = 0.40 * t + 0.35 * s + 0.25 * g
    else:
        score = 0.10 * t + 0.35 * s + 0.55 * g
    return float(score)


def moderate_use_score(util: float, optimum: float = 0.75) -> float:
    """适中利用率更稳定；过低/过高都扣分。"""
    return float(max(0.0, 1.0 - abs(util - optimum) / max(optimum, EPS)))


def dominates(a: np.ndarray, b: np.ndarray, maximize: Sequence[bool]) -> bool:
    better_or_equal = []
    strictly_better = []
    for ai, bi, is_max in zip(a, b, maximize):
        if is_max:
            better_or_equal.append(ai >= bi - EPS)
            strictly_better.append(ai > bi + EPS)
        else:
            better_or_equal.append(ai <= bi + EPS)
            strictly_better.append(ai < bi - EPS)
    return all(better_or_equal) and any(strictly_better)


def pareto_ranks(group: pd.DataFrame, obj_cols: Sequence[str], maximize: Sequence[bool]) -> pd.Series:
    vals = group[list(obj_cols)].to_numpy(dtype=float)
    idx = list(range(len(group)))
    ranks = np.zeros(len(group), dtype=int)
    current_rank = 1
    remain = idx.copy()

    while remain:
        front = []
        for i in remain:
            dominated_flag = False
            for j in remain:
                if i == j:
                    continue
                if dominates(vals[j], vals[i], maximize):
                    dominated_flag = True
                    break
            if not dominated_flag:
                front.append(i)

        for i in front:
            ranks[i] = current_rank
        remain = [i for i in remain if i not in front]
        current_rank += 1

    return pd.Series(ranks, index=group.index, dtype=int)


def crowding_distance(group: pd.DataFrame, obj_cols: Sequence[str], maximize: Sequence[bool]) -> pd.Series:
    n = len(group)
    if n == 0:
        return pd.Series(dtype=float)
    if n <= 2:
        return pd.Series(np.full(n, np.inf), index=group.index, dtype=float)

    work = group.copy()
    dist = pd.Series(np.zeros(n, dtype=float), index=group.index)

    for col, is_max in zip(obj_cols, maximize):
        x = pd.to_numeric(work[col], errors="coerce").astype(float)
        order = x.sort_values(ascending=not is_max).index.tolist()

        dist.loc[order[0]] = np.inf
        dist.loc[order[-1]] = np.inf

        x_sorted = x.loc[order]
        x_min = float(x_sorted.min())
        x_max = float(x_sorted.max())
        denom = x_max - x_min
        if denom <= EPS:
            continue

        for i in range(1, len(order) - 1):
            prev_val = float(x.loc[order[i - 1]])
            next_val = float(x.loc[order[i + 1]])
            dist.loc[order[i]] += abs(next_val - prev_val) / denom

    return dist


def to_numeric_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# =============================================================================
# 2. 读取输入
# =============================================================================

if not STEP5_SUPPORT_CSV.exists():
    raise FileNotFoundError(f"缺少主输入表：{STEP5_SUPPORT_CSV}")
if not SPECIES_POOL_CSV.exists():
    raise FileNotFoundError(f"缺少物种池表：{SPECIES_POOL_CSV}")

work = pd.read_csv(STEP5_SUPPORT_CSV, encoding="utf-8-sig")
species_pool = pd.read_csv(SPECIES_POOL_CSV, encoding="utf-8-sig")

print(f"Step5 主输入表 shape: {work.shape}")
print(f"物种池 shape: {species_pool.shape}")


# =============================================================================
# 3. QA：基础字段检查
# =============================================================================

required_cols = [
    "ClassCode", "scheme_id", "is_feasible",
    "veg_zone_primary",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "target_LAI", "LAI_safe_max",
]
missing_required = [c for c in required_cols if c not in work.columns]
if missing_required:
    raise RuntimeError(f"Step5Support 缺少必要字段: {missing_required}")

# 若缺少辅助列，给默认值，避免后续中断
defaults = {
    "tree_species": "",
    "shrub_species": "",
    "grass_species": "",
    "tree_density_per_ha": np.nan,
    "shrub_density_per_ha": np.nan,
    "grass_cover_target_pct": np.nan,
    "scheme_LAI_capacity": np.nan,
    "LAI_margin": np.nan,
    "R_gde": 0.0,
    "water_group": "medium",
    "scheme_score": 0.0,
    "constraint_mode": "",
    "management_priority": "",
    "management_suggestion": "",
    "tree_allowed": True,
}
for k, v in defaults.items():
    if k not in work.columns:
        work[k] = v

print("\n========== QA 检查 ==========")
print(f"ClassCode 缺失数: {work['ClassCode'].isna().sum()}")
print(f"scheme_id 缺失数: {work['scheme_id'].isna().sum()}")
print(f"scheme_id 是否唯一: {not work['scheme_id'].duplicated().any()}")
print(f"is_feasible=1 的候选数: {(pd.to_numeric(work['is_feasible'], errors='coerce') == 1).sum()}")
print("主类型统计：")
print(work["veg_zone_primary"].value_counts(dropna=False))


# =============================================================================
# 4. 数值化与真实字段识别
# =============================================================================

possible_num_cols = [
    "ClassCode", "GDE_Level", "R_gde", "LAI_safe_max", "LAI_max_total", "LAI_current_3yr_mean",
    "target_LAI", "scheme_LAI_capacity", "LAI_margin",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
    "P_mean", "PET_mean", "AET_mean", "Runoff_mean", "Soil_mean", "Tmean_mean",
    "GWSA_mean_period_mean", "GWSA_trend_mean", "AI_full", "AI_prelim_noGW_mean", "AI_with_GWSA_proxy_mean",
    "Elevation_mean", "Slope_mean", "GDE_frac_mean", "GDE_stability_mean", "GDE_persistence_count_mean",
    "NDVI_mean", "NDVI_std", "FVC_mean", "FVC_std", "NPP_mean", "NPP_std", "WUE_mean", "WUE_std",
    "WindErosion_mean", "BareSoilFrac_mean", "SandConnectivity_mean",
]
to_numeric_if_exists(work, possible_num_cols)

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

has_real_stability_inputs = all(
    c is not None for c in [NDVI_MEAN_COL, NDVI_STD_COL, NPP_MEAN_COL, NPP_STD_COL, WUE_MEAN_COL, WUE_STD_COL]
)
has_real_sand_inputs = all(
    c is not None for c in [WIND_EROSION_COL, BARE_SOIL_COL, SAND_CONN_COL]
)

if not all(c is not None for c in [P_COL, PET_COL, AET_COL, SOIL_COL, RUNOFF_COL, AI_COL, GWSA_TREND_COL, LAI_CUR_COL]):
    raise RuntimeError("Step5Support 缺少关键水文字段，无法继续。")
if not has_real_stability_inputs:
    raise RuntimeError("Step5Support 缺少真实稳定性字段，无法继续。")
if not has_real_sand_inputs:
    raise RuntimeError("Step5Support 缺少真实固沙字段，无法继续。")

print("\n字段识别结果：")
print(f"P_COL={P_COL}")
print(f"PET_COL={PET_COL}")
print(f"AET_COL={AET_COL}")
print(f"SOIL_COL={SOIL_COL}")
print(f"RUNOFF_COL={RUNOFF_COL}")
print(f"AI_COL={AI_COL}")
print(f"GWSA_TREND_COL={GWSA_TREND_COL}")
print(f"LAI_CUR_COL={LAI_CUR_COL}")
print(f"真实稳定性输入是否齐全: {has_real_stability_inputs}")
print(f"真实固沙输入是否存在: {has_real_sand_inputs}")


# =============================================================================
# 5. 第五步前再次做硬约束复核
# =============================================================================

work["valid_lai_constraint"] = pd.to_numeric(work["target_LAI"], errors="coerce") <= (
    pd.to_numeric(work["LAI_safe_max"], errors="coerce") + EPS
)

work["valid_ratio_sum"] = np.isclose(
    pd.to_numeric(work["tree_ratio"], errors="coerce").fillna(0.0)
    + pd.to_numeric(work["shrub_ratio"], errors="coerce").fillna(0.0)
    + pd.to_numeric(work["grass_ratio"], errors="coerce").fillna(0.0),
    1.0,
    atol=1e-4,
)

work["valid_tree_entry"] = True
if "tree_allowed" in work.columns:
    tree_allowed_bool = work["tree_allowed"].astype(str).str.lower().isin(
        ["1", "true", "yes", "y", "t", "是"]
    )
    work.loc[(~tree_allowed_bool) & (pd.to_numeric(work["tree_ratio"], errors="coerce").fillna(0.0) > EPS), "valid_tree_entry"] = False

work["valid_for_moo"] = (
    pd.to_numeric(work["is_feasible"], errors="coerce").fillna(0).eq(1)
    & work["valid_lai_constraint"]
    & work["valid_ratio_sum"]
    & work["valid_tree_entry"]
)

print("\n第五步硬约束复核：")
print(work[["valid_lai_constraint", "valid_ratio_sum", "valid_tree_entry", "valid_for_moo"]].mean())

work = work.loc[work["valid_for_moo"]].copy()
if work.empty:
    raise RuntimeError("第五步输入为空：没有通过硬约束复核的候选方案。")


# =============================================================================
# 6. 物种性状赋值
# =============================================================================

trait_defaults = {
    "sand_tol": 2.0,
    "drought_tol": 2.0,
    "cold_tol": 2.0,
    "salt_tol": 1.0,
    "high_water_demand": 0.0,
    "gde_sensitive_exclusion": 0.0,
    "lai_coeff": 1.0,
}

for trait_col, default_val in trait_defaults.items():
    work[f"trait_{trait_col}"] = work.apply(
        lambda r: weighted_scheme_trait(r, species_pool, trait_col, default_val), axis=1
    )

work["species_n_total"] = (
    work["tree_species"].map(split_species).map(len)
    + work["shrub_species"].map(split_species).map(len)
    + work["grass_species"].map(split_species).map(len)
)


# =============================================================================
# 7. 构建环境 proxy
# =============================================================================

# 7.1 结构与利用率
work["density_utilization"] = np.clip(
    safe_div(work["target_LAI"], work["LAI_safe_max"], default=0.0),
    0.0,
    1.2,
)
work["moderate_use"] = work["density_utilization"].map(moderate_use_score)
work["layer_evenness"] = work[["tree_ratio", "shrub_ratio", "grass_ratio"]].apply(
    lambda x: shannon_evenness(x.tolist()), axis=1
)
work["layer_fraction"] = work[["tree_ratio", "shrub_ratio", "grass_ratio"]].apply(
    lambda x: present_layers_fraction(x.tolist()), axis=1
)
work["structure_sand_score"] = work.apply(zone_structure_score, axis=1)

# 7.2 水分压力背景
work["AET_P_ratio"] = safe_div(work[AET_COL], work[P_COL], default=np.nan)
work["PET_P_ratio"] = safe_div(work[PET_COL], work[P_COL], default=np.nan)

work["dryness_proxy"] = robust_minmax(work[AI_COL], reverse=True)

hydro_parts = []
if work["AET_P_ratio"].notna().any():
    hydro_parts.append(robust_minmax(work["AET_P_ratio"], reverse=False))
if work["PET_P_ratio"].notna().any():
    hydro_parts.append(robust_minmax(work["PET_P_ratio"], reverse=False))
hydro_parts.append(robust_minmax(work[SOIL_COL], reverse=True))

work["GWSA_decline"] = (-pd.to_numeric(work[GWSA_TREND_COL], errors="coerce")).clip(lower=0)
hydro_parts.append(robust_minmax(work["GWSA_decline"], reverse=False))

work["hydro_stress_base"] = np.mean(
    np.vstack([x.to_numpy(dtype=float) for x in hydro_parts]),
    axis=0
)

# 7.3 固沙需求背景
sand_parts = [
    robust_minmax(work[WIND_EROSION_COL], reverse=False),
    robust_minmax(work[BARE_SOIL_COL], reverse=False),
    robust_minmax(work[SAND_CONN_COL], reverse=False),
]
work["sand_demand_base"] = np.mean(
    np.vstack([x.to_numpy(dtype=float) for x in sand_parts]),
    axis=0
)

# 7.4 GDE 背景
work["gde_level_risk"] = work["GDE_Level"].map(map_gde_level).fillna(0.0)
work["gde_base_risk"] = robust_minmax(work["R_gde"], reverse=False)

# 7.5 真实生态稳定性背景
ndvi_mean_n = robust_minmax(work[NDVI_MEAN_COL], reverse=False)
ndvi_std_n = robust_minmax(work[NDVI_STD_COL], reverse=False)
npp_mean_n = robust_minmax(work[NPP_MEAN_COL], reverse=False)
npp_std_n = robust_minmax(work[NPP_STD_COL], reverse=False)
wue_mean_n = robust_minmax(work[WUE_MEAN_COL], reverse=False)
wue_std_n = robust_minmax(work[WUE_STD_COL], reverse=False)

qv = ndvi_mean_n
sv = (
    0.6 * (npp_mean_n * (1.0 - npp_std_n))
    + 0.2 * (wue_mean_n * (1.0 - wue_std_n))
    + 0.2 * (ndvi_mean_n * (1.0 - ndvi_std_n))
)
work["eco_background_real"] = (qv * sv).clip(lower=0.0, upper=1.0)


# =============================================================================
# 8. 四个目标函数
# =============================================================================

# 目标1：最大化防风固沙效益
work["obj1_sand_benefit"] = (
    0.35 * work["structure_sand_score"]
    + 0.20 * work["density_utilization"].clip(0, 1)
    + 0.15 * (work["trait_sand_tol"] / 3.0).clip(0, 1)
    + 0.10 * (work["trait_drought_tol"] / 3.0).clip(0, 1)
    + 0.20 * work["sand_demand_base"]
).clip(lower=0.0, upper=1.2)

# 目标2：最大化生态稳定性
work["obj2_eco_stability"] = (
    0.55 * work["eco_background_real"]
    + 0.15 * work["layer_evenness"]
    + 0.10 * work["layer_fraction"]
    + 0.10 * (work["trait_drought_tol"] / 3.0).clip(0, 1)
    + 0.10 * work["moderate_use"]
).clip(lower=0.0, upper=1.2)

# 目标3：最小化水资源压力
work["obj3_water_pressure"] = (
    0.35 * work["density_utilization"].clip(0, 1.2)
    + 0.20 * work["hydro_stress_base"]
    + 0.15 * work["dryness_proxy"]
    + 0.15 * work["tree_ratio"].clip(0, 1)
    + 0.10 * work["trait_high_water_demand"].clip(0, 1)
    + 0.05 * (1.0 - robust_minmax(work["LAI_margin"], reverse=False))
).clip(lower=0.0, upper=1.5)

# 目标4：最小化 GDE 风险
work["obj4_gde_risk"] = (
    0.40 * (work["gde_base_risk"] * (0.5 + 0.5 * work["density_utilization"].clip(0, 1.0)))
    + 0.20 * work["gde_level_risk"]
    + 0.15 * (work["tree_ratio"] * work["density_utilization"].clip(0, 1.0))
    + 0.15 * work["trait_high_water_demand"].clip(0, 1)
    + 0.10 * work["trait_gde_sensitive_exclusion"].clip(0, 1)
).clip(lower=0.0, upper=1.5)


# =============================================================================
# 9. 目标归一化（用于比较与 compromise）
# =============================================================================

for raw_col in [
    "obj1_sand_benefit", "obj2_eco_stability", "obj3_water_pressure", "obj4_gde_risk"
]:
    work[f"{raw_col}_norm"] = robust_minmax(work[raw_col])

# 统一成“越大越好”的可解释分数
work["score_sand"] = work["obj1_sand_benefit_norm"]
work["score_stability"] = work["obj2_eco_stability_norm"]
work["score_water_security"] = 1.0 - work["obj3_water_pressure_norm"]
work["score_gde_safety"] = 1.0 - work["obj4_gde_risk_norm"]


# =============================================================================
# 10. Pareto 排序（每个 ClassCode 内）
# =============================================================================

OBJ_COLS = [
    "obj1_sand_benefit",
    "obj2_eco_stability",
    "obj3_water_pressure",
    "obj4_gde_risk",
]
OBJ_MAX = [True, True, False, False]

rank_frames: List[pd.DataFrame] = []

for class_code, g in work.groupby("ClassCode", sort=True):
    g2 = g.copy()
    g2["pareto_rank"] = pareto_ranks(g2, OBJ_COLS, OBJ_MAX)
    g2["crowding_distance"] = np.nan

    for rank_i, sub_idx in g2.groupby("pareto_rank").groups.items():
        sub = g2.loc[sub_idx].copy()
        g2.loc[sub.index, "crowding_distance"] = crowding_distance(sub, OBJ_COLS, OBJ_MAX)

    rank1 = g2[g2["pareto_rank"] == 1].copy()
    if rank1.empty:
        g2["best_compromise_score"] = np.nan
        g2["is_best_compromise"] = 0
    else:
        # 每个 ClassCode 内做一次局地标准化
        s1 = robust_minmax(rank1["obj1_sand_benefit"])
        s2 = robust_minmax(rank1["obj2_eco_stability"])
        s3 = 1.0 - robust_minmax(rank1["obj3_water_pressure"])
        s4 = 1.0 - robust_minmax(rank1["obj4_gde_risk"])

        compromise = W_SAND * s1 + W_STABILITY * s2 + W_WATER_SECURITY * s3 + W_GDE_SAFETY * s4
        rank1["best_compromise_score"] = compromise

        rank1 = rank1.sort_values(
            by=["best_compromise_score", "crowding_distance", "scheme_score"],
            ascending=[False, False, False],
        ).copy()
        best_scheme_id = rank1.iloc[0]["scheme_id"]

        g2 = g2.merge(rank1[["scheme_id", "best_compromise_score"]], on="scheme_id", how="left")
        g2["is_best_compromise"] = (g2["scheme_id"] == best_scheme_id).astype(int)

    rank_frames.append(g2)

moo = pd.concat(rank_frames, axis=0, ignore_index=True)
moo["is_pareto_front"] = (moo["pareto_rank"] == 1).astype(int)


# =============================================================================
# 11. 输出 1：全部打分候选
# =============================================================================

ordered_cols = [
    "ClassCode", "scheme_id", "veg_zone_primary", "management_suggestion",
    "BaseZone", "GDE_Level", "constraint_mode", "management_priority",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_species", "shrub_species", "grass_species",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
    "target_LAI", "LAI_safe_max", "scheme_LAI_capacity", "LAI_margin",
    "R_gde", "water_group",
    "trait_sand_tol", "trait_drought_tol", "trait_high_water_demand", "trait_gde_sensitive_exclusion",
    "density_utilization", "layer_evenness", "layer_fraction",
    "sand_demand_base", "hydro_stress_base", "gde_base_risk",
    "obj1_sand_benefit", "obj2_eco_stability", "obj3_water_pressure", "obj4_gde_risk",
    "score_sand", "score_stability", "score_water_security", "score_gde_safety",
    "pareto_rank", "crowding_distance", "best_compromise_score",
    "is_pareto_front", "is_best_compromise",
]
existing_cols = [c for c in ordered_cols if c in moo.columns]
moo = moo[existing_cols].copy()
moo.to_csv(OUT_SCORED_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 12. 输出 2：Pareto 前沿
# =============================================================================

pareto_df = moo.loc[moo["is_pareto_front"] == 1].copy()
pareto_df = pareto_df.sort_values(
    by=["ClassCode", "best_compromise_score", "crowding_distance"],
    ascending=[True, False, False]
)
pareto_df.to_csv(OUT_PARETO_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 13. 输出 3：每个 ClassCode 的最佳折中解
# =============================================================================

best_df = moo.loc[moo["is_best_compromise"] == 1].copy()
best_df = best_df.sort_values(by=["ClassCode"]).copy()
best_df.to_csv(OUT_BEST_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 14. 输出 4：Pareto 解区间
# =============================================================================

range_df = (
    pareto_df.groupby(["ClassCode", "veg_zone_primary"], as_index=False)
    .agg(
        Pareto_n=("scheme_id", "count"),
        Tree_ratio_min=("tree_ratio", "min"),
        Tree_ratio_max=("tree_ratio", "max"),
        Shrub_ratio_min=("shrub_ratio", "min"),
        Shrub_ratio_max=("shrub_ratio", "max"),
        Grass_ratio_min=("grass_ratio", "min"),
        Grass_ratio_max=("grass_ratio", "max"),
        LAI_target_min=("target_LAI", "min"),
        LAI_target_max=("target_LAI", "max"),
        Tree_density_min=("tree_density_per_ha", "min"),
        Tree_density_max=("tree_density_per_ha", "max"),
        Shrub_density_min=("shrub_density_per_ha", "min"),
        Shrub_density_max=("shrub_density_per_ha", "max"),
        Grass_cover_min=("grass_cover_target_pct", "min"),
        Grass_cover_max=("grass_cover_target_pct", "max"),
        Sand_benefit_best=("obj1_sand_benefit", "max"),
        Eco_stability_best=("obj2_eco_stability", "max"),
        Water_pressure_best=("obj3_water_pressure", "min"),
        GDE_risk_best=("obj4_gde_risk", "min"),
    )
)
range_df.to_csv(OUT_RANGE_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 15. 输出 5：按主恢复类型汇总
# =============================================================================

zone_summary = (
    pareto_df.groupby("veg_zone_primary", as_index=False)
    .agg(
        Scheme_n=("scheme_id", "count"),
        Class_n=("ClassCode", "nunique"),
        Tree_ratio_mean=("tree_ratio", "mean"),
        Shrub_ratio_mean=("shrub_ratio", "mean"),
        Grass_ratio_mean=("grass_ratio", "mean"),
        Target_LAI_mean=("target_LAI", "mean"),
        Sand_benefit_mean=("obj1_sand_benefit", "mean"),
        Eco_stability_mean=("obj2_eco_stability", "mean"),
        Water_pressure_mean=("obj3_water_pressure", "mean"),
        GDE_risk_mean=("obj4_gde_risk", "mean"),
    )
)
zone_summary.to_csv(OUT_ZONE_SUMMARY_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 16. QA 报告
# =============================================================================

qa_lines = []
qa_lines.append("========== Step5 QA ==========")
qa_lines.append(f"主输入数: {len(pd.read_csv(STEP5_SUPPORT_CSV, encoding='utf-8-sig'))}")
qa_lines.append(f"通过第五步硬约束复核数: {len(work)}")
qa_lines.append(f"Scored candidate 数: {len(moo)}")
qa_lines.append(f"Pareto front 数: {len(pareto_df)}")
qa_lines.append(f"Best compromise 数: {len(best_df)}")
qa_lines.append("")
qa_lines.append("Pareto front 按主类型统计:")
qa_lines.append(pareto_df["veg_zone_primary"].value_counts().to_string())
qa_lines.append("")
qa_lines.append("每个 ClassCode 的 Pareto 解数量（前10）:")
qa_lines.append(
    pareto_df.groupby("ClassCode")["scheme_id"].count().sort_values(ascending=False).head(10).to_string()
)
qa_lines.append("")
qa_lines.append(f"真实稳定性输入是否齐全: {has_real_stability_inputs}")
qa_lines.append(f"真实固沙输入是否存在: {has_real_sand_inputs}")
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

OUT_QA_TXT.write_text("\n".join(qa_lines), encoding="utf-8")

# =============================================================================
# 17. 终端输出
# =============================================================================

print("\n========== 第五步结果摘要 ==========")
print(f"输出主表: {OUT_SCORED_CSV}")
print(f"输出 Pareto 前沿: {OUT_PARETO_CSV}")
print(f"输出最佳折中解: {OUT_BEST_CSV}")
print(f"输出 Pareto 区间: {OUT_RANGE_CSV}")
print(f"输出类型汇总: {OUT_ZONE_SUMMARY_CSV}")
print(f"输出 QA: {OUT_QA_TXT}")

print("\nPareto front 按主恢复类型统计：")
print(pareto_df["veg_zone_primary"].value_counts())

print("\n每个 ClassCode 的 Pareto 解数量（前10）：")
print(pareto_df.groupby("ClassCode")["scheme_id"].count().sort_values(ascending=False).head(10))

print("\nBest compromise 前5行：")
print(best_df.head())
