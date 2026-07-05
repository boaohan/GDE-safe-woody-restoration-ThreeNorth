"""
08_build_candidate_library.py

Builds feasible tree-shrub-grass candidate configurations by ClassCode under LAI and GDE constraints.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. 参数区
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

VEGTYPE_CSV = DATA_DIR / "ThreeNorth_Class_VegType_By_LAIsafe.csv"

# ---- 若这两个模板不存在，脚本会自动生成 seed 版本 ----
SPECIES_POOL_CSV = DATA_DIR / "ThreeNorth_SpeciesPool_Seed.csv"
RATIO_TEMPLATE_CSV = DATA_DIR / "ThreeNorth_RatioTemplate_Seed.csv"

OUT_MAIN_CSV = DATA_DIR / "ThreeNorth_Class_CandidateLibrary.csv"
OUT_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_CandidateLibrary_Summary.csv"
OUT_CLASS_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_CandidateLibrary_byClass.csv"
OUT_NOFEASIBLE_CSV = DATA_DIR / "ThreeNorth_Class_CandidateLibrary_NoFeasible.csv"

# ---- LAI 系数：与手册保持一致，后续可以替换为物种级系数 ----
DEFAULT_LAI_TREE = 3.0
DEFAULT_LAI_SHRUB = 1.0
DEFAULT_LAI_GRASS = 0.3

# ---- 目标 LAI 离散点：在第三步 target_LAI_lower~upper 内取点 ----
TARGET_LAI_GRID = [0.25, 0.50, 0.75]

# ---- 单类最多保留多少个候选方案，防止爆表 ----
MAX_SCHEMES_PER_CLASS = 36

# ---- 物种组合长度 ----
TREE_PICK_N = 1
SHRUB_PICK_N = 2
GRASS_PICK_N = 2


# =============================================================================
# 1. Seed 模板
# =============================================================================


def make_seed_species_pool() -> pd.DataFrame:
    rows = [
        # ---- 乔木：低耗水 / 干旱半干旱优先 ----
        ["樟子松", "乔木", "dry", 3, 3, 3, 2, 0, 0, 2.6, 450, 900],
        ["油松", "乔木", "dry", 3, 3, 3, 1, 0, 0, 2.8, 420, 850],
        ["白榆", "乔木", "dry", 3, 3, 3, 2, 0, 0, 2.4, 380, 760],
        ["山杏", "乔木", "dry", 3, 3, 3, 1, 0, 0, 2.1, 320, 680],
        ["彰武松", "乔木", "dry", 3, 3, 3, 2, 0, 0, 2.7, 420, 840],
        ["侧柏", "乔木", "dry", 2, 3, 3, 2, 0, 0, 2.2, 350, 700],
        ["圆柏", "乔木", "dry", 2, 3, 3, 1, 0, 0, 2.0, 320, 650],

        # ---- 乔木：中等水分区 ----
        ["沙地云杉", "乔木", "medium", 2, 2, 3, 1, 0, 1, 3.0, 450, 900],
        ["小黑杨", "乔木", "medium", 2, 2, 3, 1, 1, 1, 3.2, 500, 1000],
        ["青杨", "乔木", "medium", 2, 2, 3, 1, 1, 1, 3.1, 500, 1000],
        ["元宝槭", "乔木", "medium", 2, 2, 3, 1, 0, 0, 2.5, 360, 760],

        # ---- 灌木：干旱优先 ----
        ["小叶锦鸡儿", "灌木", "dry", 3, 3, 3, 2, 0, 0, 1.0, 1600, 3800],
        ["拧条锦鸡儿", "灌木", "dry", 3, 3, 3, 2, 0, 0, 1.1, 1400, 3200],
        ["沙棘", "灌木", "dry", 3, 3, 3, 2, 0, 0, 1.2, 1500, 3000],
        ["沙柳", "灌木", "dry", 3, 3, 3, 1, 0, 0, 1.0, 1800, 3600],
        ["差巴嘎蒿", "灌木", "dry", 3, 3, 3, 1, 0, 0, 0.7, 2200, 5000],
        ["细枝羊柴", "灌木", "dry", 3, 3, 3, 1, 0, 0, 0.8, 1800, 4200],
        ["沙拐枣", "灌木", "dry", 3, 3, 3, 1, 0, 0, 0.9, 1500, 3200],
        ["荆条", "灌木", "dry", 2, 3, 3, 1, 0, 0, 0.9, 1500, 3000],
        ["胡枝子", "灌木", "dry", 2, 3, 3, 1, 0, 0, 0.9, 1800, 3500],
        ["紫穗槐", "灌木", "dry", 2, 2, 3, 2, 0, 0, 1.0, 1600, 3200],

        # ---- 灌木：中等水分区 ----
        ["黄柳", "灌木", "medium", 2, 2, 3, 1, 0, 0, 1.1, 1800, 3400],
        ["树锦鸡儿", "灌木", "medium", 2, 2, 3, 1, 0, 0, 1.0, 1600, 3200],
        ["连翘", "灌木", "medium", 1, 2, 3, 1, 0, 0, 0.9, 1500, 2800],
        ["暴马丁香", "灌木", "medium", 1, 2, 3, 1, 0, 0, 0.8, 1200, 2500],
        ["锦带花", "灌木", "medium", 1, 2, 3, 1, 0, 0, 0.8, 1200, 2400],

        # ---- 草本：干旱优先 ----
        ["羊草", "草本", "dry", 3, 3, 3, 2, 0, 0, 0.35, 60, 90],
        ["沙生冰草", "草本", "dry", 3, 3, 3, 2, 0, 0, 0.35, 60, 90],
        ["沙打旺", "草本", "dry", 3, 3, 3, 1, 0, 0, 0.30, 50, 85],
        ["沙蓬", "草本", "dry", 3, 3, 3, 1, 0, 0, 0.25, 40, 75],
        ["蒙古韭", "草本", "dry", 2, 3, 3, 1, 0, 0, 0.25, 40, 70],
        ["地榆", "草本", "dry", 2, 3, 3, 1, 0, 0, 0.22, 35, 65],
        ["披碱草", "草本", "dry", 2, 3, 3, 2, 0, 0, 0.30, 50, 80],
        ["无芒雀麦", "草本", "dry", 2, 3, 3, 1, 0, 0, 0.30, 50, 80],
        ["草木樨", "草本", "dry", 2, 3, 3, 2, 0, 0, 0.28, 45, 75],

        # ---- 草本：中等水分区 ----
        ["紫花苜蓿", "草本", "medium", 2, 2, 3, 2, 0, 0, 0.30, 50, 80],
        ["结缕草", "草本", "medium", 2, 2, 3, 3, 0, 0, 0.25, 45, 70],
        ["白车轴草", "草本", "medium", 1, 2, 3, 1, 0, 0, 0.22, 40, 65],
        ["糙隐子草", "草本", "medium", 2, 2, 3, 1, 0, 0, 0.24, 40, 70],
        ["达乌里羊茅", "草本", "medium", 2, 2, 3, 1, 0, 0, 0.26, 45, 70],
    ]

    cols = [
        "species_name", "life_form", "water_group",
        "sand_tol", "drought_tol", "cold_tol", "salt_tol",
        "high_water_demand", "gde_sensitive_exclusion",
        "lai_coeff", "density_min", "density_max"
    ]
    return pd.DataFrame(rows, columns=cols)



def make_seed_ratio_templates() -> pd.DataFrame:
    rows = [
        # ratio_order 固定采用 T:S:G
        ["保育/自然恢复", "conservation", 0.0, 0.0, 0.0, "保育优先，不进入人工配置"],

        ["低覆盖灌草", "dry_low_cover", 0.0, 0.2, 0.8, "经验模板-草本主导"],
        ["低覆盖灌草", "dry_low_cover", 0.0, 0.3, 0.7, "经验模板-灌草均衡"],
        ["低覆盖灌草", "dry_low_cover", 0.0, 0.4, 0.6, "低覆盖灌草扩展模板"],

        ["灌草主导", "shrub_grass", 0.0, 0.4, 0.6, "配置手册（灌:草）40:60"],
        ["灌草主导", "shrub_grass", 0.0, 0.5, 0.5, "灌草主导均衡模板"],
        ["灌草主导", "shrub_grass", 0.0, 0.6, 0.4, "配置手册（灌:草）60:40"],
        ["灌草主导", "shrub_grass", 0.1, 0.3, 0.6, "经营手册 1:3:6（按T:S:G解释）"],
        ["灌草主导", "shrub_grass", 0.1, 0.2, 0.7, "经营手册 1:2:7（按T:S:G解释）"],

        ["低密乔灌草", "low_tree_mix", 0.1, 0.2, 0.7, "低密乔木进入"],
        ["低密乔灌草", "low_tree_mix", 0.1, 0.3, 0.6, "低密乔木进入"],
        ["低密乔灌草", "low_tree_mix", 0.2, 0.2, 0.6, "经营手册 2:2:6"],
        ["低密乔灌草", "low_tree_mix", 0.2, 0.1, 0.7, "经营手册 2:1:7"],
        ["低密乔灌草", "low_tree_mix", 0.2, 0.3, 0.5, "经营手册 2:3:5"],

        ["乔灌草", "tree_shrub_grass", 0.2, 0.2, 0.6, "配置手册代表点"],
        ["乔灌草", "tree_shrub_grass", 0.3, 0.3, 0.4, "经营手册 3:3:4"],
        ["乔灌草", "tree_shrub_grass", 0.4, 0.1, 0.5, "经营手册 4:1:5"],
        ["乔灌草", "tree_shrub_grass", 0.4, 0.2, 0.4, "乔灌草扩展模板"],
        ["乔灌草", "tree_shrub_grass", 0.6, 0.2, 0.2, "经营手册 6:2:2"],
        ["乔灌草", "tree_shrub_grass", 0.6, 0.3, 0.1, "经营手册 6:3:1"],
        ["乔灌草", "tree_shrub_grass", 0.7, 0.2, 0.1, "经营手册 7:2:1"],
        ["乔灌草", "tree_shrub_grass", 0.8, 0.1, 0.1, "经营手册 8:1:1"],
    ]

    return pd.DataFrame(
        rows,
        columns=["veg_zone_primary", "template_group", "tree_ratio", "shrub_ratio", "grass_ratio", "source_note"]
    )


# =============================================================================
# 2. 工具函数
# =============================================================================


def ensure_seed_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False, encoding="utf-8-sig")



def normalize_zone_name(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    mapping = {
        "乔灌草候选": "乔灌草",
        "乔—灌—草": "乔灌草",
        "乔-灌-草": "乔灌草",
        "乔灌草混配": "乔灌草",
        "灌草": "灌草主导",
        "低覆盖灌草型": "低覆盖灌草",
        "灌草主导型": "灌草主导",
        "低密乔灌草型": "低密乔灌草",
        "保育": "保育/自然恢复",
        "自然恢复": "保育/自然恢复",
    }
    return mapping.get(text, text)



def parse_allowed_modes(s: object) -> set[str]:
    if pd.isna(s):
        return set()
    text = str(s).replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    out = {normalize_zone_name(x.strip()) for x in text.split(",") if x.strip()}
    return {x for x in out if x}



def to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    text = str(v).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "是"}



def standardize_ratio_templates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["veg_zone_primary"] = out["veg_zone_primary"].map(normalize_zone_name)
    for col in ["tree_ratio", "shrub_ratio", "grass_ratio"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["veg_zone_primary", "tree_ratio", "shrub_ratio", "grass_ratio"])
    out = out.loc[(out[["tree_ratio", "shrub_ratio", "grass_ratio"]] >= 0).all(axis=1)].copy()
    total = out[["tree_ratio", "shrub_ratio", "grass_ratio"]].sum(axis=1)
    out = out.loc[total > 0].copy()
    out[["tree_ratio", "shrub_ratio", "grass_ratio"]] = out[["tree_ratio", "shrub_ratio", "grass_ratio"]].div(total, axis=0)
    out[["tree_ratio", "shrub_ratio", "grass_ratio"]] = out[["tree_ratio", "shrub_ratio", "grass_ratio"]].round(4)
    out = out.drop_duplicates(subset=["veg_zone_primary", "template_group", "tree_ratio", "shrub_ratio", "grass_ratio"])
    return out.reset_index(drop=True)



def infer_water_group(row: pd.Series) -> str:
    zone = normalize_zone_name(row.get("veg_zone_primary", ""))
    safe_max = float(row.get("LAI_safe_max", np.nan) if pd.notna(row.get("LAI_safe_max", np.nan)) else np.nan)

    if zone in {"低覆盖灌草", "灌草主导"}:
        return "dry"
    if zone in {"低密乔灌草", "乔灌草"}:
        return "medium"
    if pd.notna(safe_max) and safe_max >= 3.0:
        return "medium"
    return "dry"



def scheme_lai_capacity(
    tree_ratio: float,
    shrub_ratio: float,
    grass_ratio: float,
    tree_lai: float = DEFAULT_LAI_TREE,
    shrub_lai: float = DEFAULT_LAI_SHRUB,
    grass_lai: float = DEFAULT_LAI_GRASS,
) -> float:
    return tree_ratio * tree_lai + shrub_ratio * shrub_lai + grass_ratio * grass_lai



def pick_target_lai_values(row: pd.Series) -> List[float]:
    lower = float(row.get("target_LAI_lower", np.nan))
    upper = float(row.get("target_LAI_upper", np.nan))
    safe_max = float(row.get("LAI_safe_max", np.nan))

    if pd.isna(safe_max) or safe_max <= 0:
        return []

    if pd.isna(lower) or pd.isna(upper):
        lower = safe_max * 0.70
        upper = safe_max * 0.90

    if upper < lower:
        lower, upper = upper, lower

    out = []
    for q in TARGET_LAI_GRID:
        val = lower + (upper - lower) * q
        val = min(val, safe_max)
        if val > 0:
            out.append(round(float(val), 4))

    return sorted(set(out))



def clip_tree_ratio_by_risk(tree_ratio: float, row: pd.Series) -> bool:
    gde_level = int(pd.to_numeric(row.get("GDE_Level", 0), errors="coerce") if pd.notna(row.get("GDE_Level", 0)) else 0)
    r_gde = float(pd.to_numeric(row.get("R_gde", 0.0), errors="coerce") if pd.notna(row.get("R_gde", 0.0)) else 0.0)
    zone = normalize_zone_name(row.get("veg_zone_primary", ""))
    tree_allowed_flag = to_bool(row.get("tree_allowed", False))

    if zone == "保育/自然恢复":
        return tree_ratio == 0.0

    if not tree_allowed_flag and tree_ratio > 0:
        return False

    if gde_level == 1 and tree_ratio > 0:
        return False

    if (gde_level == 3 or r_gde >= 0.50) and tree_ratio > 0.10:
        return False

    if zone == "灌草主导" and tree_ratio > 0.10:
        return False

    if zone == "低覆盖灌草" and tree_ratio > 0.0:
        return False

    if zone == "低密乔灌草" and tree_ratio > 0.20:
        return False

    return True



def template_is_allowed(row: pd.Series, tpl: pd.Series) -> bool:
    zone = normalize_zone_name(row.get("veg_zone_primary", ""))
    tpl_zone = normalize_zone_name(tpl.get("veg_zone_primary", ""))

    t = float(tpl["tree_ratio"])
    s = float(tpl["shrub_ratio"])
    g = float(tpl["grass_ratio"])

    # 1) 主类型必须一致
    if zone != tpl_zone:
        return False

    # 2) 风险与乔木约束
    if not clip_tree_ratio_by_risk(t, row):
        return False

    # 3) 按主类型做结构校验
    if zone == "保育/自然恢复":
        return t == 0 and s == 0 and g == 0

    if zone == "低覆盖灌草":
        return t == 0 and s >= 0 and g > 0

    if zone == "灌草主导":
        return t <= 0.10 and s > 0 and g > 0

    if zone == "低密乔灌草":
        return 0 < t <= 0.20 and s > 0 and g > 0

    if zone == "乔灌草":
        return t > 0 and s > 0 and g > 0

    return False



def rank_species(
    species_df: pd.DataFrame,
    life_form: str,
    water_group: str,
    row: pd.Series,
) -> pd.DataFrame:
    gde_level = int(pd.to_numeric(row.get("GDE_Level", 0), errors="coerce") if pd.notna(row.get("GDE_Level", 0)) else 0)
    r_gde = float(pd.to_numeric(row.get("R_gde", 0.0), errors="coerce") if pd.notna(row.get("R_gde", 0.0)) else 0.0)

    pool = species_df.loc[species_df["life_form"] == life_form].copy()

    if water_group == "dry":
        pool = pool.loc[pool["water_group"].isin(["dry"])]
    else:
        pool = pool.loc[pool["water_group"].isin(["dry", "medium"])]

    # 高风险区 / 核心保育区：剔除高耗水或 GDE 敏感排斥种
    if gde_level >= 2 or r_gde >= 0.50:
        pool = pool.loc[(pool["high_water_demand"] == 0) & (pool["gde_sensitive_exclusion"] == 0)]

    if pool.empty:
        return pool

    pool = pool.copy()
    pool["rank_score"] = (
        pool["sand_tol"].astype(float) * 0.30
        + pool["drought_tol"].astype(float) * 0.35
        + pool["cold_tol"].astype(float) * 0.15
        + pool["salt_tol"].astype(float) * 0.10
        + (1.5 - pool["high_water_demand"].astype(float)) * 0.10
    )

    return pool.sort_values(
        by=["rank_score", "lai_coeff", "density_max"],
        ascending=[False, False, False]
    ).reset_index(drop=True)



def rolling_pick_names(names: Sequence[str], k: int, variant: int) -> List[str]:
    if k <= 0:
        return []
    uniq = list(dict.fromkeys([str(x) for x in names if pd.notna(x) and str(x).strip()]))
    if not uniq:
        return []
    if len(uniq) <= k:
        return uniq

    start = variant % len(uniq)
    picked = []
    i = 0
    while len(picked) < k and i < len(uniq) * 2:
        name = uniq[(start + i) % len(uniq)]
        if name not in picked:
            picked.append(name)
        i += 1
    return picked



def weighted_mean_density(pool: pd.DataFrame) -> float:
    if pool.empty:
        return np.nan
    return float(((pool["density_min"] + pool["density_max"]) / 2).mean())



def weighted_mean_lai(pool: pd.DataFrame, default_lai: float) -> float:
    if pool.empty:
        return default_lai
    return float(pool["lai_coeff"].mean())



def build_density_targets(
    target_lai: float,
    capacity_lai: float,
    tree_ratio: float,
    shrub_ratio: float,
    grass_ratio: float,
    tree_pool: pd.DataFrame,
    shrub_pool: pd.DataFrame,
    grass_pool: pd.DataFrame,
) -> Tuple[float, float, float]:
    if capacity_lai <= 0:
        return np.nan, np.nan, np.nan

    density_scale = target_lai / capacity_lai

    tree_base = weighted_mean_density(tree_pool)
    shrub_base = weighted_mean_density(shrub_pool)
    grass_base = weighted_mean_density(grass_pool)

    tree_density = np.nan if pd.isna(tree_base) or tree_ratio <= 0 else tree_base * tree_ratio * density_scale
    shrub_density = np.nan if pd.isna(shrub_base) or shrub_ratio <= 0 else shrub_base * shrub_ratio * density_scale

    if pd.isna(grass_base) or grass_ratio <= 0:
        grass_cover = np.nan
    else:
        grass_cover = np.clip(100.0 * grass_ratio * density_scale, 5.0, 95.0)

    return (
        round(float(tree_density), 2) if pd.notna(tree_density) else np.nan,
        round(float(shrub_density), 2) if pd.notna(shrub_density) else np.nan,
        round(float(grass_cover), 2) if pd.notna(grass_cover) else np.nan,
    )



def management_label(row: pd.Series, tree_ratio: float) -> str:
    zone = normalize_zone_name(row.get("veg_zone_primary", ""))
    if zone == "保育/自然恢复":
        return "封育保护/自然恢复"
    if zone == "低覆盖灌草":
        return "低覆盖灌草恢复"
    if zone == "灌草主导":
        return "灌草主导恢复"
    if zone == "低密乔灌草":
        return "低密乔灌草恢复"
    if zone == "乔灌草":
        return "乔灌草协同恢复"
    if tree_ratio <= 0.1:
        return "灌草主导恢复"
    return "综合恢复"



def keep_top_schemes(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    score_cols = ["scheme_score", "target_LAI", "LAI_margin"]
    for c in score_cols:
        if c not in df.columns:
            df[c] = np.nan

    return (
        df.sort_values(
            by=["scheme_score", "target_LAI", "LAI_margin", "tree_ratio"],
            ascending=[False, False, False, True]
        )
        .groupby("ClassCode", group_keys=False)
        .head(max_n)
        .reset_index(drop=True)
    )


# =============================================================================
# 3. 读取数据
# =============================================================================

ensure_seed_csv(SPECIES_POOL_CSV, make_seed_species_pool())
ensure_seed_csv(RATIO_TEMPLATE_CSV, make_seed_ratio_templates())

veg_df = pd.read_csv(VEGTYPE_CSV)
species_df = pd.read_csv(SPECIES_POOL_CSV)
ratio_df = pd.read_csv(RATIO_TEMPLATE_CSV)

veg_df["veg_zone_primary"] = veg_df["veg_zone_primary"].map(normalize_zone_name)
ratio_df = standardize_ratio_templates(ratio_df)

# 兼容旧 seed：自动用标准化后的模板覆盖内存对象，不强制改写磁盘原表
for col in ["tree_ratio", "shrub_ratio", "grass_ratio"]:
    ratio_df[col] = pd.to_numeric(ratio_df[col], errors="coerce")

print(f"输入植被类型表 shape: {veg_df.shape}")
print(f"输入物种池表 shape: {species_df.shape}")
print(f"输入比例模板表 shape: {ratio_df.shape}")
print("植被类型表前5行：")
print(veg_df.head())


# =============================================================================
# 4. QA 检查
# =============================================================================

print("\n========== QA 检查 ==========")
print(f"ClassCode 是否唯一: {not veg_df['ClassCode'].duplicated().any()}")
print(f"LAI_safe_max 缺失数: {veg_df['LAI_safe_max'].isna().sum()}")
print(f"target_LAI_lower 缺失数: {veg_df['target_LAI_lower'].isna().sum()}")
print(f"target_LAI_upper 缺失数: {veg_df['target_LAI_upper'].isna().sum()}")

required_species_cols = {
    "species_name", "life_form", "water_group",
    "sand_tol", "drought_tol", "cold_tol", "salt_tol",
    "high_water_demand", "gde_sensitive_exclusion",
    "lai_coeff", "density_min", "density_max"
}
required_ratio_cols = {
    "veg_zone_primary", "template_group", "tree_ratio", "shrub_ratio", "grass_ratio", "source_note"
}

print(f"物种池字段完整: {required_species_cols.issubset(species_df.columns)}")
print(f"比例模板字段完整: {required_ratio_cols.issubset(ratio_df.columns)}")
print("标准化后的 veg_zone_primary 统计：")
print(veg_df["veg_zone_primary"].value_counts(dropna=False))
print("标准化后的比例模板统计：")
print(ratio_df["veg_zone_primary"].value_counts(dropna=False))

if not required_species_cols.issubset(species_df.columns):
    missing = sorted(required_species_cols - set(species_df.columns))
    raise ValueError(f"物种池缺少字段: {missing}")

if not required_ratio_cols.issubset(ratio_df.columns):
    missing = sorted(required_ratio_cols - set(ratio_df.columns))
    raise ValueError(f"比例模板缺少字段: {missing}")


# =============================================================================
# 5. 建立候选组合库
# =============================================================================

rows = []
no_feasible_rows = []

for _, row in veg_df.iterrows():
    class_code = row["ClassCode"]
    zone = normalize_zone_name(row["veg_zone_primary"])
    water_group = infer_water_group(row)

    target_lai_values = pick_target_lai_values(row)
    ratio_sub = ratio_df.loc[ratio_df["veg_zone_primary"] == zone].copy()
    ratio_sub = ratio_sub.loc[ratio_sub.apply(lambda r: template_is_allowed(row, r), axis=1)].copy()

    # 保育类单独处理
    if zone == "保育/自然恢复":
        current_lai = pd.to_numeric(row.get("LAI_current_3yr_mean", np.nan), errors="coerce")
        lai_safe_max = pd.to_numeric(row.get("LAI_safe_max", np.nan), errors="coerce")
        target_lai = 0.0 if pd.isna(lai_safe_max) else min(0.0 if pd.isna(current_lai) else float(current_lai), float(lai_safe_max))

        rows.append({
            "ClassCode": class_code,
            "BaseZone": row.get("BaseZone", np.nan),
            "GDE_Level": row.get("GDE_Level", np.nan),
            "constraint_mode": row.get("constraint_mode", np.nan),
            "management_priority": row.get("management_priority", np.nan),
            "veg_zone_primary": zone,
            "allowed_modes": row.get("allowed_modes", ""),
            "risk_level_simple": row.get("risk_level_simple", ""),
            "R_gde": row.get("R_gde", np.nan),
            "LAI_safe_max": lai_safe_max,
            "target_LAI_lower": row.get("target_LAI_lower", np.nan),
            "target_LAI_upper": row.get("target_LAI_upper", np.nan),
            "scheme_id": f"{int(class_code)}_0001",
            "template_group": "conservation",
            "source_note": "保育优先，不进入人工组合",
            "tree_ratio": 0.0,
            "shrub_ratio": 0.0,
            "grass_ratio": 0.0,
            "tree_species": "",
            "shrub_species": "",
            "grass_species": "",
            "tree_density_per_ha": np.nan,
            "shrub_density_per_ha": np.nan,
            "grass_cover_target_pct": np.nan,
            "target_LAI": round(float(target_lai), 4),
            "scheme_LAI_capacity": 0.0,
            "LAI_margin": round(float(lai_safe_max - target_lai), 4) if pd.notna(lai_safe_max) else np.nan,
            "water_group": water_group,
            "tree_allowed": to_bool(row.get("tree_allowed", False)),
            "management_suggestion": "封育保护/自然恢复",
            "scheme_score": 1.0,
            "is_feasible": 1,
        })
        continue

    if ratio_sub.empty:
        no_feasible_rows.append({
            "ClassCode": class_code,
            "veg_zone_primary": zone,
            "reason": "无可用比例模板"
        })
        continue

    if not target_lai_values:
        no_feasible_rows.append({
            "ClassCode": class_code,
            "veg_zone_primary": zone,
            "reason": "目标LAI为空"
        })
        continue

    tree_pool = rank_species(species_df, "乔木", water_group, row)
    shrub_pool = rank_species(species_df, "灌木", water_group, row)
    grass_pool = rank_species(species_df, "草本", water_group, row)

    if shrub_pool.empty or grass_pool.empty:
        no_feasible_rows.append({
            "ClassCode": class_code,
            "veg_zone_primary": zone,
            "reason": "灌木或草本物种池为空"
        })
        continue

    if zone in {"低密乔灌草", "乔灌草"} and tree_pool.empty:
        no_feasible_rows.append({
            "ClassCode": class_code,
            "veg_zone_primary": zone,
            "reason": "乔木物种池为空"
        })
        continue

    candidate_counter = 1

    for _, tpl in ratio_sub.iterrows():
        tree_ratio = float(tpl["tree_ratio"])
        shrub_ratio = float(tpl["shrub_ratio"])
        grass_ratio = float(tpl["grass_ratio"])

        variant_n = 2 if tree_ratio > 0 else 3

        for target_lai, variant in product(target_lai_values, range(variant_n)):
            tree_names = rolling_pick_names(tree_pool["species_name"].tolist(), TREE_PICK_N, variant) if tree_ratio > 0 else []
            shrub_names = rolling_pick_names(shrub_pool["species_name"].tolist(), SHRUB_PICK_N, variant)
            grass_names = rolling_pick_names(grass_pool["species_name"].tolist(), GRASS_PICK_N, variant)

            if tree_ratio > 0 and not tree_names:
                continue
            if shrub_ratio > 0 and not shrub_names:
                continue
            if grass_ratio > 0 and not grass_names:
                continue

            tree_sub = tree_pool.loc[tree_pool["species_name"].isin(tree_names)].copy()
            shrub_sub = shrub_pool.loc[shrub_pool["species_name"].isin(shrub_names)].copy()
            grass_sub = grass_pool.loc[grass_pool["species_name"].isin(grass_names)].copy()

            tree_lai = weighted_mean_lai(tree_sub, DEFAULT_LAI_TREE)
            shrub_lai = weighted_mean_lai(shrub_sub, DEFAULT_LAI_SHRUB)
            grass_lai = weighted_mean_lai(grass_sub, DEFAULT_LAI_GRASS)

            cap_lai = scheme_lai_capacity(
                tree_ratio=tree_ratio,
                shrub_ratio=shrub_ratio,
                grass_ratio=grass_ratio,
                tree_lai=tree_lai,
                shrub_lai=shrub_lai,
                grass_lai=grass_lai,
            )

            lai_safe_max = float(row["LAI_safe_max"])
            if cap_lai <= 0:
                continue
            if target_lai - lai_safe_max > 1e-6:
                continue

            tree_density, shrub_density, grass_cover = build_density_targets(
                target_lai=target_lai,
                capacity_lai=cap_lai,
                tree_ratio=tree_ratio,
                shrub_ratio=shrub_ratio,
                grass_ratio=grass_ratio,
                tree_pool=tree_sub,
                shrub_pool=shrub_sub,
                grass_pool=grass_sub,
            )

            lai_margin = lai_safe_max - target_lai

            species_score = (
                (tree_sub["rank_score"].mean() if not tree_sub.empty else 0.0)
                + (shrub_sub["rank_score"].mean() if not shrub_sub.empty else 0.0)
                + (grass_sub["rank_score"].mean() if not grass_sub.empty else 0.0)
            )

            safety_bonus = 1.0 - abs(lai_margin) / max(lai_safe_max, 1e-6)
            safety_bonus = max(0.0, safety_bonus)

            scheme_score = species_score * 0.6 + safety_bonus * 0.4

            rows.append({
                "ClassCode": class_code,
                "BaseZone": row.get("BaseZone", np.nan),
                "GDE_Level": row.get("GDE_Level", np.nan),
                "constraint_mode": row.get("constraint_mode", np.nan),
                "management_priority": row.get("management_priority", np.nan),
                "veg_zone_primary": zone,
                "allowed_modes": row.get("allowed_modes", ""),
                "risk_level_simple": row.get("risk_level_simple", ""),
                "R_gde": row.get("R_gde", np.nan),
                "LAI_safe_max": lai_safe_max,
                "target_LAI_lower": row.get("target_LAI_lower", np.nan),
                "target_LAI_upper": row.get("target_LAI_upper", np.nan),
                "scheme_id": f"{int(class_code)}_{candidate_counter:04d}",
                "template_group": tpl["template_group"],
                "source_note": tpl["source_note"],
                "tree_ratio": round(tree_ratio, 4),
                "shrub_ratio": round(shrub_ratio, 4),
                "grass_ratio": round(grass_ratio, 4),
                "tree_species": "、".join(tree_names),
                "shrub_species": "、".join(shrub_names),
                "grass_species": "、".join(grass_names),
                "tree_density_per_ha": tree_density,
                "shrub_density_per_ha": shrub_density,
                "grass_cover_target_pct": grass_cover,
                "target_LAI": round(target_lai, 4),
                "scheme_LAI_capacity": round(cap_lai, 4),
                "LAI_margin": round(lai_margin, 4),
                "water_group": water_group,
                "tree_allowed": to_bool(row.get("tree_allowed", False)),
                "management_suggestion": management_label(row, tree_ratio),
                "scheme_score": round(float(scheme_score), 4),
                "is_feasible": 1,
            })
            candidate_counter += 1

    if candidate_counter == 1:
        no_feasible_rows.append({
            "ClassCode": class_code,
            "veg_zone_primary": zone,
            "reason": "生成后无满足约束的候选方案"
        })

cand_df = pd.DataFrame(rows)
no_feasible_df = pd.DataFrame(no_feasible_rows)

if cand_df.empty:
    raise RuntimeError("候选组合库为空，请检查第三步输出字段、比例模板或物种池。")

cand_df = keep_top_schemes(cand_df, MAX_SCHEMES_PER_CLASS)


# =============================================================================
# 6. 输出主表
# =============================================================================

ordered_cols = [
    "ClassCode",
    "scheme_id",
    "BaseZone",
    "GDE_Level",
    "constraint_mode",
    "management_priority",

    "veg_zone_primary",
    "allowed_modes",
    "tree_allowed",
    "risk_level_simple",
    "R_gde",
    "water_group",

    "LAI_safe_max",
    "target_LAI_lower",
    "target_LAI_upper",
    "target_LAI",
    "scheme_LAI_capacity",
    "LAI_margin",

    "template_group",
    "source_note",

    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",

    "tree_species",
    "shrub_species",
    "grass_species",

    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",

    "management_suggestion",
    "scheme_score",
    "is_feasible",
]

cand_df = cand_df[ordered_cols].copy()
cand_df.to_csv(OUT_MAIN_CSV, index=False, encoding="utf-8-sig")

if not no_feasible_df.empty:
    no_feasible_df.to_csv(OUT_NOFEASIBLE_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 7. 输出汇总表
# =============================================================================

summary = (
    cand_df.groupby(["veg_zone_primary", "template_group"], as_index=False)
    .agg(
        Scheme_n=("scheme_id", "count"),
        Class_n=("ClassCode", "nunique"),
        LAI_safe_max_mean=("LAI_safe_max", "mean"),
        Target_LAI_mean=("target_LAI", "mean"),
        Tree_ratio_mean=("tree_ratio", "mean"),
        Shrub_ratio_mean=("shrub_ratio", "mean"),
        Grass_ratio_mean=("grass_ratio", "mean"),
        Scheme_score_mean=("scheme_score", "mean"),
    )
    .sort_values(["veg_zone_primary", "Scheme_n"], ascending=[True, False])
    .reset_index(drop=True)
)

summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

class_summary = (
    cand_df.groupby(["ClassCode", "veg_zone_primary"], as_index=False)
    .agg(
        Candidate_n=("scheme_id", "count"),
        Best_scheme_score=("scheme_score", "max"),
        LAI_safe_max=("LAI_safe_max", "first"),
        Target_LAI_min=("target_LAI", "min"),
        Target_LAI_max=("target_LAI", "max"),
        Tree_ratio_min=("tree_ratio", "min"),
        Tree_ratio_max=("tree_ratio", "max"),
        Shrub_ratio_min=("shrub_ratio", "min"),
        Shrub_ratio_max=("shrub_ratio", "max"),
        Grass_ratio_min=("grass_ratio", "min"),
        Grass_ratio_max=("grass_ratio", "max"),
    )
    .sort_values("ClassCode")
    .reset_index(drop=True)
)

class_summary.to_csv(OUT_CLASS_SUMMARY_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 8. 终端输出
# =============================================================================

print("\n========== 第四步结果摘要 ==========")
print(f"输出主表: {OUT_MAIN_CSV}")
print(f"输出汇总表: {OUT_SUMMARY_CSV}")
print(f"输出按单元汇总表: {OUT_CLASS_SUMMARY_CSV}")
print(f"物种池 seed 表: {SPECIES_POOL_CSV}")
print(f"比例模板 seed 表: {RATIO_TEMPLATE_CSV}")

if not no_feasible_df.empty:
    print(f"无可行解单元表: {OUT_NOFEASIBLE_CSV}")
    print(f"无可行解单元数: {no_feasible_df['ClassCode'].nunique()}")
    print("无可行解原因统计：")
    print(no_feasible_df["reason"].value_counts())

print("\n候选方案数量统计：")
print(cand_df["veg_zone_primary"].value_counts())

print("\n每个 ClassCode 的候选方案数（前10）：")
print(cand_df.groupby("ClassCode")["scheme_id"].count().sort_values(ascending=False).head(10))

print("\n前5行预览：")
print(cand_df.head())
