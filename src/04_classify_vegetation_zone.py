"""
04_classify_vegetation_zone.py

Classifies vegetation restoration type intervals from LAI_safe_max and GDE constraints.

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
# =============================================================================
# 0. 参数区
# =============================================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"
SAFE_CSV = DATA_DIR / "ThreeNorth_Class_LAI_safe_max.csv"
OUT_MAIN_CSV = DATA_DIR / "ThreeNorth_Class_VegType_By_LAIsafe.csv"
OUT_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_VegType_Summary.csv"
OUT_CROSSTAB_CSV = DATA_DIR / "ThreeNorth_Class_VegType_by_GDELevel.csv"
# =============================================================================
# 1. 读取数据
# =============================================================================
df = pd.read_csv(SAFE_CSV)
print(f"输入表 shape: {df.shape}")
print("前5行：")
print(df.head())
# =============================================================================
# 2. QA 检查
# =============================================================================
print("\n========== QA 检查 ==========")
print(f"ClassCode 是否唯一: {not df['ClassCode'].duplicated().any()}")
print(f"LAI_safe_max 缺失数: {df['LAI_safe_max'].isna().sum()}")
print(f"LAI_safe_max 负值数: {(df['LAI_safe_max'] < 0).sum()}")
print(f"LAI_safe_max > LAI_max_total 的行数: {(df['LAI_safe_max'] > df['LAI_max_total']).sum()}")
print(f"核心保育类违反硬约束的行数: {((df['GDE_Level'] == 1) & (df['LAI_safe_max'] > df['LAI_current_3yr_mean'])).sum()}")
print("\nobs_support_flag 统计：")
print(df["obs_support_flag"].value_counts(dropna=False))
print("\nmanagement_priority 统计：")
print(df["management_priority"].value_counts(dropna=False))
# =============================================================================
# 3. 分类函数
# =============================================================================
def classify_primary_type(row: pd.Series) -> str:
    safe_max = float(row["LAI_safe_max"])
    gde_level = int(row["GDE_Level"])
    r_gde = float(row["R_gde"])
    mode = str(row["constraint_mode"])
    # 1) 核心保育单元优先
    if gde_level == 1 or mode == "core_conservation":
        return "保育/自然恢复"
    # 2) 先按 LAI_safe_max 基础分段
    if safe_max < 0.5:
        return "保育/自然恢复"
    elif safe_max < 1.5:
        return "低覆盖灌草"
    elif safe_max < 3.0:
        return "灌草主导"
    elif safe_max < 4.5:
        # 3) 中高安全上限但高约束/高风险时降级
        if gde_level == 3 or r_gde >= 0.55:
            return "灌草主导"
        else:
            return "低密乔灌草"
    else:
        # 4) 高 LAI_safe_max 条件下再做风险约束
        if gde_level == 0 and r_gde < 0.35:
            return "乔灌草"
        elif gde_level in [0, 2] and r_gde < 0.50:
            return "低密乔灌草"
        else:
            return "灌草主导"
def allowed_modes(primary_type: str) -> str:
    mapping = {
        "保育/自然恢复": "保育优先;自然恢复;封育",
        "低覆盖灌草": "草本主导;低覆盖灌草",
        "灌草主导": "灌草主导;草本-灌木",
        "低密乔灌草": "低密乔灌草;灌草主导",
        "乔灌草": "乔灌草;低密乔灌草;灌草主导",
    }
    return mapping.get(primary_type, "unknown")
def tree_allowed(primary_type: str) -> int:
    return int(primary_type in ["低密乔灌草", "乔灌草"])
def target_lai_interval(row: pd.Series) -> tuple[float, float]:
    """
    根据主类型给一个建议目标 LAI 区间。
    上界永远不超过 LAI_safe_max。
    """
    t = row["veg_zone_primary"]
    safe_max = float(row["LAI_safe_max"])
    if t == "保育/自然恢复":
        lower = 0.0
        upper = min(safe_max, 1.0)
    elif t == "低覆盖灌草":
        lower = 0.5
        upper = min(safe_max, 1.5)
    elif t == "灌草主导":
        lower = 1.5
        upper = min(safe_max, 3.0)
    elif t == "低密乔灌草":
        lower = 3.0
        upper = min(safe_max, 4.5)
    elif t == "乔灌草":
        lower = 4.5
        upper = safe_max
    else:
        lower = 0.0
        upper = safe_max
    # 保险处理
    lower = max(0.0, float(lower))
    upper = max(lower, float(upper))
    return lower, upper
def management_note(row: pd.Series) -> str:
    if row["constraint_mode"] == "core_conservation":
        return "核心保育单元，不进入增配乔木候选集"
    if row["Over_safe_capacity_flag"] == 1:
        return "当前已超过安全上限，优先降密/调结构/控灌"
    if row["obs_support_flag"] == "low":
        return "观测支持弱，建议标记为低可靠性单元"
    if row["veg_zone_primary"] == "乔灌草":
        return "可进入乔灌草候选集，但仍需后续物种与密度优化"
    if row["veg_zone_primary"] == "低密乔灌草":
        return "仅允许低密乔木配置，避免高耗水结构"
    if row["veg_zone_primary"] == "灌草主导":
        return "以灌草配置为主，不建议发展高乔木比例"
    if row["veg_zone_primary"] == "低覆盖灌草":
        return "宜采用低覆盖灌草或草本恢复"
    return "以保育和自然恢复为主"
# =============================================================================
# 4. 生成分类结果
# =============================================================================
out = df.copy()
out["veg_zone_primary"] = out.apply(classify_primary_type, axis=1)
out["allowed_modes"] = out["veg_zone_primary"].apply(allowed_modes)
out["tree_allowed"] = out["veg_zone_primary"].apply(tree_allowed)
intervals = out.apply(target_lai_interval, axis=1)
out["target_LAI_lower"] = [x[0] for x in intervals]
out["target_LAI_upper"] = [x[1] for x in intervals]
out["management_note"] = out.apply(management_note, axis=1)
# 一个简单的风险等级字段，方便制图
out["risk_level_simple"] = pd.cut(
    out["R_gde"],
    bins=[-np.inf, 0.25, 0.50, np.inf],
    labels=["低风险", "中风险", "高风险"]
).astype(str)
# 可靠性备注
out["reliability_note"] = out["obs_support_flag"].map({
    "high": "观测支持高",
    "medium": "观测支持中等",
    "low": "观测支持低"
}).fillna("未知")
# =============================================================================
# 5. 输出主表
# =============================================================================
main_cols = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "constraint_mode",
    "management_priority",
    "obs_support_flag",
    "reliability_note",
    "Area_km2",
    "GDE_frac_mean",
    "GDE_stability_mean",
    "GDE_persistence_mean",
    "GWSA_trend_mean",
    "R_gde",
    "risk_level_simple",
    "LAI_current_3yr_mean",
    "LAI_max_total",
    "LAI_safe_max",
    "Restoration_gap_safe",
    "Over_safe_capacity_flag",
    "veg_zone_primary",
    "allowed_modes",
    "tree_allowed",
    "target_LAI_lower",
    "target_LAI_upper",
    "management_note",
]
out_main = out[main_cols].copy()
out_main.to_csv(OUT_MAIN_CSV, index=False, encoding="utf-8-sig")
# =============================================================================
# 6. 输出汇总表
# =============================================================================
summary = (
    out.groupby("veg_zone_primary", as_index=False)
    .agg(
        Class_n=("ClassCode", "count"),
        Area_km2_sum=("Area_km2", "sum"),
        Area_km2_mean=("Area_km2", "mean"),
        LAI_safe_max_mean=("LAI_safe_max", "mean"),
        LAI_max_total_mean=("LAI_max_total", "mean"),
        R_gde_mean=("R_gde", "mean"),
        Over_safe_capacity_n=("Over_safe_capacity_flag", "sum"),
        Tree_allowed_n=("tree_allowed", "sum"),
    )
    .sort_values("Area_km2_sum", ascending=False)
    .reset_index(drop=True)
)
summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
# =============================================================================
# 7. 输出 GDE 等级 × 植被类型交叉表
# =============================================================================
crosstab = pd.crosstab(
    out["GDE_Level"],
    out["veg_zone_primary"],
    margins=True
)
crosstab.to_csv(OUT_CROSSTAB_CSV, encoding="utf-8-sig")
# =============================================================================
# 8. 终端输出
# =============================================================================
print("\n========== 分类结果摘要 ==========")
print(f"输出主表: {OUT_MAIN_CSV}")
print(f"输出汇总表: {OUT_SUMMARY_CSV}")
print(f"输出交叉表: {OUT_CROSSTAB_CSV}")
print("\n类型数量统计：")
print(out["veg_zone_primary"].value_counts())
print("\n类型面积统计（km²）：")
print(out.groupby("veg_zone_primary")["Area_km2"].sum().sort_values(ascending=False))
print("\n前5行预览：")
print(out_main.head())
