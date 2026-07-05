"""
03_compute_gde_safe_lai.py

Computes the GDE safety-reduction factor, GDE-constrained safe LAI, and safe-LAI retention ratio.

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
CLASS_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_LAImaxTotal_q25.csv"
OUT_SAFE_CSV = DATA_DIR / "ThreeNorth_Class_LAI_safe_max.csv"
OUT_SAFE_LEVEL_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_LAI_safe_max_byLevel.csv"
# ---- 风险权重（可在敏感性分析中调整） ----
W_GDE_FRAC = 0.30
W_GDE_STABILITY = 0.20
W_GDE_PERSISTENCE = 0.15
W_GWSA_DECLINE = 0.20
W_GDE_LEVEL = 0.15
# ---- 风险上限，避免 R_gde 过于极端 ----
RISK_MIN = 0.00
RISK_MAX = 0.80
# ---- 可靠性阈值 ----
LOW_SUPPORT_THRESHOLD = 1       # Nat_pixel_count_mean <= 0
MID_SUPPORT_THRESHOLD = 30      # Nat_pixel_count_mean < 30
# =============================================================================
# 1. 工具函数
# =============================================================================
def robust_minmax(
    s: pd.Series,
    q_low: float = 0.05,
    q_high: float = 0.95,
    reverse: bool = False,
) -> pd.Series:
    """
    稳健标准化到 0-1，使用 5%-95% 分位数截断。
    reverse=True 时做反向标准化。
    """
    x = s.astype(float).copy()
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
def map_gde_level_to_risk(level_series: pd.Series) -> pd.Series:
    """
    GDE_Level 风险映射：
    0 = 无约束 -> 0.00
    1 = 核心保育低敏感 -> 0.30 （后面会再叠加硬约束）
    2 = 中度敏感 -> 0.45
    3 = 高敏感约束 -> 0.70
    """
    risk_map = {
        0: 0.00,
        1: 0.30,
        2: 0.45,
        3: 0.70,
    }
    return level_series.map(risk_map).astype(float).fillna(0.0)
def reliability_flag(row: pd.Series) -> str:
    """
    根据观测支持情况给单元打可靠性标签
    """
    nat_cnt = row.get("Nat_pixel_count_mean", np.nan)
    obs_p95 = row.get("LAI_nat_max_p95_mean", np.nan)
    if pd.isna(obs_p95) or pd.isna(nat_cnt) or nat_cnt <= LOW_SUPPORT_THRESHOLD:
        return "low"
    elif nat_cnt < MID_SUPPORT_THRESHOLD:
        return "medium"
    else:
        return "high"
# =============================================================================
# 2. 读取输入
# =============================================================================
df = pd.read_csv(CLASS_SUMMARY_CSV)
print(f"输入表 shape: {df.shape}")
print("前5行：")
print(df.head())
# =============================================================================
# 3. QA 检查
# =============================================================================
print("\n========== QA 检查 ==========")
print(f"ClassCode 是否唯一: {not df['ClassCode'].duplicated().any()}")
print(f"LAI_max_total 缺失数: {df['LAI_max_total'].isna().sum()}")
print(f"LAI_max_total 负值数: {(df['LAI_max_total'] < 0).sum()}")
print(f"LAI_current_3yr_mean > LAI_max_total 的行数: {(df['LAI_current_3yr_mean'] > df['LAI_max_total']).sum()}")
print("\n缺失值最多字段：")
print(df.isna().sum().sort_values(ascending=False).head(10))
# =============================================================================
# 4. 构建风险分量
# =============================================================================
safe_df = df.copy()
# 4.1 GDE 覆盖风险
safe_df["risk_gde_frac"] = robust_minmax(safe_df["GDE_frac_mean"])
# 4.2 GDE 稳定性风险
safe_df["risk_gde_stability"] = robust_minmax(safe_df["GDE_stability_mean"])
# 4.3 GDE 持续性风险
safe_df["risk_gde_persistence"] = robust_minmax(safe_df["GDE_persistence_mean"])
# 4.4 总水储量下降风险：只对“下降”部分计风险
safe_df["GWSA_decline"] = (-safe_df["GWSA_trend_mean"]).clip(lower=0)
safe_df["risk_gwsa_decline"] = robust_minmax(safe_df["GWSA_decline"])
# 4.5 GDE 等级风险
safe_df["risk_gde_level"] = map_gde_level_to_risk(safe_df["GDE_Level"])
# =============================================================================
# 5. 组合成 R_gde
# =============================================================================
safe_df["R_gde_raw"] = (
    W_GDE_FRAC * safe_df["risk_gde_frac"]
    + W_GDE_STABILITY * safe_df["risk_gde_stability"]
    + W_GDE_PERSISTENCE * safe_df["risk_gde_persistence"]
    + W_GWSA_DECLINE * safe_df["risk_gwsa_decline"]
    + W_GDE_LEVEL * safe_df["risk_gde_level"]
)
safe_df["R_gde"] = safe_df["R_gde_raw"].clip(lower=RISK_MIN, upper=RISK_MAX)
# 风险比例
safe_df["safe_ratio_pre"] = 1.0 - safe_df["R_gde"]
# =============================================================================
# 6. 先算无硬约束的安全上限
# =============================================================================
safe_df["LAI_safe_pre"] = safe_df["LAI_max_total"] * safe_df["safe_ratio_pre"]
# =============================================================================
# 7. 对核心保育单元加硬约束
# =============================================================================
core_mask = safe_df["GDE_Level"] == 1
safe_df["LAI_safe_max"] = safe_df["LAI_safe_pre"]
# 核心保育类：安全上限不超过当前 LAI
safe_df.loc[core_mask, "LAI_safe_max"] = np.minimum(
    safe_df.loc[core_mask, "LAI_safe_pre"],
    safe_df.loc[core_mask, "LAI_current_3yr_mean"]
)
# 下界保护
safe_df["LAI_safe_max"] = safe_df["LAI_safe_max"].clip(lower=0.0)
# 最终安全比例
safe_df["safe_ratio_final"] = np.where(
    safe_df["LAI_max_total"] > 0,
    safe_df["LAI_safe_max"] / safe_df["LAI_max_total"],
    np.nan
)
# =============================================================================
# 8. 生成管理与解释字段
# =============================================================================
# 8.1 约束模式
def constraint_mode(level: int) -> str:
    if level == 0:
        return "no_constraint"
    elif level == 1:
        return "core_conservation"
    elif level == 2:
        return "moderate_constraint"
    elif level == 3:
        return "high_constraint"
    else:
        return "unknown"
safe_df["constraint_mode"] = safe_df["GDE_Level"].apply(constraint_mode)
# 8.2 安全恢复空间（正值表示还能增加，负值表示已超安全上限）
safe_df["Restoration_gap_safe"] = safe_df["LAI_safe_max"] - safe_df["LAI_current_3yr_mean"]
# 8.3 是否超安全上限
safe_df["Over_safe_capacity_flag"] = (safe_df["Restoration_gap_safe"] < 0).astype(int)
# 8.4 观测支持等级
safe_df["obs_support_flag"] = safe_df.apply(reliability_flag, axis=1)
# 8.5 可选：建议管理优先级
def management_priority(row: pd.Series) -> str:
    if row["constraint_mode"] == "core_conservation":
        return "conservation_first"
    elif row["Over_safe_capacity_flag"] == 1 and row["constraint_mode"] in ["moderate_constraint", "high_constraint"]:
        return "strict_control"
    elif row["constraint_mode"] == "high_constraint":
        return "risk_avoidance"
    elif row["constraint_mode"] == "moderate_constraint":
        return "cautious_restoration"
    else:
        return "restoration_possible"
safe_df["management_priority"] = safe_df.apply(management_priority, axis=1)
# =============================================================================
# 9. 列整理
# =============================================================================
ordered_cols = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "constraint_mode",
    "management_priority",
    "obs_support_flag",
    "Area_km2",
    "GDE_frac_mean",
    "GDE_stability_mean",
    "GDE_persistence_mean",
    "GWSA_trend_mean",
    "LAI_current_3yr_mean",
    "LAI_cap_pred_q25",
    "LAI_cap_pred_q50",
    "LAI_cap_pred_q75",
    "LAI_cap_pred_mean",
    "LAI_cap_pred_min",
    "LAI_cap_pred_max",
    "LAI_max_total",
    "risk_gde_frac",
    "risk_gde_stability",
    "risk_gde_persistence",
    "GWSA_decline",
    "risk_gwsa_decline",
    "risk_gde_level",
    "R_gde_raw",
    "R_gde",
    "safe_ratio_pre",
    "LAI_safe_pre",
    "LAI_safe_max",
    "safe_ratio_final",
    "Restoration_gap_safe",
    "Over_safe_capacity_flag",
    "LAI_nat_max_p95_mean",
    "Nat_pixel_count_mean",
]
safe_df = safe_df[ordered_cols]
# =============================================================================
# 10. 输出
# =============================================================================
safe_df.to_csv(OUT_SAFE_CSV, index=False, encoding="utf-8-sig")
level_summary = (
    safe_df.groupby(["GDE_Level", "constraint_mode"], as_index=False)
    .agg(
        Class_n=("ClassCode", "count"),
        Area_km2_sum=("Area_km2", "sum"),
        R_gde_mean=("R_gde", "mean"),
        LAI_max_total_mean=("LAI_max_total", "mean"),
        LAI_safe_max_mean=("LAI_safe_max", "mean"),
        Safe_ratio_mean=("safe_ratio_final", "mean"),
        Over_safe_capacity_n=("Over_safe_capacity_flag", "sum"),
    )
)
level_summary.to_csv(OUT_SAFE_LEVEL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
# =============================================================================
# 11. 终端输出
# =============================================================================
print("\n========== 结果摘要 ==========")
print(f"输出主表: {OUT_SAFE_CSV}")
print(f"输出分级汇总表: {OUT_SAFE_LEVEL_SUMMARY_CSV}")
print("\nR_gde 描述统计：")
print(safe_df["R_gde"].describe())
print("\nLAI_max_total 描述统计：")
print(safe_df["LAI_max_total"].describe())
print("\nLAI_safe_max 描述统计：")
print(safe_df["LAI_safe_max"].describe())
print("\n按观测支持等级统计：")
print(safe_df["obs_support_flag"].value_counts(dropna=False))
print("\n按管理优先级统计：")
print(safe_df["management_priority"].value_counts(dropna=False))
print("\n前5行预览：")
print(safe_df.head())
