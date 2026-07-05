"""
02_model_hydroclimatic_lai_capacity.py

Fits the hydroclimatic LAI carrying-capacity model and summarizes the q25 LAI capacity by ClassCode.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# 0. 参数区
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

PANEL_CSV = DATA_DIR / "ThreeNorth_ClassYear_Table_Full_2005_2024.csv"
YEAR_SUMMARY_CSV = DATA_DIR / "ThreeNorth_ClassYear_YearSummary_2005_2024.csv"

OUT_FILTERED_CSV = DATA_DIR / "ThreeNorth_ModelInput_Filtered.csv"
OUT_PANEL_PRED_CSV = DATA_DIR / "ThreeNorth_ClassYear_With_LAIcapPred.csv"
OUT_CLASS_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_LAImaxTotal_q25.csv"
OUT_METRICS_TXT = DATA_DIR / "ThreeNorth_LAImaxTotal_ModelMetrics.txt"

# 过滤阈值：建议先用 30；如果你后面想更保守，可以改成 50
MIN_NAT_PIXEL_COUNT = 30

# 分位数模型：0.95 对应“上包络”
QUANTILE_ALPHA = 0.95

# 时间保守阈值：q25 作为“最大可持续总 LAI”
SUSTAINABLE_Q = 0.25

RANDOM_STATE = 42


# =============================================================================
# 1. 读取数据
# =============================================================================

panel = pd.read_csv(PANEL_CSV)
year_summary = pd.read_csv(YEAR_SUMMARY_CSV)

print(f"面板表 shape: {panel.shape}")
print(f"年份状态表 shape: {year_summary.shape}")


# =============================================================================
# 2. QA 检查
# =============================================================================

def qa_report(df: pd.DataFrame, ys: pd.DataFrame) -> None:
    print("\n========== QA 检查 ==========")
    print(f"唯一年份数: {df['Year'].nunique()}")
    print(f"唯一 ClassCode 数: {df['ClassCode'].nunique()}")
    print(f"Year-ClassCode 重复行数: {df.duplicated(subset=['Year', 'ClassCode']).sum()}")

    print("\n每年行数：")
    print(df.groupby('Year').size().describe())

    print("\n每个 ClassCode 行数：")
    print(df.groupby('ClassCode').size().describe())

    print("\n缺失值最多的字段：")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nNat_pixel_count 描述统计：")
    print(df["Nat_pixel_count"].describe())

    print("\nLAI_nat_max_p95 描述统计：")
    print(df["LAI_nat_max_p95"].describe())

    print("\n年份状态表：")
    print(ys.head())
    print(ys["Status"].value_counts(dropna=False))


qa_report(panel, year_summary)


# =============================================================================
# 3. 只保留有效年份
# =============================================================================

ok_years = year_summary.loc[year_summary["Status"] == "ok", "Year"].tolist()
panel = panel[panel["Year"].isin(ok_years)].copy()

print(f"\n保留 ok 年份后 shape: {panel.shape}")


# =============================================================================
# 4. 定义目标变量与特征
# =============================================================================

# ---- 目标变量：自然植被观测上限 ----
TARGET = "LAI_nat_max_p95"

# ---- 这一阶段只拟合“最大可持续总 LAI”，不用 GDE 风险字段 ----
# 不放入：
# 1) GDE_frac_mean / GDE_stability_mean / GDE_persistence_mean / GDE_Level
#    因为这些更适合留到下一步构建 GDE 风险函数
# 2) LAI_current_3yr_mean / LAI_mean / LAI_max_mean / LAI_nat_mean
#    因为这些属于植被状态变量，会造成信息泄漏

NUM_FEATURES = [
    "P_mean",
    "PET_mean",
    "AET_mean",
    "Runoff_mean",
    "Soil_mean",
    "Tmean_mean",
    "GWSA_mean_period",
    "GWSA_trend_mean",
    "Elevation_mean",
    "Slope_mean",
    "AI_prelim_noGW_mean",
    "AI_with_GWSA_proxy_mean",
]

CAT_FEATURES = [
    "BaseZone",  # 基础生态水文区可作为分层/类别背景
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# =============================================================================
# 5. 过滤可建模样本
# =============================================================================

model_df = panel.copy()

# 目标变量与样本量过滤
model_df = model_df[
    model_df[TARGET].notna() &
    (model_df["Nat_pixel_count"] >= MIN_NAT_PIXEL_COUNT)
].copy()

# 解释变量缺失过滤
model_df = model_df.dropna(subset=ALL_FEATURES)

# 明显异常过滤（保守）
model_df = model_df[
    (model_df[TARGET] > 0) &
    (model_df[TARGET] < 10)
].copy()

print(f"\n进入建模样本数: {len(model_df)}")
print(f"过滤后唯一年份数: {model_df['Year'].nunique()}")
print(f"过滤后唯一 ClassCode 数: {model_df['ClassCode'].nunique()}")

model_df.to_csv(OUT_FILTERED_CSV, index=False, encoding="utf-8-sig")
print(f"已保存过滤后建模样本: {OUT_FILTERED_CSV}")


# =============================================================================
# 6. 建立分位数回归模型（年度 LAI 上限）
# =============================================================================

X = model_df[ALL_FEATURES].copy()
y = model_df[TARGET].copy()
groups = model_df["Year"].copy()

# sklearn 兼容写法
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), CAT_FEATURES),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), NUM_FEATURES),
    ],
    remainder="drop"
)

# 用 quantile loss 拟合 0.95 分位数上包络
model = GradientBoostingRegressor(
    loss="quantile",
    alpha=QUANTILE_ALPHA,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=RANDOM_STATE,
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model),
])


# =============================================================================
# 7. 先做一个按年份分组的验证
# =============================================================================

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
year_train = model_df["Year"].iloc[train_idx].unique().tolist()
year_test = model_df["Year"].iloc[test_idx].unique().tolist()

pipeline.fit(X_train, y_train)
y_pred_test = pipeline.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, None)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
r2 = r2_score(y_test, y_pred_test)
pinball = mean_pinball_loss(y_test, y_pred_test, alpha=QUANTILE_ALPHA)

metrics_text = []
metrics_text.append("=== 年份分组验证 ===")
metrics_text.append(f"训练年份: {sorted(year_train)}")
metrics_text.append(f"测试年份: {sorted(year_test)}")
metrics_text.append(f"样本数(train/test): {len(X_train)} / {len(X_test)}")
metrics_text.append(f"MAE: {mae:.4f}")
metrics_text.append(f"RMSE: {rmse:.4f}")
metrics_text.append(f"R2: {r2:.4f}")
metrics_text.append(f"PinballLoss(alpha={QUANTILE_ALPHA}): {pinball:.4f}")

print("\n".join(metrics_text))


# =============================================================================
# 8. 用全部可建模样本重新训练最终模型
# =============================================================================

pipeline.fit(X, y)

# 对全表进行年度上限预测
pred_df = panel.copy()
pred_input = pred_df[ALL_FEATURES].copy()

# 若某些行特征缺失，pipeline 内部会做插补
pred_df["LAI_cap_pred_annual"] = np.clip(pipeline.predict(pred_input), 0, None)

# 保存带预测值的面板表
pred_df.to_csv(OUT_PANEL_PRED_CSV, index=False, encoding="utf-8-sig")
print(f"\n已保存年度预测面板: {OUT_PANEL_PRED_CSV}")


# =============================================================================
# 9. 从年度上限收缩成“最大可持续总 LAI”
# =============================================================================

# 对每个 ClassCode，取年度预测上限的时间下四分位数
class_summary = (
    pred_df.groupby("ClassCode", as_index=False)
    .agg(
        BaseZone=("BaseZone", "first"),
        GDE_Level=("GDE_Level", "first"),
        Area_km2=("Area_km2", "first"),
        GDE_frac_mean=("GDE_frac_mean", "first"),
        GDE_stability_mean=("GDE_stability_mean", "first"),
        GDE_persistence_mean=("GDE_persistence_mean", "first"),
        GWSA_trend_mean=("GWSA_trend_mean", "first"),
        LAI_current_3yr_mean=("LAI_current_3yr_mean", "first"),
        LAI_cap_pred_q25=("LAI_cap_pred_annual", lambda s: float(np.nanquantile(s, SUSTAINABLE_Q))),
        LAI_cap_pred_q50=("LAI_cap_pred_annual", lambda s: float(np.nanquantile(s, 0.50))),
        LAI_cap_pred_q75=("LAI_cap_pred_annual", lambda s: float(np.nanquantile(s, 0.75))),
        LAI_cap_pred_mean=("LAI_cap_pred_annual", "mean"),
        LAI_cap_pred_min=("LAI_cap_pred_annual", "min"),
        LAI_cap_pred_max=("LAI_cap_pred_annual", "max"),
        LAI_nat_max_p95_mean=("LAI_nat_max_p95", "mean"),
        Nat_pixel_count_mean=("Nat_pixel_count", "mean"),
    )
    .sort_values("ClassCode")
    .reset_index(drop=True)
)

# 这就是你要的“最大可持续总 LAI”
class_summary["LAI_max_total"] = class_summary["LAI_cap_pred_q25"]

class_summary.to_csv(OUT_CLASS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
print(f"已保存 ClassCode 汇总表: {OUT_CLASS_SUMMARY_CSV}")


# =============================================================================
# 10. 保存评估信息
# =============================================================================

with open(OUT_METRICS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(metrics_text))

print(f"已保存模型评估信息: {OUT_METRICS_TXT}")


# =============================================================================
# 11. 终端摘要
# =============================================================================

print("\n========== 结果摘要 ==========")
print(f"面板总行数: {len(panel)}")
print(f"建模样本数: {len(model_df)}")
print(f"LAI_max_total 输出 ClassCode 数: {len(class_summary)}")

print("\nLAI_max_total 描述统计：")
print(class_summary["LAI_max_total"].describe())

print("\n前5行预览：")
print(class_summary.head())
