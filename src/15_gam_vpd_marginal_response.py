"""
15_gam_vpd_marginal_response.py

Fits spline-based GAM-style marginal responses and compares hydroclimate models with and without VPD.

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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
# ============================================================
# 0. 路径区
# ============================================================
ROOT = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"
INPUT_CSV = ROOT / "S9_Hydroclimate_GAM_Input_ByClass_CHECKED.csv"
OUT_SUMMARY = ROOT / "S9_GAM_Model_Compare.csv"
OUT_SAFE_NOVPD = ROOT / "S9_GAM_Main_NoVPD_safe_LAI_ratio.png"
OUT_TREE_NOVPD = ROOT / "S9_GAM_Main_NoVPD_tree_ratio_best.png"
OUT_SAFE_WITHVPD = ROOT / "S9_GAM_WithVPD_safe_LAI_ratio.png"
OUT_TREE_WITHVPD = ROOT / "S9_GAM_WithVPD_tree_ratio_best.png"
# ============================================================
# 0.1 图形样式参数：字号主要在这里调
# ============================================================
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
    "axes.linewidth": 1.2,
})
# 横轴显示名称。想换中文，直接改这里。
DISPLAY_LABELS = {
    "P_20yr": "Precipitation (P)",
    "PET_20yr": "PET",
    "AET_P": "AET/P",
    "Soil_20yr": "Soil moisture",
    "VPD_20yr": "VPD",
    "safe_LAI_ratio": "Safe LAI ratio",
    "tree_ratio_best": "Tree ratio",
}
# 图例名称。想换中文，直接改这里。
LEGEND_LABELS = {
    "points": "ClassCode optimization units",
    "line": "GAM-fitted marginal response",
    "ci": "95% confidence interval",
}
# ============================================================
# 1. 读取数据
# ============================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV.resolve()}")
df = pd.read_csv(INPUT_CSV)
MAIN_PREDICTORS = ["P_20yr", "PET_20yr", "AET_P", "Soil_20yr"]
WITH_VPD_PREDICTORS = ["P_20yr", "PET_20yr", "AET_P", "Soil_20yr", "VPD_20yr"]
RESPONSES = ["safe_LAI_ratio", "tree_ratio_best"]
# 控制变量。这里不把BaseZone作为58个哑变量强行塞入，避免208样本下过拟合。
# GDE变量用于控制GDE强度，真正的GDE×climate交互可在下一步单独做。
CONTROL_VARS = [c for c in ["GDE_frac_mean", "GDE_stability_mean"] if c in df.columns]
# ============================================================
# 2. 数据清理和模型函数
# ============================================================
def clean_data(data: pd.DataFrame, predictors: list[str], response: str) -> pd.DataFrame:
    need = predictors + CONTROL_VARS + [response]
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise KeyError(f"输入表缺少字段：{missing}")
    d = data[need].copy()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().reset_index(drop=True)
    # 裁剪自变量极端值，避免曲线被1-2个点拉坏
    for c in predictors:
        q01, q99 = d[c].quantile([0.01, 0.99])
        d[c] = d[c].clip(q01, q99)
    if response == "safe_LAI_ratio":
        d[response] = d[response].clip(0, 1.2)
    if response == "tree_ratio_best":
        d[response] = d[response].clip(0, 1.0)
    return d
def make_spline_design(d: pd.DataFrame, predictors: list[str], df_spline: int = 5):
    """
    用 patsy 的 bs() 构造 B-spline 设计矩阵，近似GAM。
    重要：保留 design_info，后续预测网格必须复用同一组 knots。
    """
    terms = [f"bs({x}, df={df_spline}, degree=3, include_intercept=False)" for x in predictors]
    for c in CONTROL_VARS:
        terms.append(c)
    formula = " + ".join(terms)
    X0 = dmatrix(formula, d, return_type="dataframe")
    design_info = X0.design_info
    X = sm.add_constant(X0, has_constant="add")
    return X, formula, design_info
def fit_model(data: pd.DataFrame, predictors: list[str], response: str, model_name: str):
    d = clean_data(data, predictors, response)
    X, formula, design_info = make_spline_design(d, predictors, df_spline=5)
    y = d[response]
    model = sm.OLS(y, X).fit()
    model._patsy_design_info = design_info
    row = {
        "model": model_name,
        "response": response,
        "n": len(d),
        "aic": model.aic,
        "bic": model.bic,
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "formula": formula,
    }
    return model, d, X, formula, row
def predict_grid(model, d: pd.DataFrame, predictors: list[str], formula: str, var: str, response: str):
    """
    其他变量固定中位数，只让var变化，生成边际响应。
    """
    q01, q99 = d[var].quantile([0.01, 0.99])
    xs = np.linspace(q01, q99, 120)
    base = {}
    for c in predictors + CONTROL_VARS:
        base[c] = d[c].median()
    grid = pd.DataFrame([base] * len(xs))
    grid[var] = xs
    # 复用训练阶段的 spline knots，避免每个预测网格重新估计基函数
    Xg0 = build_design_matrices([model._patsy_design_info], grid)[0]
    Xg = pd.DataFrame(Xg0, columns=model._patsy_design_info.column_names)
    Xg = sm.add_constant(Xg, has_constant="add")
    # 对齐训练矩阵列
    Xg = Xg.reindex(columns=model.model.exog_names, fill_value=0)
    pred = model.get_prediction(Xg).summary_frame(alpha=0.05)
    return xs, pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(), pred["mean_ci_upper"].to_numpy()
# ============================================================
# 3. 作图函数：NoVPD = 2×2，WithVPD = 横向排列
# ============================================================
def plot_marginals(data: pd.DataFrame, predictors: list[str], response: str, model_name: str, out_fig: Path):
    model, d, X, formula, row = fit_model(data, predictors, response, model_name)
    # NoVPD 主文图：4个变量，做四象限 2×2
    if model_name == "NoVPD" and len(predictors) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8), squeeze=False)
        axes = axes.ravel()
        legend_bottom = True
    else:
        # WithVPD 扩展图：仍然横向排，主要作为补充图
        fig, axes = plt.subplots(1, len(predictors), figsize=(5.2 * len(predictors), 4.8), squeeze=False)
        axes = axes[0]
        legend_bottom = True
    legend_handles = None
    panel_letters = list("abcdefghijklmnopqrstuvwxyz")
    for i, (ax, var) in enumerate(zip(axes, predictors)):
        xs, mean, low, high = predict_grid(model, d, predictors, formula, var, response)
        scatter_h = ax.scatter(
            d[var],
            d[response],
            s=34,
            alpha=0.50,
            edgecolors="none",
        )
        line_h, = ax.plot(xs, mean, linewidth=3.0)
        band_h = ax.fill_between(xs, low, high, alpha=0.22)
        if legend_handles is None:
            legend_handles = [scatter_h, line_h, band_h]
        ax.set_xlabel(DISPLAY_LABELS.get(var, var), labelpad=8)
        ax.set_ylabel(DISPLAY_LABELS.get(response, response), labelpad=8)
        # 左上角加面板编号，方便论文排版引用
        ax.set_title(f"({panel_letters[i]}) {DISPLAY_LABELS.get(var, var)}", loc="left", pad=10)
        ax.tick_params(axis="both", which="major", length=5, width=1.1)
        ax.grid(False)
    # 如果是横向图，可能 axes 数量刚好；这里保险隐藏多余坐标轴
    for j in range(len(predictors), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{model_name}: {DISPLAY_LABELS.get(response, response)}", y=0.985)
    if legend_bottom and legend_handles is not None:
        fig.legend(
            legend_handles,
            [LEGEND_LABELS["points"], LEGEND_LABELS["line"], LEGEND_LABELS["ci"]],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            handlelength=2.8,
            columnspacing=1.8,
        )
    # 给底部图例留空间
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return row
# ============================================================
# 4. 拟合与作图
# ============================================================
rows = []
for response in RESPONSES:
    out_fig = OUT_SAFE_NOVPD if response == "safe_LAI_ratio" else OUT_TREE_NOVPD
    rows.append(plot_marginals(df, MAIN_PREDICTORS, response, "NoVPD", out_fig))
    if df["VPD_20yr"].notna().sum() >= 40:
        out_fig_vpd = OUT_SAFE_WITHVPD if response == "safe_LAI_ratio" else OUT_TREE_WITHVPD
        rows.append(plot_marginals(df, WITH_VPD_PREDICTORS, response, "WithVPD", out_fig_vpd))
summary = pd.DataFrame(rows)
# VPD纳入主文判断
decisions = []
for response in RESPONSES:
    sub = summary[summary["response"] == response].set_index("model")
    if "NoVPD" not in sub.index or "WithVPD" not in sub.index:
        continue
    delta_aic = sub.loc["WithVPD", "aic"] - sub.loc["NoVPD", "aic"]
    delta_r2 = sub.loc["WithVPD", "r2"] - sub.loc["NoVPD", "r2"]
    if delta_aic <= -2 and delta_r2 >= 0.02:
        rec = "VPD明显改善，可放主文第5列"
    elif delta_r2 >= 0.01:
        rec = "VPD轻微改善，建议放补充材料"
    else:
        rec = "VPD改善不明显，建议放补充材料或不重点解释"
    decisions.append({
        "response": response,
        "delta_aic_WithVPD_minus_NoVPD": delta_aic,
        "delta_r2_WithVPD_minus_NoVPD": delta_r2,
        "recommendation": rec
    })
decisions = pd.DataFrame(decisions)
summary = summary.merge(decisions, on="response", how="left")
summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
print("GAM机制图与VPD比较完成。")
print(summary.to_string(index=False))
print(f"输出模型比较表：{OUT_SUMMARY.resolve()}")
