"""
12_gde_threshold_sensitivity.py

Tests sensitivity of restoration outputs to alternative GDE fraction thresholds.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径区：只改这里
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

OUT_DIR = DATA_DIR / "GDE_threshold_sensitivity_outputs_FIXED"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BYCLASS = OUT_DIR / "S6a_GDE_threshold_sensitivity_byClass_FIXED.csv"
OUT_SUMMARY = OUT_DIR / "S6a_GDE_threshold_sensitivity_summary_FIXED.csv"
OUT_TRANSITION = OUT_DIR / "S6a_GDE_threshold_sensitivity_transition_area_FIXED.csv"
OUT_LEVEL_CHANGE = OUT_DIR / "S6a_GDE_level_change_matrix_area_FIXED.csv"
OUT_TEMPLATES = OUT_DIR / "S6a_mode_density_templates_FIXED.csv"
OUT_REPORT = OUT_DIR / "S6a_GDE_threshold_sensitivity_report_FIXED.md"


# ============================================================
# 1. 敏感性情景
# ============================================================

SCENARIOS = {
    "baseline_05_30_60": {"low": 0.05, "mid": 0.30, "high": 0.60},
    "T1_relaxed_high_10_30_50": {"low": 0.10, "mid": 0.30, "high": 0.50},
    "T2_lower_mid_high_05_20_50": {"low": 0.05, "mid": 0.20, "high": 0.50},
    "T3_strict_low_10_30_60": {"low": 0.10, "mid": 0.30, "high": 0.60},
    "T4_mid_shift_05_25_55": {"low": 0.05, "mid": 0.25, "high": 0.55},
}

BASELINE_SCENARIO = "baseline_05_30_60"
RGDE_MAX = 0.95


# ============================================================
# 2. 文件匹配规则
# ============================================================

FILE_PATTERNS = {
    "hydro": [
        "ThreeNorth_Class_HydroSupport_2005_2024*.csv",
    ],
    "lai_total": [
        "ThreeNorth_Class_LAImaxTotal_q25*.csv",
        "*LAImaxTotal*q25*.csv",
    ],
    "lai_safe": [
        "ThreeNorth_Class_LAI_safe_max*.csv",
        "*LAI_safe_max*.csv",
    ],
    "mode_impl": [
        "ThreeNorth_Class_ModeImplementation*.csv",
        "*ModeImplementation*.csv",
    ],
    "mode_map": [
        "ThreeNorth_Class_ModeMapReady*.csv",
        "Fig6_ModeMapReady*.csv",
        "*ModeMapReady*.csv",
    ],
    "best": [
        "ThreeNorth_Class_MOO_BestCompromise*.csv",
        "*BestCompromise*.csv",
    ],
}


# ============================================================
# 3. 工具函数
# ============================================================

def find_latest(patterns: list[str]) -> Path | None:
    files = []
    for pat in patterns:
        files.extend(DATA_DIR.glob(pat))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not files:
        return None
    return files[-1]


def read_required(name: str) -> pd.DataFrame:
    path = find_latest(FILE_PATTERNS[name])
    if path is None:
        raise FileNotFoundError(f"未找到 {name} 文件。匹配规则: {FILE_PATTERNS[name]}")

    print(f"[READ] {name}: {path.name}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    df.attrs["source_path"] = str(path)
    return df


def read_optional(name: str) -> pd.DataFrame | None:
    path = find_latest(FILE_PATTERNS[name])
    if path is None:
        print(f"[MISSING] {name}: 未找到文件")
        return None

    print(f"[READ] {name}: {path.name}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    df.attrs["source_path"] = str(path)
    return df


def to_numeric_if_exists(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_col(df: pd.DataFrame | None, candidates: list[str], required: bool = True) -> str | None:
    if df is None:
        if required:
            raise KeyError(f"DataFrame 为空，无法查找字段: {candidates}")
        return None

    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(f"找不到字段，候选字段为: {candidates}")

    return None


def normalize_mode(x: object) -> str:
    s = str(x)

    if re.search("Close|近自然", s, flags=re.IGNORECASE):
        return "Close-to-nature stand management"

    if re.search("Herbaceous|exclosure|草本|封育|自然恢复|保育", s, flags=re.IGNORECASE):
        return "Herbaceous–exclosure recovery"

    if re.search("Tree|乔", s, flags=re.IGNORECASE) and re.search("shrub|灌", s, flags=re.IGNORECASE):
        return "Tree–shrub–grass restoration"

    if re.search("Shrub|灌草|灌", s, flags=re.IGNORECASE):
        return "Shrub–grass restoration"

    return "Unknown"


def assign_gde_level(gde_frac: float, low: float, mid: float, high: float) -> int | float:
    """
    GDE_Level 规则：
    < low      -> 0 非 GDE
    low-mid    -> 3 破碎型高风险 GDE
    mid-high   -> 2 中度 GDE 约束
    >= high    -> 1 GDE 主导保育
    """
    if pd.isna(gde_frac):
        return np.nan

    if gde_frac < low:
        return 0
    if gde_frac >= high:
        return 1
    if gde_frac >= mid:
        return 2
    return 3


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def median_iqr_string(x: pd.Series, digits: int = 3) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return f"{med:.{digits}f} [{q25:.{digits}f}–{q75:.{digits}f}]"


def infer_mode_from_lai_and_level(
    lai_safe: float,
    gde_level: int,
    base_mode: str,
    lai_current: float | None = None,
) -> str:
    """
    诊断性规则，不替代完整 MOO。
    用于检验 GDE 阈值改变是否会改变最终模式方向。
    """
    if pd.isna(lai_safe):
        return "Herbaceous–exclosure recovery"

    if gde_level == 1:
        return "Herbaceous–exclosure recovery"

    if gde_level == 3:
        if lai_safe >= 1.5:
            return "Shrub–grass restoration"
        return "Herbaceous–exclosure recovery"

    if base_mode == "Close-to-nature stand management":
        if lai_current is not None and pd.notna(lai_current) and lai_current >= 2.5:
            return "Close-to-nature stand management"
        if lai_safe >= 3.0:
            return "Close-to-nature stand management"

    if lai_safe >= 3.0:
        return "Tree–shrub–grass restoration"
    if lai_safe >= 1.5:
        return "Shrub–grass restoration"
    return "Herbaceous–exclosure recovery"


def build_templates(df: pd.DataFrame) -> pd.DataFrame:
    """
    从主情景 Mode 表中提取各模式的结构模板。
    用于把敏感性情景中发生 GDE_Level 变化的单元转成 target_LAI、ratio 和 density。
    """
    tmp = df.copy()

    needed = [
        "mode_base",
        "target_LAI_base",
        "tree_ratio_base",
        "shrub_ratio_base",
        "grass_ratio_base",
        "tree_density_base",
        "shrub_density_base",
        "grass_cover_base",
    ]

    for c in needed:
        if c not in tmp.columns:
            tmp[c] = np.nan

    tmp = tmp.loc[tmp["target_LAI_base"].notna() & (tmp["target_LAI_base"] > 0)].copy()

    tmp["tree_density_per_LAI"] = tmp["tree_density_base"] / tmp["target_LAI_base"].replace(0, np.nan)
    tmp["shrub_density_per_LAI"] = tmp["shrub_density_base"] / tmp["target_LAI_base"].replace(0, np.nan)

    rows = []

    for mode, g in tmp.groupby("mode_base"):
        rows.append({
            "mode": mode,
            "n": len(g),
            "tree_ratio": g["tree_ratio_base"].median(),
            "shrub_ratio": g["shrub_ratio_base"].median(),
            "grass_ratio": g["grass_ratio_base"].median(),
            "tree_density_per_LAI": g["tree_density_per_LAI"].median(),
            "shrub_density_per_LAI": g["shrub_density_per_LAI"].median(),
            "grass_cover": g["grass_cover_base"].median(),
        })

    templates = pd.DataFrame(rows)

    def ensure(mode: str, defaults: dict):
        nonlocal templates
        if len(templates) == 0 or mode not in set(templates["mode"].astype(str)):
            templates = pd.concat(
                [templates, pd.DataFrame([{"mode": mode, "n": 0, **defaults}])],
                ignore_index=True,
            )

    ensure(
        "Tree–shrub–grass restoration",
        {
            "tree_ratio": 0.20,
            "shrub_ratio": 0.50,
            "grass_ratio": 0.30,
            "tree_density_per_LAI": 80.0,
            "shrub_density_per_LAI": 800.0,
            "grass_cover": 95.0,
        },
    )

    ensure(
        "Shrub–grass restoration",
        {
            "tree_ratio": 0.00,
            "shrub_ratio": 0.65,
            "grass_ratio": 0.35,
            "tree_density_per_LAI": 0.0,
            "shrub_density_per_LAI": 1500.0,
            "grass_cover": 95.0,
        },
    )

    ensure(
        "Herbaceous–exclosure recovery",
        {
            "tree_ratio": 0.00,
            "shrub_ratio": 0.00,
            "grass_ratio": 1.00,
            "tree_density_per_LAI": 0.0,
            "shrub_density_per_LAI": 0.0,
            "grass_cover": 90.0,
        },
    )

    ensure(
        "Close-to-nature stand management",
        {
            "tree_ratio": 0.15,
            "shrub_ratio": 0.50,
            "grass_ratio": 0.35,
            "tree_density_per_LAI": 75.0,
            "shrub_density_per_LAI": 800.0,
            "grass_cover": 95.0,
        },
    )

    return templates


# ============================================================
# 4. 读取数据
# ============================================================

hydro = read_required("hydro")
lai_total = read_required("lai_total")
lai_safe = read_required("lai_safe")

mode_impl = read_optional("mode_impl")
mode_map = read_optional("mode_map")
best = read_optional("best")

mode_main = mode_impl if mode_impl is not None else mode_map
if mode_main is None:
    raise FileNotFoundError("没有找到 ModeImplementation 或 ModeMapReady。")


# ============================================================
# 5. 类型转换
# ============================================================

numeric_cols = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "_GDE_Level",
    "Area_km2",
    "area_km2",
    "Area",
    "GDE_frac",
    "GDE_frac_mean",
    "gde_frac",
    "gde_frac_mean",
    "GDE_fraction",
    "GDE_fraction_mean",
    "GDEFrac",
    "GDE_Frac",
    "GDE_stability",
    "GDE_stability_mean",
    "GDE_persistence",
    "GDE_persistence_mean",
    "GDE_persistence_count_mean",
    "GWSA_trend",
    "LAI_current_3yr",
    "LAI_current_3yr_mean",
    "LAI_max_total",
    "LAImax_total",
    "LAImaxTotal_q25",
    "LAI_max_total_q25",
    "LAI_safe_max",
    "LAI_safe_max_mean",
    "target_LAI",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
    "R_gde",
    "R_gde_raw",
    "safe_ratio",
    "safe_ratio_final",
]

for df0 in [hydro, lai_total, lai_safe, mode_main, best]:
    if df0 is not None:
        to_numeric_if_exists(df0, numeric_cols)


# ============================================================
# 6. 字段识别
# ============================================================

gde_frac_candidates = [
    "GDE_frac",
    "GDE_frac_mean",
    "gde_frac",
    "gde_frac_mean",
    "GDE_fraction",
    "GDE_fraction_mean",
    "GDEFrac",
    "GDE_Frac",
]

gde_frac_col = pick_col(hydro, gde_frac_candidates, required=False)

if gde_frac_col is None:
    gde_frac_col = pick_col(lai_safe, gde_frac_candidates, required=False)

if gde_frac_col is None:
    gde_frac_col = pick_col(lai_total, gde_frac_candidates, required=False)

if gde_frac_col is None:
    raise KeyError(
        "没有找到 GDE_frac / GDE_frac_mean 字段。请确认 HydroSupport、LAI_safe_max 或 LAImaxTotal 表中包含连续 GDE 面积占比字段。"
    )

print(f"[INFO] 使用 GDE fraction 字段: {gde_frac_col}")

lai_total_col = pick_col(
    lai_total,
    ["LAI_max_total", "LAImax_total", "LAImaxTotal_q25", "LAI_max_total_q25"],
    required=True,
)

lai_safe_col = pick_col(
    lai_safe,
    ["LAI_safe_max", "LAI_safe_max_mean"],
    required=True,
)

lai_current_col = pick_col(
    hydro,
    ["LAI_current_3yr", "LAI_current_3yr_mean"],
    required=False,
)

final_mode_col = pick_col(
    mode_main,
    ["final_mode", "FinalMode", "mode"],
    required=True,
)

area_col_mode = pick_col(
    mode_main,
    ["Area_km2", "area_km2", "Area"],
    required=False,
)

area_col_hydro = pick_col(
    hydro,
    ["Area_km2", "area_km2", "Area"],
    required=False,
)


# ============================================================
# 7. 构建基础表
# ============================================================

hydro_cols = ["ClassCode"]

for c in ["BaseZone", "GDE_Level"]:
    if c in hydro.columns:
        hydro_cols.append(c)

if area_col_hydro is not None and area_col_hydro in hydro.columns:
    hydro_cols.append(area_col_hydro)

if gde_frac_col in hydro.columns:
    hydro_cols.append(gde_frac_col)

if lai_current_col is not None and lai_current_col in hydro.columns:
    hydro_cols.append(lai_current_col)

hydro_cols = list(dict.fromkeys(hydro_cols))
base = hydro[hydro_cols].copy()

rename_base = {}

if "GDE_Level" in base.columns:
    rename_base["GDE_Level"] = "GDE_Level_base"

if area_col_hydro is not None and area_col_hydro in base.columns:
    rename_base[area_col_hydro] = "Area_km2"

if gde_frac_col in base.columns:
    rename_base[gde_frac_col] = "GDE_frac"

if lai_current_col is not None and lai_current_col in base.columns:
    rename_base[lai_current_col] = "LAI_current_3yr"

base = base.rename(columns=rename_base)

# 如果 GDE_frac 不在 HydroSupport，则从 LAI_safe 或 LAI_total 补
if "GDE_frac" not in base.columns:
    if gde_frac_col in lai_safe.columns:
        base = base.merge(
            lai_safe[["ClassCode", gde_frac_col]].rename(columns={gde_frac_col: "GDE_frac"}),
            on="ClassCode",
            how="left",
        )
    elif gde_frac_col in lai_total.columns:
        base = base.merge(
            lai_total[["ClassCode", gde_frac_col]].rename(columns={gde_frac_col: "GDE_frac"}),
            on="ClassCode",
            how="left",
        )

# 如果面积不在 HydroSupport，则从 Mode 表补
if "Area_km2" not in base.columns:
    if area_col_mode is None:
        raise KeyError("HydroSupport 和 Mode 表中都没有 Area_km2 / area_km2 / Area 字段。")
    base = base.merge(
        mode_main[["ClassCode", area_col_mode]].rename(columns={area_col_mode: "Area_km2"}),
        on="ClassCode",
        how="left",
    )

if "BaseZone" not in base.columns:
    base["BaseZone"] = np.nan

# 如果 GDE_Level_base 缺失，从 Mode 表补
if "GDE_Level_base" not in base.columns:
    if "GDE_Level" in mode_main.columns:
        base = base.merge(
            mode_main[["ClassCode", "GDE_Level"]].rename(columns={"GDE_Level": "GDE_Level_base"}),
            on="ClassCode",
            how="left",
        )
    else:
        raise KeyError("没有找到 GDE_Level 字段。")

# GDE_frac 单位检查：如果是 0–100 百分数，则转为 0–1
if "GDE_frac" in base.columns:
    base["GDE_frac"] = pd.to_numeric(base["GDE_frac"], errors="coerce")
    max_gde_frac = base["GDE_frac"].max()

    if pd.notna(max_gde_frac) and max_gde_frac > 1.5:
        print("[INFO] GDE_frac 看起来是 0–100 百分数，自动除以 100 转为 0–1。")
        base["GDE_frac"] = base["GDE_frac"] / 100.0

    print("[INFO] GDE_frac range:", base["GDE_frac"].min(), base["GDE_frac"].max())

missing_gde_frac = int(base["GDE_frac"].isna().sum())
if missing_gde_frac > 0:
    print(f"[WARNING] GDE_frac 缺失 {missing_gde_frac} / {len(base)} 条。")

# LAI total / safe
base = base.merge(
    lai_total[["ClassCode", lai_total_col]].rename(columns={lai_total_col: "LAImaxTotal"}),
    on="ClassCode",
    how="left",
)

base = base.merge(
    lai_safe[["ClassCode", lai_safe_col]].rename(columns={lai_safe_col: "LAI_safe_base"}),
    on="ClassCode",
    how="left",
)

# 主情景 mode 表
mode_cols = ["ClassCode", final_mode_col]

for c in [
    "target_LAI",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
    "optimized_flag",
    "backfill_reason",
]:
    if c in mode_main.columns:
        mode_cols.append(c)

mode_cols = list(dict.fromkeys(mode_cols))

mode_use = mode_main[mode_cols].copy()
mode_use = mode_use.rename(columns={
    final_mode_col: "mode_base",
    "target_LAI": "target_LAI_base",
    "tree_ratio": "tree_ratio_base",
    "shrub_ratio": "shrub_ratio_base",
    "grass_ratio": "grass_ratio_base",
    "tree_density_per_ha": "tree_density_base",
    "shrub_density_per_ha": "shrub_density_base",
    "grass_cover_target_pct": "grass_cover_base",
})

mode_use["mode_base"] = mode_use["mode_base"].map(normalize_mode)

base = base.merge(mode_use, on="ClassCode", how="left")

# 如果主情景 target_LAI 全缺失，尝试从 BestCompromise 补
if "target_LAI_base" in base.columns and base["target_LAI_base"].isna().all():
    if best is not None and "target_LAI" in best.columns:
        base = base.drop(columns=["target_LAI_base"])
        base = base.merge(
            best[["ClassCode", "target_LAI"]].rename(columns={"target_LAI": "target_LAI_base"}),
            on="ClassCode",
            how="left",
        )

for c in [
    "target_LAI_base",
    "tree_ratio_base",
    "shrub_ratio_base",
    "grass_ratio_base",
    "tree_density_base",
    "shrub_density_base",
    "grass_cover_base",
]:
    if c not in base.columns:
        base[c] = np.nan

# 基础 R_gde
base["R_base"] = 1 - base["LAI_safe_base"] / base["LAImaxTotal"].replace(0, np.nan)
base["R_base"] = base["R_base"].clip(0, RGDE_MAX)

# 如果 R_base 缺失，用 lai_safe 里的 R_gde 补
if "R_gde" in lai_safe.columns:
    r_tmp = lai_safe[["ClassCode", "R_gde"]].rename(columns={"R_gde": "R_gde_from_table"})
    base = base.merge(r_tmp, on="ClassCode", how="left")
    base["R_base"] = base["R_base"].fillna(base["R_gde_from_table"])
    base = base.drop(columns=["R_gde_from_table"])

fallback_R = {
    0: 0.17,
    1: 0.81,
    2: 0.43,
    3: 0.29,
}

observed_median_R = base.groupby("GDE_Level_base")["R_base"].median().to_dict()

for k, v in fallback_R.items():
    if k not in observed_median_R or pd.isna(observed_median_R[k]):
        observed_median_R[k] = v

base["R_level_median_base"] = base["GDE_Level_base"].map(observed_median_R)
base["R_residual"] = base["R_base"] - base["R_level_median_base"]
base["R_residual"] = base["R_residual"].fillna(0)

templates = build_templates(base)
templates.to_csv(OUT_TEMPLATES, index=False, encoding="utf-8-sig")
template_dict = templates.set_index("mode").to_dict(orient="index")


# ============================================================
# 8. 逐情景计算
# ============================================================

records = []

for scenario, th in SCENARIOS.items():
    low = th["low"]
    mid = th["mid"]
    high = th["high"]

    tmp = base.copy()
    tmp["scenario"] = scenario
    tmp["threshold_low"] = low
    tmp["threshold_mid"] = mid
    tmp["threshold_high"] = high

    if scenario == BASELINE_SCENARIO:
        # baseline 直接使用主情景，确保 baseline 与 base 完全一致
        tmp["GDE_Level_sensitivity"] = tmp["GDE_Level_base"]
        tmp["GDE_level_changed"] = False
        tmp["R_sensitivity"] = tmp["R_base"]
        tmp["LAI_safe_sensitivity"] = tmp["LAI_safe_base"]
        tmp["mode_sensitivity"] = tmp["mode_base"]

        tmp["target_LAI_sensitivity"] = tmp["target_LAI_base"]
        tmp["tree_ratio_sensitivity"] = tmp["tree_ratio_base"]
        tmp["shrub_ratio_sensitivity"] = tmp["shrub_ratio_base"]
        tmp["grass_ratio_sensitivity"] = tmp["grass_ratio_base"]
        tmp["tree_density_sensitivity"] = tmp["tree_density_base"]
        tmp["shrub_density_sensitivity"] = tmp["shrub_density_base"]
        tmp["grass_cover_sensitivity"] = tmp["grass_cover_base"]

    else:
        # 1. 重新分级
        tmp["GDE_Level_sensitivity"] = tmp["GDE_frac"].apply(
            lambda x: assign_gde_level(x, low, mid, high)
        )

        tmp["GDE_level_changed"] = tmp["GDE_Level_sensitivity"] != tmp["GDE_Level_base"]

        # 2. 重新计算 R_gde 和 LAI_safe
        tmp["R_level_median_sensitivity"] = tmp["GDE_Level_sensitivity"].map(observed_median_R)
        tmp["R_sensitivity"] = tmp["R_residual"] + tmp["R_level_median_sensitivity"]
        tmp["R_sensitivity"] = tmp["R_sensitivity"].clip(0, RGDE_MAX)

        tmp["LAI_safe_sensitivity"] = tmp["LAImaxTotal"] * (1 - tmp["R_sensitivity"])

        # GDE 主导保育硬约束：不超过当前 LAI
        if "LAI_current_3yr" in tmp.columns:
            mask_core = tmp["GDE_Level_sensitivity"].eq(1) & tmp["LAI_current_3yr"].notna()
            tmp.loc[mask_core, "LAI_safe_sensitivity"] = np.minimum(
                tmp.loc[mask_core, "LAI_safe_sensitivity"],
                tmp.loc[mask_core, "LAI_current_3yr"],
            )

        # 3. 根据新 GDE_Level 和新 LAI_safe 推导规则型模式
        tmp["mode_sensitivity"] = tmp.apply(
            lambda r: infer_mode_from_lai_and_level(
                lai_safe=r["LAI_safe_sensitivity"],
                gde_level=int(r["GDE_Level_sensitivity"]) if pd.notna(r["GDE_Level_sensitivity"]) else 0,
                base_mode=r["mode_base"],
                lai_current=r["LAI_current_3yr"] if "LAI_current_3yr" in tmp.columns else None,
            ),
            axis=1,
        )

        # 4. 对变化后的模式应用结构模板
        def apply_template(row: pd.Series) -> pd.Series:
            mode = row["mode_sensitivity"]
            tpl = template_dict.get(mode, template_dict["Herbaceous–exclosure recovery"])

            target = row["LAI_safe_sensitivity"]

            if pd.isna(target) or target < 0:
                return pd.Series({
                    "target_LAI_sensitivity": np.nan,
                    "tree_ratio_sensitivity": np.nan,
                    "shrub_ratio_sensitivity": np.nan,
                    "grass_ratio_sensitivity": np.nan,
                    "tree_density_sensitivity": np.nan,
                    "shrub_density_sensitivity": np.nan,
                    "grass_cover_sensitivity": np.nan,
                })

            tree_ratio = tpl["tree_ratio"]
            shrub_ratio = tpl["shrub_ratio"]
            grass_ratio = tpl["grass_ratio"]

            tree_density = target * tpl["tree_density_per_LAI"]
            shrub_density = target * tpl["shrub_density_per_LAI"]

            grass_cover = tpl["grass_cover"]
            if pd.isna(grass_cover):
                grass_cover = 95.0
            grass_cover = max(0.0, min(95.0, float(grass_cover)))

            return pd.Series({
                "target_LAI_sensitivity": target,
                "tree_ratio_sensitivity": tree_ratio,
                "shrub_ratio_sensitivity": shrub_ratio,
                "grass_ratio_sensitivity": grass_ratio,
                "tree_density_sensitivity": tree_density,
                "shrub_density_sensitivity": shrub_density,
                "grass_cover_sensitivity": grass_cover,
            })

        sens_struct = tmp.apply(apply_template, axis=1)
        tmp = pd.concat([tmp, sens_struct], axis=1)

        # ====================================================
        # 关键修正：
        # 如果 GDE 等级没有变化，则沿用主情景配置，
        # 避免模板重算造成 target_LAI / density 的伪差异。
        # ====================================================
        unchanged = ~tmp["GDE_level_changed"].fillna(False)

        tmp.loc[unchanged, "mode_sensitivity"] = tmp.loc[unchanged, "mode_base"]
        tmp.loc[unchanged, "LAI_safe_sensitivity"] = tmp.loc[unchanged, "LAI_safe_base"]

        tmp.loc[unchanged, "target_LAI_sensitivity"] = tmp.loc[unchanged, "target_LAI_base"]
        tmp.loc[unchanged, "tree_ratio_sensitivity"] = tmp.loc[unchanged, "tree_ratio_base"]
        tmp.loc[unchanged, "shrub_ratio_sensitivity"] = tmp.loc[unchanged, "shrub_ratio_base"]
        tmp.loc[unchanged, "grass_ratio_sensitivity"] = tmp.loc[unchanged, "grass_ratio_base"]
        tmp.loc[unchanged, "tree_density_sensitivity"] = tmp.loc[unchanged, "tree_density_base"]
        tmp.loc[unchanged, "shrub_density_sensitivity"] = tmp.loc[unchanged, "shrub_density_base"]
        tmp.loc[unchanged, "grass_cover_sensitivity"] = tmp.loc[unchanged, "grass_cover_base"]

    # 5. 计算变化量
    tmp["mode_changed"] = tmp["mode_sensitivity"] != tmp["mode_base"]
    tmp["area_mode_changed_km2"] = np.where(tmp["mode_changed"], tmp["Area_km2"], 0.0)
    tmp["area_level_changed_km2"] = np.where(tmp["GDE_level_changed"], tmp["Area_km2"], 0.0)

    tmp["delta_LAI_safe"] = tmp["LAI_safe_sensitivity"] - tmp["LAI_safe_base"]
    tmp["delta_target_LAI"] = tmp["target_LAI_sensitivity"] - tmp["target_LAI_base"]
    tmp["delta_tree_ratio"] = tmp["tree_ratio_sensitivity"] - tmp["tree_ratio_base"]
    tmp["delta_shrub_ratio"] = tmp["shrub_ratio_sensitivity"] - tmp["shrub_ratio_base"]
    tmp["delta_grass_ratio"] = tmp["grass_ratio_sensitivity"] - tmp["grass_ratio_base"]
    tmp["delta_tree_density"] = tmp["tree_density_sensitivity"] - tmp["tree_density_base"]
    tmp["delta_shrub_density"] = tmp["shrub_density_sensitivity"] - tmp["shrub_density_base"]
    tmp["delta_grass_cover"] = tmp["grass_cover_sensitivity"] - tmp["grass_cover_base"]

    records.append(tmp)

sens = pd.concat(records, ignore_index=True)


# ============================================================
# 9. 汇总统计
# ============================================================

def summarize_scenario(g: pd.DataFrame) -> pd.Series:
    area = g["Area_km2"]
    total_area = area.sum()

    return pd.Series({
        "n_units": len(g),
        "area_km2": total_area,

        "GDE_level_changed_area_km2": g["area_level_changed_km2"].sum(),
        "GDE_level_changed_area_pct": 100 * g["area_level_changed_km2"].sum() / total_area if total_area > 0 else np.nan,

        "mode_changed_area_km2": g["area_mode_changed_km2"].sum(),
        "mode_changed_area_pct": 100 * g["area_mode_changed_km2"].sum() / total_area if total_area > 0 else np.nan,
        "mode_consistency_area_pct": 100 - 100 * g["area_mode_changed_km2"].sum() / total_area if total_area > 0 else np.nan,

        "delta_LAI_safe_median_IQR": median_iqr_string(g["delta_LAI_safe"]),
        "abs_delta_LAI_safe_median_IQR": median_iqr_string(g["delta_LAI_safe"].abs()),
        "delta_target_LAI_median_IQR": median_iqr_string(g["delta_target_LAI"]),
        "delta_tree_ratio_median_IQR": median_iqr_string(g["delta_tree_ratio"]),
        "delta_tree_density_median_IQR": median_iqr_string(g["delta_tree_density"]),

        "delta_LAI_safe_area_weighted_mean": weighted_mean(g["delta_LAI_safe"], area),
        "delta_target_LAI_area_weighted_mean": weighted_mean(g["delta_target_LAI"], area),
        "delta_tree_ratio_area_weighted_mean": weighted_mean(g["delta_tree_ratio"], area),
        "delta_tree_density_area_weighted_mean": weighted_mean(g["delta_tree_density"], area),
    })


summary_all = (
    sens.groupby("scenario", dropna=False)
    .apply(summarize_scenario)
    .reset_index()
)

summary_gde = (
    sens.groupby(["scenario", "GDE_Level_base"], dropna=False)
    .apply(summarize_scenario)
    .reset_index()
    .rename(columns={"GDE_Level_base": "baseline_GDE_Level"})
)

summary = pd.concat(
    [
        summary_all.assign(group_type="all", baseline_GDE_Level="all"),
        summary_gde.assign(group_type="by_GDE"),
    ],
    ignore_index=True,
)

# 模式转移矩阵
transition_rows = []

for scenario, g in sens.groupby("scenario"):
    tab = pd.pivot_table(
        g,
        values="Area_km2",
        index="mode_sensitivity",
        columns="mode_base",
        aggfunc="sum",
        fill_value=0,
    )

    for idx in tab.index:
        for col in tab.columns:
            transition_rows.append({
                "scenario": scenario,
                "mode_sensitivity": idx,
                "mode_base": col,
                "area_km2": tab.loc[idx, col],
            })

transition = pd.DataFrame(transition_rows)

# GDE Level change matrix
level_rows = []

for scenario, g in sens.groupby("scenario"):
    tab = pd.pivot_table(
        g,
        values="Area_km2",
        index="GDE_Level_sensitivity",
        columns="GDE_Level_base",
        aggfunc="sum",
        fill_value=0,
    )

    for idx in tab.index:
        for col in tab.columns:
            level_rows.append({
                "scenario": scenario,
                "GDE_Level_sensitivity": idx,
                "GDE_Level_base": col,
                "area_km2": tab.loc[idx, col],
            })

level_change = pd.DataFrame(level_rows)


# ============================================================
# 10. 输出
# ============================================================

sens = sens.sort_values(["scenario", "ClassCode"]).reset_index(drop=True)

sens.to_csv(OUT_BYCLASS, index=False, encoding="utf-8-sig")
summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
transition.to_csv(OUT_TRANSITION, index=False, encoding="utf-8-sig")
level_change.to_csv(OUT_LEVEL_CHANGE, index=False, encoding="utf-8-sig")


# ============================================================
# 11. Markdown 报告
# ============================================================

lines = []

lines.append("# S6a GDE threshold sensitivity report")
lines.append("")
lines.append("## Definition")
lines.append("")
lines.append(
    "This diagnostic sensitivity analysis recalculates GDE levels using alternative GDE_frac thresholds. "
    "The continuous GDE fraction field used here is the mean GDE fraction of each modelling stratum "
    "(GDE_frac_mean when available). "
    "R_gde is adjusted using empirical baseline risk medians by GDE level while preserving within-stratum residual heterogeneity. "
    "The analysis then recalculates LAI_safe_max and rule-based final restoration modes."
)
lines.append("")
lines.append(
    "This is a diagnostic threshold-sensitivity analysis, not a full re-run of the original multi-objective optimization."
)
lines.append("")
lines.append(
    "For defensive reporting, target_LAI, tree–shrub–grass ratios and density thresholds were recalculated only for units whose GDE_Level changed under an alternative threshold scenario. "
    "Units with unchanged GDE_Level retained their baseline configuration to avoid template-induced pseudo-differences."
)
lines.append("")

lines.append("## Input GDE fraction")
lines.append("")
lines.append(f"GDE fraction field used: {gde_frac_col}")
lines.append(f"GDE_frac min: {base['GDE_frac'].min()}")
lines.append(f"GDE_frac max: {base['GDE_frac'].max()}")
lines.append("")

lines.append("## Scenario thresholds")
lines.append("")
for name, th in SCENARIOS.items():
    lines.append(f"- {name}: low={th['low']}, mid={th['mid']}, high={th['high']}")
lines.append("")

lines.append("## Summary")
lines.append("")
lines.append(summary.to_string(index=False))
lines.append("")

lines.append("## Main interpretation")
lines.append("")

for _, r in summary_all.iterrows():
    lines.append(
        f"{r['scenario']}: "
        f"GDE-level changed area = {r['GDE_level_changed_area_pct']:.2f}%; "
        f"mode-changed area = {r['mode_changed_area_pct']:.2f}%; "
        f"mode consistency = {r['mode_consistency_area_pct']:.2f}%; "
        f"|ΔLAI_safe| median [IQR] = {r['abs_delta_LAI_safe_median_IQR']}; "
        f"Δtarget_LAI median [IQR] = {r['delta_target_LAI_median_IQR']}; "
        f"Δtree_ratio median [IQR] = {r['delta_tree_ratio_median_IQR']}."
    )

lines.append("")
lines.append("## Notes for manuscript")
lines.append("")
lines.append(
    "High mode-consistency across alternative thresholds indicates that the main restoration pattern is not driven by a single arbitrary GDE_frac threshold combination. "
    "If mode changes are concentrated in GDE_Level 1 and 3, this supports the interpretation that GDE-sensitive zones are the most responsive to threshold uncertainty."
)

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


print("GDE 分级阈值敏感性分析完成。")
print(f"逐单元结果: {OUT_BYCLASS}")
print(f"汇总表: {OUT_SUMMARY}")
print(f"模式转移矩阵: {OUT_TRANSITION}")
print(f"GDE等级转移矩阵: {OUT_LEVEL_CHANGE}")
print(f"模板表: {OUT_TEMPLATES}")
print(f"报告: {OUT_REPORT}")

print("\n总体汇总：")
print(summary_all.to_string(index=False))
