"""
13_weight_sensitivity.py

Tests deterministic and random-weight sensitivity of the multi-objective optimization results.

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

OUT_DIR = DATA_DIR / "weight_sensitivity_outputs_FIXED"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DETERMINISTIC = OUT_DIR / "S6c_weight_sensitivity_byClass_deterministic_FIXED.csv"
OUT_SUMMARY = OUT_DIR / "S6c_weight_sensitivity_summary_FIXED.csv"
OUT_TRANSITION = OUT_DIR / "S6c_weight_sensitivity_mode_transition_area_FIXED.csv"

OUT_RANDOM_BYDRAW = OUT_DIR / "S6c_weight_sensitivity_random_byDraw_FIXED.csv"
OUT_RANDOM_BYCLASS = OUT_DIR / "S6c_weight_sensitivity_random_byClass_FIXED.csv"
OUT_RANDOM_SUMMARY = OUT_DIR / "S6c_weight_sensitivity_random_summary_FIXED.csv"

OUT_REPORT = OUT_DIR / "S6c_weight_sensitivity_report_FIXED.md"


# ============================================================
# 1. 权重情景
# ============================================================

WEIGHT_SCENARIOS = {
    "equal_25_each": {
        "score_sand_used": 0.25,
        "score_stability_used": 0.25,
        "score_water_security_used": 0.25,
        "score_gde_safety_used": 0.25,
    },
    "sand_emphasis_40": {
        "score_sand_used": 0.40,
        "score_stability_used": 0.20,
        "score_water_security_used": 0.20,
        "score_gde_safety_used": 0.20,
    },
    "stability_emphasis_40": {
        "score_sand_used": 0.20,
        "score_stability_used": 0.40,
        "score_water_security_used": 0.20,
        "score_gde_safety_used": 0.20,
    },
    "water_emphasis_40": {
        "score_sand_used": 0.20,
        "score_stability_used": 0.20,
        "score_water_security_used": 0.40,
        "score_gde_safety_used": 0.20,
    },
    "gde_emphasis_40": {
        "score_sand_used": 0.20,
        "score_stability_used": 0.20,
        "score_water_security_used": 0.20,
        "score_gde_safety_used": 0.40,
    },
    "water_gde_emphasis_35_35": {
        "score_sand_used": 0.15,
        "score_stability_used": 0.15,
        "score_water_security_used": 0.35,
        "score_gde_safety_used": 0.35,
    },
}

RANDOM_N = 1000
RANDOM_SEED = 202502


# ============================================================
# 2. 文件匹配规则
# ============================================================

FILE_PATTERNS = {
    "hydro": [
        "ThreeNorth_Class_HydroSupport_2005_2024*.csv",
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
    "pareto": [
        "ThreeNorth_Class_MOO_ParetoFront*.csv",
        "*ParetoFront*.csv",
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


def to_numeric_if_exists(df: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame | None:
    if df is None:
        return df
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


def infer_mode_from_ratios(row: pd.Series, fallback: str = "Unknown") -> str:
    """
    根据候选解的乔灌草比例推断模式。
    如果 ParetoFront 中已有 final_mode/template_group 等字段，会优先用这些字段。
    """
    for c in ["final_mode", "mode", "template_group", "veg_zone_primary"]:
        if c in row.index and pd.notna(row[c]):
            s = str(row[c])
            if any(k in s for k in ["乔", "Tree", "tree"]):
                if any(k in s for k in ["灌", "Shrub", "shrub"]):
                    return "Tree–shrub–grass restoration"
            if any(k in s for k in ["灌草", "Shrub", "shrub"]):
                return "Shrub–grass restoration"
            if any(k in s for k in ["草本", "封育", "Herbaceous", "exclosure", "保育"]):
                return "Herbaceous–exclosure recovery"
            if any(k in s for k in ["近自然", "Close"]):
                return "Close-to-nature stand management"

    tree = row.get("tree_ratio", np.nan)
    shrub = row.get("shrub_ratio", np.nan)
    grass = row.get("grass_ratio", np.nan)

    tree = 0 if pd.isna(tree) else float(tree)
    shrub = 0 if pd.isna(shrub) else float(shrub)
    grass = 0 if pd.isna(grass) else float(grass)

    if tree > 0.10:
        return "Tree–shrub–grass restoration"
    if shrub > 0.05:
        return "Shrub–grass restoration"
    if grass > 0.50 or (tree == 0 and shrub == 0):
        return "Herbaceous–exclosure recovery"

    return fallback


def make_solution_key(row: pd.Series) -> str:
    """
    构建候选解唯一标识。
    优先用 scheme_id；没有则用关键连续变量四舍五入组合。
    """
    for c in ["scheme_id", "SchemeID", "candidate_id", "CandidateID"]:
        if c in row.index and pd.notna(row[c]):
            return f"{c}:{row[c]}"

    parts = []
    for c in [
        "target_LAI",
        "tree_ratio",
        "shrub_ratio",
        "grass_ratio",
        "tree_density_per_ha",
        "shrub_density_per_ha",
        "grass_cover_target_pct",
    ]:
        if c in row.index:
            v = row[c]
            if pd.isna(v):
                parts.append(f"{c}=NA")
            else:
                parts.append(f"{c}={float(v):.6f}")

    return "|".join(parts)


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


def minmax_by_class(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
    benefit: bool = True,
) -> pd.DataFrame:
    """
    若 score_* 不存在，则从 obj_* 生成 0–1 分数。
    benefit=True 表示越大越好；False 表示越小越好。
    """
    out = df.copy()
    out[out_col] = np.nan

    for cc, g in out.groupby("ClassCode"):
        x = pd.to_numeric(g[value_col], errors="coerce")
        xmin = x.min()
        xmax = x.max()

        if pd.isna(xmin) or pd.isna(xmax):
            val = pd.Series(np.nan, index=g.index)
        elif abs(xmax - xmin) < 1e-12:
            val = pd.Series(0.5, index=g.index)
        else:
            if benefit:
                val = (x - xmin) / (xmax - xmin)
            else:
                val = (xmax - x) / (xmax - xmin)

        out.loc[g.index, out_col] = val

    return out


def prepare_pareto_scores(pareto: pd.DataFrame) -> pd.DataFrame:
    """
    确保存在统一的四个 score_*_used 字段。
    优先使用已有 score_*；否则从 obj_* 转换。
    """
    p = pareto.copy()

    if "score_sand" in p.columns:
        p["score_sand_used"] = pd.to_numeric(p["score_sand"], errors="coerce")
    elif "obj1_sand_benefit" in p.columns:
        p = minmax_by_class(p, "obj1_sand_benefit", "score_sand_used", benefit=True)
    else:
        raise KeyError("ParetoFront 缺少 score_sand 或 obj1_sand_benefit。")

    if "score_stability" in p.columns:
        p["score_stability_used"] = pd.to_numeric(p["score_stability"], errors="coerce")
    elif "obj2_eco_stability" in p.columns:
        p = minmax_by_class(p, "obj2_eco_stability", "score_stability_used", benefit=True)
    else:
        raise KeyError("ParetoFront 缺少 score_stability 或 obj2_eco_stability。")

    if "score_water_security" in p.columns:
        p["score_water_security_used"] = pd.to_numeric(p["score_water_security"], errors="coerce")
    elif "obj3_water_pressure" in p.columns:
        p = minmax_by_class(p, "obj3_water_pressure", "score_water_security_used", benefit=False)
    else:
        raise KeyError("ParetoFront 缺少 score_water_security 或 obj3_water_pressure。")

    if "score_gde_safety" in p.columns:
        p["score_gde_safety_used"] = pd.to_numeric(p["score_gde_safety"], errors="coerce")
    elif "obj4_gde_risk" in p.columns:
        p = minmax_by_class(p, "obj4_gde_risk", "score_gde_safety_used", benefit=False)
    else:
        raise KeyError("ParetoFront 缺少 score_gde_safety 或 obj4_gde_risk。")

    for c in [
        "score_sand_used",
        "score_stability_used",
        "score_water_security_used",
        "score_gde_safety_used",
    ]:
        p[c] = p[c].clip(0, 1)

    p["solution_key"] = p.apply(make_solution_key, axis=1)
    p["mode_from_solution"] = p.apply(lambda r: infer_mode_from_ratios(r), axis=1)

    return p


def select_by_weights(
    pareto: pd.DataFrame,
    weights: dict[str, float],
    scenario: str,
) -> pd.DataFrame:
    """
    每个 ClassCode 选择加权得分最高的 Pareto 解。
    """
    p = pareto.copy()

    score_cols = [
        "score_sand_used",
        "score_stability_used",
        "score_water_security_used",
        "score_gde_safety_used",
    ]

    for c in score_cols:
        if c not in p.columns:
            raise KeyError(f"缺少 {c}")

    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-8:
        weights = {k: v / wsum for k, v in weights.items()}

    p["weight_sand"] = weights["score_sand_used"]
    p["weight_stability"] = weights["score_stability_used"]
    p["weight_water"] = weights["score_water_security_used"]
    p["weight_gde"] = weights["score_gde_safety_used"]

    p["weighted_score"] = (
        p["score_sand_used"] * weights["score_sand_used"]
        + p["score_stability_used"] * weights["score_stability_used"]
        + p["score_water_security_used"] * weights["score_water_security_used"]
        + p["score_gde_safety_used"] * weights["score_gde_safety_used"]
    )

    # 若同分，优先选择水安全/GDE安全更高的解，再选 target_LAI 较低的解
    sort_cols = [
        "ClassCode",
        "weighted_score",
        "score_water_security_used",
        "score_gde_safety_used",
    ]
    ascending = [True, False, False, False]

    if "target_LAI" in p.columns:
        sort_cols.append("target_LAI")
        ascending.append(True)

    p = p.sort_values(sort_cols, ascending=ascending)

    selected = p.groupby("ClassCode", as_index=False).head(1).copy()
    selected["scenario"] = scenario

    return selected


def build_base_table(
    hydro: pd.DataFrame,
    mode_main: pd.DataFrame,
    best: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    构建 208 个单元的 baseline 表。
    backfill 单元也保留，用于全域面积一致性计算。
    """
    area_col_hydro = pick_col(hydro, ["Area_km2", "area_km2", "Area"], required=False)
    area_col_mode = pick_col(mode_main, ["Area_km2", "area_km2", "Area"], required=False)
    final_mode_col = pick_col(mode_main, ["final_mode", "FinalMode", "mode"], required=True)

    base_cols = ["ClassCode"]

    for c in ["BaseZone", "GDE_Level"]:
        if c in hydro.columns:
            base_cols.append(c)

    if area_col_hydro is not None and area_col_hydro in hydro.columns:
        base_cols.append(area_col_hydro)

    base_cols = list(dict.fromkeys(base_cols))
    base = hydro[base_cols].copy()

    rename = {}
    if area_col_hydro is not None and area_col_hydro in base.columns:
        rename[area_col_hydro] = "Area_km2"
    if "GDE_Level" in base.columns:
        rename["GDE_Level"] = "GDE_Level_base"

    base = base.rename(columns=rename)

    if "Area_km2" not in base.columns:
        if area_col_mode is None:
            raise KeyError("HydroSupport 和 Mode 表都缺少面积字段。")
        base = base.merge(
            mode_main[["ClassCode", area_col_mode]].rename(columns={area_col_mode: "Area_km2"}),
            on="ClassCode",
            how="left",
        )

    if "GDE_Level_base" not in base.columns:
        if "GDE_Level" in mode_main.columns:
            base = base.merge(
                mode_main[["ClassCode", "GDE_Level"]].rename(columns={"GDE_Level": "GDE_Level_base"}),
                on="ClassCode",
                how="left",
            )
        else:
            base["GDE_Level_base"] = np.nan

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

    # 用 BestCompromise 补充 baseline 优化解信息
    if best is not None:
        best2 = best.copy()
        best2["baseline_solution_key"] = best2.apply(make_solution_key, axis=1)

        best_keep = ["ClassCode", "baseline_solution_key"]

        for c in [
            "target_LAI",
            "tree_ratio",
            "shrub_ratio",
            "grass_ratio",
            "tree_density_per_ha",
            "shrub_density_per_ha",
            "grass_cover_target_pct",
        ]:
            if c in best2.columns:
                best_keep.append(c)

        best_keep = list(dict.fromkeys(best_keep))
        best_use = best2[best_keep].copy()

        best_use = best_use.rename(columns={
            "target_LAI": "target_LAI_best",
            "tree_ratio": "tree_ratio_best",
            "shrub_ratio": "shrub_ratio_best",
            "grass_ratio": "grass_ratio_best",
            "tree_density_per_ha": "tree_density_best",
            "shrub_density_per_ha": "shrub_density_best",
            "grass_cover_target_pct": "grass_cover_best",
        })

        base = base.merge(best_use, on="ClassCode", how="left")

        fill_pairs = [
            ("target_LAI_base", "target_LAI_best"),
            ("tree_ratio_base", "tree_ratio_best"),
            ("shrub_ratio_base", "shrub_ratio_best"),
            ("grass_ratio_base", "grass_ratio_best"),
            ("tree_density_base", "tree_density_best"),
            ("shrub_density_base", "shrub_density_best"),
            ("grass_cover_base", "grass_cover_best"),
        ]

        for a, b in fill_pairs:
            if a not in base.columns:
                base[a] = np.nan
            if b in base.columns:
                base[a] = base[a].fillna(base[b])

    if "baseline_solution_key" not in base.columns:
        base["baseline_solution_key"] = ""

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

    if "optimized_flag" not in base.columns:
        base["optimized_flag"] = np.where(
            base["baseline_solution_key"].astype(str).str.len() > 0,
            1,
            0,
        )

    return base


def assemble_scenario_result(
    base: pd.DataFrame,
    selected: pd.DataFrame | None,
    scenario: str,
    weights: dict[str, float] | None,
) -> pd.DataFrame:
    """
    把某个权重情景的重选结果合并回 208 个单元。
    非 Pareto 优化单元保持主情景配置。
    """
    out = base.copy()
    out["scenario"] = scenario

    if weights is None:
        out["weight_sand"] = np.nan
        out["weight_stability"] = np.nan
        out["weight_water"] = np.nan
        out["weight_gde"] = np.nan
    else:
        out["weight_sand"] = weights["score_sand_used"]
        out["weight_stability"] = weights["score_stability_used"]
        out["weight_water"] = weights["score_water_security_used"]
        out["weight_gde"] = weights["score_gde_safety_used"]

    out["selection_status"] = "baseline_or_backfill"
    out["selected_solution_key"] = out["baseline_solution_key"]
    out["weighted_score"] = np.nan

    out["mode_sensitivity"] = out["mode_base"]
    out["target_LAI_sensitivity"] = out["target_LAI_base"]
    out["tree_ratio_sensitivity"] = out["tree_ratio_base"]
    out["shrub_ratio_sensitivity"] = out["shrub_ratio_base"]
    out["grass_ratio_sensitivity"] = out["grass_ratio_base"]
    out["tree_density_sensitivity"] = out["tree_density_base"]
    out["shrub_density_sensitivity"] = out["shrub_density_base"]
    out["grass_cover_sensitivity"] = out["grass_cover_base"]

    if selected is not None and len(selected) > 0:
        sel_cols = [
            "ClassCode",
            "solution_key",
            "weighted_score",
            "mode_from_solution",
            "target_LAI",
            "tree_ratio",
            "shrub_ratio",
            "grass_ratio",
            "tree_density_per_ha",
            "shrub_density_per_ha",
            "grass_cover_target_pct",
            "score_sand_used",
            "score_stability_used",
            "score_water_security_used",
            "score_gde_safety_used",
        ]

        sel_cols = [c for c in sel_cols if c in selected.columns]

        s = selected[sel_cols].copy()
        s = s.rename(columns={
            "solution_key": "selected_solution_key_new",
            "weighted_score": "weighted_score_new",
            "mode_from_solution": "mode_sensitivity_new",
            "target_LAI": "target_LAI_sensitivity_new",
            "tree_ratio": "tree_ratio_sensitivity_new",
            "shrub_ratio": "shrub_ratio_sensitivity_new",
            "grass_ratio": "grass_ratio_sensitivity_new",
            "tree_density_per_ha": "tree_density_sensitivity_new",
            "shrub_density_per_ha": "shrub_density_sensitivity_new",
            "grass_cover_target_pct": "grass_cover_sensitivity_new",
        })

        out = out.merge(s, on="ClassCode", how="left")

        has_selection = out["selected_solution_key_new"].notna()

        out.loc[has_selection, "selection_status"] = "reselected_from_pareto"
        out.loc[has_selection, "selected_solution_key"] = out.loc[has_selection, "selected_solution_key_new"]
        out.loc[has_selection, "weighted_score"] = out.loc[has_selection, "weighted_score_new"]

        for c in [
            "mode_sensitivity",
            "target_LAI_sensitivity",
            "tree_ratio_sensitivity",
            "shrub_ratio_sensitivity",
            "grass_ratio_sensitivity",
            "tree_density_sensitivity",
            "shrub_density_sensitivity",
            "grass_cover_sensitivity",
        ]:
            newc = c + "_new"
            if newc in out.columns:
                out.loc[has_selection, c] = out.loc[has_selection, newc]

        drop_new = [c for c in out.columns if c.endswith("_new")]
        out = out.drop(columns=drop_new)

    out["solution_changed"] = (
        out["selection_status"].eq("reselected_from_pareto")
        & out["baseline_solution_key"].astype(str).ne(out["selected_solution_key"].astype(str))
    )

    out["mode_changed"] = out["mode_sensitivity"].astype(str).ne(out["mode_base"].astype(str))

    out["area_mode_changed_km2"] = np.where(out["mode_changed"], out["Area_km2"], 0.0)
    out["area_solution_changed_km2"] = np.where(out["solution_changed"], out["Area_km2"], 0.0)

    out["delta_target_LAI"] = out["target_LAI_sensitivity"] - out["target_LAI_base"]
    out["delta_tree_ratio"] = out["tree_ratio_sensitivity"] - out["tree_ratio_base"]
    out["delta_shrub_ratio"] = out["shrub_ratio_sensitivity"] - out["shrub_ratio_base"]
    out["delta_grass_ratio"] = out["grass_ratio_sensitivity"] - out["grass_ratio_base"]
    out["delta_tree_density"] = out["tree_density_sensitivity"] - out["tree_density_base"]
    out["delta_shrub_density"] = out["shrub_density_sensitivity"] - out["shrub_density_base"]
    out["delta_grass_cover"] = out["grass_cover_sensitivity"] - out["grass_cover_base"]

    return out


def summarize_scenario(g: pd.DataFrame) -> pd.Series:
    area = g["Area_km2"]
    total_area = area.sum()
    optimized = g["selection_status"].eq("reselected_from_pareto")

    return pd.Series({
        "n_units": len(g),
        "n_optimized_units": int(optimized.sum()),
        "area_km2": total_area,

        "solution_changed_area_km2": g["area_solution_changed_km2"].sum(),
        "solution_changed_area_pct": 100 * g["area_solution_changed_km2"].sum() / total_area if total_area > 0 else np.nan,

        "mode_changed_area_km2": g["area_mode_changed_km2"].sum(),
        "mode_changed_area_pct": 100 * g["area_mode_changed_km2"].sum() / total_area if total_area > 0 else np.nan,
        "mode_consistency_area_pct": 100 - 100 * g["area_mode_changed_km2"].sum() / total_area if total_area > 0 else np.nan,

        "delta_target_LAI_median_IQR": median_iqr_string(g["delta_target_LAI"]),
        "delta_tree_ratio_median_IQR": median_iqr_string(g["delta_tree_ratio"]),
        "delta_tree_density_median_IQR": median_iqr_string(g["delta_tree_density"]),
        "delta_shrub_density_median_IQR": median_iqr_string(g["delta_shrub_density"]),

        "delta_target_LAI_area_weighted_mean": weighted_mean(g["delta_target_LAI"], area),
        "delta_tree_ratio_area_weighted_mean": weighted_mean(g["delta_tree_ratio"], area),
        "delta_tree_density_area_weighted_mean": weighted_mean(g["delta_tree_density"], area),
    })


# ============================================================
# 4. 读取数据
# ============================================================

hydro = read_required("hydro")
pareto = read_required("pareto")
best = read_optional("best")

mode_impl = read_optional("mode_impl")
mode_map = read_optional("mode_map")

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
    "Area_km2",
    "area_km2",
    "Area",
    "target_LAI",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
    "obj1_sand_benefit",
    "obj2_eco_stability",
    "obj3_water_pressure",
    "obj4_gde_risk",
    "score_sand",
    "score_stability",
    "score_water_security",
    "score_gde_safety",
    "best_compromise_score",
    "optimized_flag",
]

for df0 in [hydro, pareto, best, mode_main]:
    if df0 is not None:
        to_numeric_if_exists(df0, numeric_cols)


# ============================================================
# 6. 准备数据
# ============================================================

base = build_base_table(hydro, mode_main, best)
pareto2 = prepare_pareto_scores(pareto)

print("[INFO] baseline units:", len(base))
print("[INFO] ParetoFront rows:", len(pareto2))
print("[INFO] Pareto ClassCode n:", pareto2["ClassCode"].nunique())


# ============================================================
# 7. 固定权重情景
# ============================================================

scenario_results = []

# reported baseline: 不重选
baseline_res = assemble_scenario_result(
    base=base,
    selected=None,
    scenario="reported_baseline",
    weights=None,
)
scenario_results.append(baseline_res)

# 其他权重情景：从 ParetoFront 重选
for scenario, weights in WEIGHT_SCENARIOS.items():
    selected = select_by_weights(
        pareto=pareto2,
        weights=weights,
        scenario=scenario,
    )

    res = assemble_scenario_result(
        base=base,
        selected=selected,
        scenario=scenario,
        weights=weights,
    )

    scenario_results.append(res)

det = pd.concat(scenario_results, ignore_index=True)

summary_all = (
    det.groupby("scenario", dropna=False)
    .apply(summarize_scenario)
    .reset_index()
)

summary_gde = (
    det.groupby(["scenario", "GDE_Level_base"], dropna=False)
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

for scenario, g in det.groupby("scenario"):
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


# ============================================================
# 8. 随机权重敏感性
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)
random_weights = rng.dirichlet(alpha=np.ones(4), size=RANDOM_N)

random_draw_rows = []
random_class_records = []

for i in range(RANDOM_N):
    w = random_weights[i]

    weights = {
        "score_sand_used": float(w[0]),
        "score_stability_used": float(w[1]),
        "score_water_security_used": float(w[2]),
        "score_gde_safety_used": float(w[3]),
    }

    scenario = f"random_{i:04d}"

    selected = select_by_weights(
        pareto=pareto2,
        weights=weights,
        scenario=scenario,
    )

    res = assemble_scenario_result(
        base=base,
        selected=selected,
        scenario=scenario,
        weights=weights,
    )

    s = summarize_scenario(res)

    random_draw_rows.append({
        "draw_id": i,
        "weight_sand": weights["score_sand_used"],
        "weight_stability": weights["score_stability_used"],
        "weight_water": weights["score_water_security_used"],
        "weight_gde": weights["score_gde_safety_used"],
        "solution_changed_area_pct": s["solution_changed_area_pct"],
        "mode_changed_area_pct": s["mode_changed_area_pct"],
        "mode_consistency_area_pct": s["mode_consistency_area_pct"],
        "delta_target_LAI_area_weighted_mean": s["delta_target_LAI_area_weighted_mean"],
        "delta_tree_ratio_area_weighted_mean": s["delta_tree_ratio_area_weighted_mean"],
        "delta_tree_density_area_weighted_mean": s["delta_tree_density_area_weighted_mean"],
    })

    keep = res.loc[
        res["selection_status"].eq("reselected_from_pareto"),
        [
            "ClassCode",
            "GDE_Level_base",
            "Area_km2",
            "selected_solution_key",
            "baseline_solution_key",
            "solution_changed",
            "mode_base",
            "mode_sensitivity",
            "mode_changed",
            "target_LAI_sensitivity",
            "tree_ratio_sensitivity",
            "tree_density_sensitivity",
        ],
    ].copy()

    keep["draw_id"] = i
    random_class_records.append(keep)

random_by_draw = pd.DataFrame(random_draw_rows)
random_long = pd.concat(random_class_records, ignore_index=True)


def summarize_random_class(g: pd.DataFrame) -> pd.Series:
    n = len(g)

    same_solution = (~g["solution_changed"]).sum() / n if n > 0 else np.nan
    same_mode = (~g["mode_changed"]).sum() / n if n > 0 else np.nan

    mode_counts = g["mode_sensitivity"].value_counts(normalize=True)
    modal_mode = mode_counts.index[0] if len(mode_counts) > 0 else "NA"
    modal_mode_freq = mode_counts.iloc[0] if len(mode_counts) > 0 else np.nan

    return pd.Series({
        "n_draws": n,
        "baseline_solution_frequency": same_solution,
        "baseline_mode_frequency": same_mode,
        "modal_mode": modal_mode,
        "modal_mode_frequency": modal_mode_freq,
        "target_LAI_median_IQR": median_iqr_string(g["target_LAI_sensitivity"]),
        "tree_ratio_median_IQR": median_iqr_string(g["tree_ratio_sensitivity"]),
        "tree_density_median_IQR": median_iqr_string(g["tree_density_sensitivity"]),
    })


random_by_class = (
    random_long.groupby(["ClassCode", "GDE_Level_base", "Area_km2"], dropna=False)
    .apply(summarize_random_class)
    .reset_index()
)

# ============================================================
# 关键修正：
# delta_tree_ratio_area_weighted_mean 的 q75 必须用 quantile(0.75)，
# 不能误写成 quantile(0.25)。
# ============================================================

random_summary = pd.DataFrame({
    "metric": [
        "mode_consistency_area_pct",
        "mode_changed_area_pct",
        "solution_changed_area_pct",
        "delta_target_LAI_area_weighted_mean",
        "delta_tree_ratio_area_weighted_mean",
        "delta_tree_density_area_weighted_mean",
    ],
    "median": [
        random_by_draw["mode_consistency_area_pct"].median(),
        random_by_draw["mode_changed_area_pct"].median(),
        random_by_draw["solution_changed_area_pct"].median(),
        random_by_draw["delta_target_LAI_area_weighted_mean"].median(),
        random_by_draw["delta_tree_ratio_area_weighted_mean"].median(),
        random_by_draw["delta_tree_density_area_weighted_mean"].median(),
    ],
    "q25": [
        random_by_draw["mode_consistency_area_pct"].quantile(0.25),
        random_by_draw["mode_changed_area_pct"].quantile(0.25),
        random_by_draw["solution_changed_area_pct"].quantile(0.25),
        random_by_draw["delta_target_LAI_area_weighted_mean"].quantile(0.25),
        random_by_draw["delta_tree_ratio_area_weighted_mean"].quantile(0.25),
        random_by_draw["delta_tree_density_area_weighted_mean"].quantile(0.25),
    ],
    "q75": [
        random_by_draw["mode_consistency_area_pct"].quantile(0.75),
        random_by_draw["mode_changed_area_pct"].quantile(0.75),
        random_by_draw["solution_changed_area_pct"].quantile(0.75),
        random_by_draw["delta_target_LAI_area_weighted_mean"].quantile(0.75),
        random_by_draw["delta_tree_ratio_area_weighted_mean"].quantile(0.75),
        random_by_draw["delta_tree_density_area_weighted_mean"].quantile(0.75),
    ],
})


# ============================================================
# 9. 输出
# ============================================================

det.to_csv(OUT_DETERMINISTIC, index=False, encoding="utf-8-sig")
summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
transition.to_csv(OUT_TRANSITION, index=False, encoding="utf-8-sig")

random_by_draw.to_csv(OUT_RANDOM_BYDRAW, index=False, encoding="utf-8-sig")
random_by_class.to_csv(OUT_RANDOM_BYCLASS, index=False, encoding="utf-8-sig")
random_summary.to_csv(OUT_RANDOM_SUMMARY, index=False, encoding="utf-8-sig")


# ============================================================
# 10. Markdown 报告
# ============================================================

lines = []

lines.append("# S6c multi-objective weight sensitivity report")
lines.append("")
lines.append("## Definition")
lines.append("")
lines.append(
    "This sensitivity analysis recalculates the best-compromise solution within each ParetoFront by changing the weights of four normalized objectives: sand-fixation benefit, ecological stability, water security and GDE safety. "
    "The reported baseline is the original selected BestCompromise and is not reselected. Other deterministic scenarios and 1000 random Dirichlet weight vectors reselect the maximum weighted-score Pareto solution."
)
lines.append("")
lines.append(
    "Backfilled ecological-safety units without ParetoFront records are retained as their reported baseline configurations, so area-based summaries are calculated for the full modelling domain."
)
lines.append("")

lines.append("## Deterministic weight scenarios")
lines.append("")
for name, weights in WEIGHT_SCENARIOS.items():
    lines.append(
        f"- {name}: "
        f"sand={weights['score_sand_used']}, "
        f"stability={weights['score_stability_used']}, "
        f"water={weights['score_water_security_used']}, "
        f"GDE={weights['score_gde_safety_used']}"
    )
lines.append("")

lines.append("## Deterministic summary")
lines.append("")
lines.append(summary.to_string(index=False))
lines.append("")

lines.append("## Random-weight summary")
lines.append("")
lines.append(random_summary.to_string(index=False))
lines.append("")

lines.append("## Main interpretation")
lines.append("")

for _, r in summary_all.iterrows():
    lines.append(
        f"{r['scenario']}: "
        f"solution-changed area = {r['solution_changed_area_pct']:.2f}%; "
        f"mode-changed area = {r['mode_changed_area_pct']:.2f}%; "
        f"mode consistency = {r['mode_consistency_area_pct']:.2f}%; "
        f"Δtarget_LAI median [IQR] = {r['delta_target_LAI_median_IQR']}; "
        f"Δtree_ratio median [IQR] = {r['delta_tree_ratio_median_IQR']}."
    )

lines.append("")
lines.append("## Notes for manuscript")
lines.append("")
lines.append(
    "If solution changes are frequent but final modes and key thresholds remain stable, the ParetoFront contains multiple near-equivalent solutions but the ecological interpretation is robust. "
    "If GDE- or water-emphasis scenarios reduce target_LAI or tree_ratio mainly in GDE-sensitive strata, this supports the conclusion that water and GDE safeguards constrain woody-planting intensity."
)

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


print("多目标权重敏感性分析完成。")
print(f"固定权重逐单元结果: {OUT_DETERMINISTIC}")
print(f"固定权重汇总: {OUT_SUMMARY}")
print(f"模式转移矩阵: {OUT_TRANSITION}")
print(f"随机权重逐次结果: {OUT_RANDOM_BYDRAW}")
print(f"随机权重逐单元稳定性: {OUT_RANDOM_BYCLASS}")
print(f"随机权重汇总: {OUT_RANDOM_SUMMARY}")
print(f"报告: {OUT_REPORT}")

print("\n固定权重总体汇总：")
print(summary_all.to_string(index=False))

print("\n随机权重汇总：")
print(random_summary.to_string(index=False))
