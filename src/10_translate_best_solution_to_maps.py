"""
10_translate_best_solution_to_maps.py

Translates optimal solutions into implementable restoration modes and map-ready tables.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd


# =============================================================================
# 0. 参数区
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"
DATA_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path(__file__).resolve().parent

BEST_PATTERNS = ["ThreeNorth_Class_MOO_BestCompromise.csv"]
PARETO_PATTERNS = ["ThreeNorth_Class_MOO_ParetoFront.csv"]
RANGE_PATTERNS = ["ThreeNorth_Class_MOO_ParetoRanges_byClass.csv"]
HYDRO_PATTERNS = ["ThreeNorth_Class_HydroSupport_2005_2024.csv"]
OBS_PATTERNS = ["ThreeNorth_Class_ObservedStructure.csv"]

OUT_IMPL_CSV = DATA_DIR / "ThreeNorth_Class_ModeImplementation.csv"
OUT_MAPREADY_CSV = DATA_DIR / "ThreeNorth_Class_ModeMapReady.csv"
OUT_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_ModeSummary.csv"
OUT_ZONE_SUMMARY_CSV = DATA_DIR / "ThreeNorth_Class_ModeSummary_byBaseZone.csv"
OUT_QA_TXT = DATA_DIR / "ThreeNorth_Class_Mode_QA.txt"

EPS = 1e-9


# =============================================================================
# 1. 工具函数
# =============================================================================

def pick_one(patterns: List[str], required: bool = True) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(DATA_DIR.glob(pat))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"在 {DATA_DIR} 中找不到匹配文件：{patterns}")
    return None


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def corr_share(num: pd.Series, den: pd.Series) -> pd.Series:
    out = np.full(len(num), np.nan, dtype=float)
    n = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(n) & np.isfinite(d) & (d > 0.05)
    out[mask] = n[mask] / d[mask]
    return pd.Series(out, index=num.index, dtype=float)


def safe_int_from_row(row: pd.Series, key: str, default: int = 0) -> int:
    val = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    if pd.isna(val):
        return int(default)
    return int(val)


def safe_float_from_row(row: pd.Series, key: str, default: float = np.nan) -> float:
    val = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    return float(val) if pd.notna(val) else float(default)


def map_final_mode(row: pd.Series) -> str:
    optimized_flag = safe_int_from_row(row, "optimized_flag", 0)

    # 未进入第五步优化的类：统一回填为草本/封育恢复型
    if optimized_flag == 0:
        return "草本/封育恢复型"

    z = str(row.get("veg_zone_primary", "") or "").strip()

    if z == "灌草主导":
        return "灌草主导恢复型"

    if z == "低覆盖灌草":
        return "草本/封育恢复型"

    if z in ["乔灌草", "低密乔灌草"]:
        obs_valid = safe_int_from_row(row, "ObsValid", 0)
        woody = safe_float_from_row(row, "WoodyFrac_obs", np.nan)
        dens = safe_float_from_row(row, "CurrentDensityProxy", np.nan)

        if obs_valid == 1 and (
            (pd.notna(woody) and woody >= 0.25) or
            (pd.notna(dens) and dens >= 0.35)
        ):
            return "存量林分近自然经营优化型"

        return "乔灌草协同恢复型"

    return "草本/封育恢复型"


def map_management_measure(row: pd.Series) -> str:
    mode = str(row.get("final_mode", ""))
    raw_suggestion = str(row.get("management_suggestion", "") or "").strip()
    optimized_flag = safe_int_from_row(row, "optimized_flag", 0)
    gde_level = safe_int_from_row(row, "GDE_Level", 0)

    if mode == "乔灌草协同恢复型":
        return raw_suggestion if raw_suggestion else "补植补造+复层配置"

    if mode == "灌草主导恢复型":
        return raw_suggestion if raw_suggestion else "补植补造+灌草配置"

    if mode == "草本/封育恢复型":
        if optimized_flag == 0 or gde_level == 1:
            return "封育保护/自然恢复"
        return raw_suggestion if raw_suggestion else "草本恢复+封育"

    if mode == "存量林分近自然经营优化型":
        return "抚育间伐+树种调整"

    return "封育保护/自然恢复"


# =============================================================================
# 2. 读取输入
# =============================================================================

BEST_CSV = pick_one(BEST_PATTERNS, required=True)
PARETO_CSV = pick_one(PARETO_PATTERNS, required=True)
RANGE_CSV = pick_one(RANGE_PATTERNS, required=True)
HYDRO_CSV = pick_one(HYDRO_PATTERNS, required=True)
OBS_CSV = pick_one(OBS_PATTERNS, required=False)

best = pd.read_csv(BEST_CSV, encoding="utf-8-sig")
pareto = pd.read_csv(PARETO_CSV, encoding="utf-8-sig")
ranges = pd.read_csv(RANGE_CSV, encoding="utf-8-sig")
hydro = pd.read_csv(HYDRO_CSV, encoding="utf-8-sig")
obs = pd.read_csv(OBS_CSV, encoding="utf-8-sig") if OBS_CSV is not None else pd.DataFrame()

print(f"BestCompromise: {best.shape}")
print(f"ParetoFront: {pareto.shape}")
print(f"ParetoRanges: {ranges.shape}")
print(f"HydroSupport: {hydro.shape}")
print(f"ObservedStructure: {obs.shape if not obs.empty else '未读取'}")

safe_numeric(best, ["ClassCode", "BaseZone", "GDE_Level", "tree_ratio", "shrub_ratio", "grass_ratio",
                    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
                    "target_LAI", "LAI_safe_max", "LAI_margin", "best_compromise_score",
                    "obj1_sand_benefit", "obj2_eco_stability", "obj3_water_pressure", "obj4_gde_risk"])
safe_numeric(ranges, ["ClassCode", "Pareto_n", "Tree_ratio_min", "Tree_ratio_max",
                      "Shrub_ratio_min", "Shrub_ratio_max", "Grass_ratio_min", "Grass_ratio_max",
                      "LAI_target_min", "LAI_target_max", "Tree_density_min", "Tree_density_max",
                      "Shrub_density_min", "Shrub_density_max", "Grass_cover_min", "Grass_cover_max",
                      "Sand_benefit_best", "Eco_stability_best", "Water_pressure_best", "GDE_risk_best"])
safe_numeric(hydro, ["ClassCode", "BaseZone", "GDE_Level", "Pixel_n", "Area_km2",
                     "LAI_current_3yr_mean", "NaturalFrac_mean", "NaturalMask_1km_mean"])
safe_numeric(obs, ["ClassCode", "TreeFrac_obs", "ShrubFrac_obs", "GrassFrac_obs", "WoodyFrac_obs",
                   "ObsVegFrac_total", "CurrentDensityProxy", "LAI_current_3yr_mean",
                   "FVC_current_3yr_mean", "NaturalFrac_mean", "NaturalMask_1km_mean"])


# =============================================================================
# 3. 以 HydroSupport 作为全区主表
# =============================================================================

master = hydro[[
    "ClassCode", "Pixel_n", "Area_km2", "BaseZone", "GDE_Level",
    "LAI_current_3yr_mean", "NaturalFrac_mean", "NaturalMask_1km_mean"
]].drop_duplicates(subset=["ClassCode"]).copy()

best_keep = [
    "ClassCode", "scheme_id", "veg_zone_primary", "management_suggestion",
    "constraint_mode", "management_priority",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_species", "shrub_species", "grass_species",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
    "target_LAI", "LAI_safe_max", "scheme_LAI_capacity", "LAI_margin",
    "R_gde", "water_group",
    "obj1_sand_benefit", "obj2_eco_stability", "obj3_water_pressure", "obj4_gde_risk",
    "score_sand", "score_stability", "score_water_security", "score_gde_safety",
    "best_compromise_score"
]
best_keep = [c for c in best_keep if c in best.columns]
best_sub = best[best_keep].drop_duplicates(subset=["ClassCode"]).copy()

ranges_sub = ranges.drop_duplicates(subset=["ClassCode"]).copy()

impl = master.merge(best_sub, on="ClassCode", how="left")
impl = impl.merge(ranges_sub, on=["ClassCode"], how="left", suffixes=("", "_range"))

impl["optimized_flag"] = impl["scheme_id"].notna().astype(int)

# 合并 ObservedStructure
if not obs.empty and "ClassCode" in obs.columns:
    obs_sub = obs.drop_duplicates(subset=["ClassCode"]).copy()

    if "ObsVegFrac_total" in obs_sub.columns:
        obs_sub["TreeShare_corr"] = corr_share(obs_sub["TreeFrac_obs"], obs_sub["ObsVegFrac_total"])
        obs_sub["ShrubShare_corr"] = corr_share(obs_sub["ShrubFrac_obs"], obs_sub["ObsVegFrac_total"])
        obs_sub["GrassShare_corr"] = corr_share(obs_sub["GrassFrac_obs"], obs_sub["ObsVegFrac_total"])

        area_col = "Area_km2" if "Area_km2" in obs_sub.columns else None
        if area_col is not None:
            area_ok = pd.to_numeric(obs_sub[area_col], errors="coerce").fillna(0) >= 5
        else:
            area_ok = pd.Series(True, index=obs_sub.index)

        obs_sub["ObsValid"] = (
            (pd.to_numeric(obs_sub["ObsVegFrac_total"], errors="coerce") >= 0.05) & area_ok
        ).astype(int)
    else:
        obs_sub["TreeShare_corr"] = np.nan
        obs_sub["ShrubShare_corr"] = np.nan
        obs_sub["GrassShare_corr"] = np.nan
        obs_sub["ObsValid"] = 0

    obs_keep = [
        "ClassCode", "TreeFrac_obs", "ShrubFrac_obs", "GrassFrac_obs", "WoodyFrac_obs",
        "ObsVegFrac_total", "CurrentDensityProxy",
        "TreeShare_corr", "ShrubShare_corr", "GrassShare_corr", "ObsValid"
    ]
    obs_keep = [c for c in obs_keep if c in obs_sub.columns]
    impl = impl.merge(obs_sub[obs_keep], on="ClassCode", how="left")

else:
    impl["ObsValid"] = 0

# 统一补值，避免 NaN 转 int 报错
if "ObsValid" not in impl.columns:
    impl["ObsValid"] = 0
impl["ObsValid"] = pd.to_numeric(impl["ObsValid"], errors="coerce").fillna(0).astype(int)

for c in ["WoodyFrac_obs", "CurrentDensityProxy",
          "TreeFrac_obs", "ShrubFrac_obs", "GrassFrac_obs",
          "TreeShare_corr", "ShrubShare_corr", "GrassShare_corr"]:
    if c not in impl.columns:
        impl[c] = np.nan


# =============================================================================
# 4. 最终模式映射
# =============================================================================

impl["final_mode"] = impl.apply(map_final_mode, axis=1)

mode_code_map = {
    "乔灌草协同恢复型": 1,
    "灌草主导恢复型": 2,
    "草本/封育恢复型": 3,
    "存量林分近自然经营优化型": 4,
}
impl["final_mode_code"] = impl["final_mode"].map(mode_code_map).astype("Int64")

impl["management_measure"] = impl.apply(map_management_measure, axis=1)

impl["backfill_reason"] = np.where(
    impl["optimized_flag"] == 0,
    np.where(
        pd.to_numeric(impl["GDE_Level"], errors="coerce").fillna(0).astype(int) == 1,
        "核心保育/自然恢复单元回填",
        "未进入人工优化单元，按草本/封育模式回填"
    ),
    ""
)

if "TreeShare_corr" in impl.columns:
    impl["Delta_Tree"] = impl["tree_ratio"] - impl["TreeShare_corr"]
    impl["Delta_Shrub"] = impl["shrub_ratio"] - impl["ShrubShare_corr"]
    impl["Delta_Grass"] = impl["grass_ratio"] - impl["GrassShare_corr"]
else:
    impl["Delta_Tree"] = np.nan
    impl["Delta_Shrub"] = np.nan
    impl["Delta_Grass"] = np.nan

if "CurrentDensityProxy" in impl.columns:
    impl["DensityOpt_proxy"] = np.where(
        pd.to_numeric(impl["LAI_safe_max"], errors="coerce") > EPS,
        pd.to_numeric(impl["target_LAI"], errors="coerce") / pd.to_numeric(impl["LAI_safe_max"], errors="coerce"),
        np.nan
    )
    impl["Delta_Density"] = impl["DensityOpt_proxy"] - pd.to_numeric(impl["CurrentDensityProxy"], errors="coerce")
else:
    impl["DensityOpt_proxy"] = np.nan
    impl["Delta_Density"] = np.nan


# =============================================================================
# 5. 输出
# =============================================================================

mapready_cols = [
    "ClassCode", "final_mode", "final_mode_code", "management_measure",
    "optimized_flag", "backfill_reason",
    "BaseZone", "GDE_Level", "Pixel_n", "Area_km2",
    "veg_zone_primary", "management_suggestion",
    "scheme_id", "target_LAI", "LAI_safe_max", "LAI_margin",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_species", "shrub_species", "grass_species",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
    "Pareto_n",
    "Tree_ratio_min", "Tree_ratio_max",
    "Shrub_ratio_min", "Shrub_ratio_max",
    "Grass_ratio_min", "Grass_ratio_max",
    "LAI_target_min", "LAI_target_max",
    "Tree_density_min", "Tree_density_max",
    "Shrub_density_min", "Shrub_density_max",
    "Grass_cover_min", "Grass_cover_max",
    "obj1_sand_benefit", "obj2_eco_stability", "obj3_water_pressure", "obj4_gde_risk",
    "score_sand", "score_stability", "score_water_security", "score_gde_safety",
    "best_compromise_score",
    "TreeFrac_obs", "ShrubFrac_obs", "GrassFrac_obs", "WoodyFrac_obs",
    "CurrentDensityProxy", "TreeShare_corr", "ShrubShare_corr", "GrassShare_corr",
    "Delta_Tree", "Delta_Shrub", "Delta_Grass", "Delta_Density",
]
mapready_cols = [c for c in mapready_cols if c in impl.columns]
impl[mapready_cols].to_csv(OUT_MAPREADY_CSV, index=False, encoding="utf-8-sig")

impl_cols = [
    "ClassCode", "final_mode", "management_measure",
    "BaseZone", "GDE_Level", "optimized_flag",
    "veg_zone_primary", "target_LAI", "LAI_safe_max",
    "tree_ratio", "shrub_ratio", "grass_ratio",
    "tree_density_per_ha", "shrub_density_per_ha", "grass_cover_target_pct",
    "Tree_ratio_min", "Tree_ratio_max",
    "Shrub_ratio_min", "Shrub_ratio_max",
    "Grass_ratio_min", "Grass_ratio_max",
    "LAI_target_min", "LAI_target_max",
    "backfill_reason"
]
impl_cols = [c for c in impl_cols if c in impl.columns]
impl[impl_cols].to_csv(OUT_IMPL_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 6. 汇总输出
# =============================================================================

summary = (
    impl.groupby("final_mode", as_index=False)
    .agg(
        Class_n=("ClassCode", "count"),
        Optimized_class_n=("optimized_flag", "sum"),
        Area_km2=("Area_km2", "sum"),
        Target_LAI_mean=("target_LAI", "mean"),
        Tree_ratio_mean=("tree_ratio", "mean"),
        Shrub_ratio_mean=("shrub_ratio", "mean"),
        Grass_ratio_mean=("grass_ratio", "mean"),
    )
    .sort_values("final_mode")
)
summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

zone_summary = (
    impl.groupby(["BaseZone", "final_mode"], as_index=False)
    .agg(
        Class_n=("ClassCode", "count"),
        Area_km2=("Area_km2", "sum"),
        Target_LAI_mean=("target_LAI", "mean"),
        Tree_ratio_mean=("tree_ratio", "mean"),
        Shrub_ratio_mean=("shrub_ratio", "mean"),
        Grass_ratio_mean=("grass_ratio", "mean"),
    )
    .sort_values(["BaseZone", "final_mode"])
)
zone_summary.to_csv(OUT_ZONE_SUMMARY_CSV, index=False, encoding="utf-8-sig")


# =============================================================================
# 7. QA
# =============================================================================

qa_lines = []
qa_lines.append("========== Step6 QA ==========")
qa_lines.append(f"BestCompromise file: {BEST_CSV.name}")
qa_lines.append(f"ParetoFront file: {PARETO_CSV.name}")
qa_lines.append(f"ParetoRanges file: {RANGE_CSV.name}")
qa_lines.append(f"HydroSupport file: {HYDRO_CSV.name}")
qa_lines.append(f"ObservedStructure file: {OBS_CSV.name if OBS_CSV is not None else '未读取'}")
qa_lines.append("")
qa_lines.append(f"全区 ClassCode 总数: {impl['ClassCode'].nunique()}")
qa_lines.append(f"进入 Step5 优化的 ClassCode 数: {int(impl['optimized_flag'].sum())}")
qa_lines.append(f"未进入 Step5、在 Step6 回填的 ClassCode 数: {int((impl['optimized_flag'] == 0).sum())}")
qa_lines.append("")
qa_lines.append("最终模式统计：")
qa_lines.append(summary.to_string(index=False))
qa_lines.append("")
qa_lines.append(f"ObservedStructure 有效类数: {int((impl['ObsValid'] == 1).sum())}")

OUT_QA_TXT.write_text("\n".join(qa_lines), encoding="utf-8")

print("完成。")
print(f"输出 implementation 表: {OUT_IMPL_CSV}")
print(f"输出 map-ready 表: {OUT_MAPREADY_CSV}")
print(f"输出模式汇总表: {OUT_SUMMARY_CSV}")
print(f"输出分区汇总表: {OUT_ZONE_SUMMARY_CSV}")
print(f"输出 QA: {OUT_QA_TXT}")
