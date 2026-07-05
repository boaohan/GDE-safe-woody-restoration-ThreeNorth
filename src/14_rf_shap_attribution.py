"""
14_rf_shap_attribution.py

Runs strict random-forest attribution diagnostics and optional SHAP importance for safe-LAI and restoration outputs. SHAP is enabled for release consistency with the manuscript.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径区：只改这里
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

OUT_DIR = DATA_DIR / "attribution_diagnostics_outputs_STRICT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURE_TABLE = OUT_DIR / "S8_STRICT_feature_table.csv"
OUT_FEATURES_USED = OUT_DIR / "S8_STRICT_features_used.csv"
OUT_COLLINEARITY = OUT_DIR / "S8_STRICT_collinearity_removed.csv"
OUT_PERFORMANCE = OUT_DIR / "S8_STRICT_attribution_model_performance.csv"
OUT_IMPORTANCE = OUT_DIR / "S8_STRICT_permutation_importance.csv"
OUT_PREDICTIONS = OUT_DIR / "S8_STRICT_predictions_byClass.csv"
OUT_CONFUSION = OUT_DIR / "S8_STRICT_confusion_matrix_final_mode.csv"
OUT_MORAN = OUT_DIR / "S8_STRICT_residual_moranI.csv"
OUT_SHAP = OUT_DIR / "S8_STRICT_SHAP_importance_optional.csv"
OUT_REPORT = OUT_DIR / "S8_STRICT_attribution_diagnostics_report.md"


# ============================================================
# 1. 参数
# ============================================================

RANDOM_STATE = 202502
N_SPLITS_MAX = 5
COLLINEARITY_R = 0.80
MIN_NONMISSING_RATE = 0.60
MIN_UNIQUE_VALUES = 3

# 快速稳定版参数
N_ESTIMATORS = 300
PERMUTATION_N_REPEATS = 20
ENABLE_SHAP = True
MORAN_PERMUTATIONS = 199


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
    "observed": [
        "ThreeNorth_Class_ObservedStructure*.csv",
        "*ObservedStructure*.csv",
    ],
    "eco_function": [
        "ThreeNorth_Class_EcoFunction*.csv",
        "*EcoFunction*.csv",
    ],
    "step5_support": [
        "ThreeNorth_Class_Step5Support_RealMetrics*.csv",
        "*Step5Support*RealMetrics*.csv",
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


def read_optional(name: str) -> pd.DataFrame | None:
    path = find_latest(FILE_PATTERNS[name])
    if path is None:
        print(f"[MISSING] {name}: 未找到文件")
        return None

    print(f"[READ] {name}: {path.name}", flush=True)

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    df.attrs["source_path"] = str(path)
    return df


def read_required(name: str) -> pd.DataFrame:
    df = read_optional(name)
    if df is None:
        raise FileNotFoundError(f"缺少必需文件: {name}")
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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_group_kfold(groups: pd.Series, n_samples: int) -> tuple[object, np.ndarray | None]:
    groups = groups.fillna(-999)
    n_groups = groups.nunique()

    if n_groups >= 3:
        n_splits = min(N_SPLITS_MAX, n_groups)
        return GroupKFold(n_splits=n_splits), groups.to_numpy()

    n_splits = min(N_SPLITS_MAX, max(3, min(n_samples, 5)))
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None


def encode_categorical_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("category").cat.codes.replace(-1, np.nan)
    return out


def remove_collinear_features(X: pd.DataFrame, priority: list[str]) -> tuple[list[str], pd.DataFrame]:
    """
    Spearman |r| > COLLINEARITY_R 时剔除一个变量。
    优先保留 priority 列表靠前的变量。
    """
    if X.shape[1] <= 1:
        return list(X.columns), pd.DataFrame()

    X_num = encode_categorical_to_numeric(X)
    corr = X_num.corr(method="spearman").abs()

    cols = list(X_num.columns)
    keep = set(cols)
    removed_rows = []

    priority_rank = {v: i for i, v in enumerate(priority)}

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]

            if a not in keep or b not in keep:
                continue

            r = corr.loc[a, b]
            if pd.isna(r) or r <= COLLINEARITY_R:
                continue

            ra = priority_rank.get(a, 9999)
            rb = priority_rank.get(b, 9999)

            if ra <= rb:
                drop = b
                keep_col = a
            else:
                drop = a
                keep_col = b

            if drop in keep:
                keep.remove(drop)
                removed_rows.append({
                    "removed_feature": drop,
                    "kept_feature": keep_col,
                    "spearman_abs_r": r,
                    "reason": f"|Spearman r|>{COLLINEARITY_R}",
                })

    kept = [c for c in cols if c in keep]
    removed_df = pd.DataFrame(removed_rows)
    return kept, removed_df


def get_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    candidates = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("x", "y"),
        ("centroid_x", "centroid_y"),
        ("X", "Y"),
    ]

    for x, y in candidates:
        if x in df.columns and y in df.columns:
            return x, y

    return None, None


def moran_i_knn(x: np.ndarray, coords: np.ndarray, k: int = 8, permutations: int = MORAN_PERMUTATIONS) -> dict:
    """
    简单 KNN Moran's I。
    如果没有坐标，则跳过。
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return {
            "moran_I": np.nan,
            "p_value": np.nan,
            "note": "skipped: sklearn.neighbors unavailable",
        }

    x = np.asarray(x, dtype=float)
    coords = np.asarray(coords, dtype=float)

    mask = np.isfinite(x) & np.isfinite(coords).all(axis=1)
    x = x[mask]
    coords = coords[mask]

    n = len(x)
    if n < k + 2:
        return {
            "moran_I": np.nan,
            "p_value": np.nan,
            "note": "skipped: insufficient samples",
        }

    x0 = x - x.mean()
    denom = np.sum(x0 ** 2)
    if denom <= 0:
        return {
            "moran_I": np.nan,
            "p_value": np.nan,
            "note": "skipped: zero variance",
        }

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]

    wij_sum = n * k
    num = 0.0
    for i in range(n):
        num += np.sum(x0[i] * x0[neigh[i]])

    I_obs = (n / wij_sum) * (num / denom)

    rng = np.random.default_rng(RANDOM_STATE)
    perm_I = []

    for _ in range(permutations):
        xp = rng.permutation(x0)
        num_p = 0.0
        for i in range(n):
            num_p += np.sum(xp[i] * xp[neigh[i]])
        perm_I.append((n / wij_sum) * (num_p / denom))

    perm_I = np.asarray(perm_I)
    p = (np.sum(np.abs(perm_I) >= abs(I_obs)) + 1) / (permutations + 1)

    return {
        "moran_I": float(I_obs),
        "p_value": float(p),
        "note": f"KNN k={k}, permutations={permutations}",
    }


def filter_features(data: pd.DataFrame, features: list[str]) -> list[str]:
    out = []
    for c in features:
        if c not in data.columns:
            continue

        nonmiss = data[c].notna().mean()
        nunique = data[c].nunique(dropna=True)

        if nonmiss < MIN_NONMISSING_RATE:
            continue
        if nunique < MIN_UNIQUE_VALUES:
            continue

        out.append(c)

    return out


# ============================================================
# 4. 读取数据
# ============================================================

hydro = read_required("hydro")
lai_total = read_required("lai_total")
lai_safe = read_required("lai_safe")

mode_impl = read_optional("mode_impl")
mode_map = read_optional("mode_map")
best = read_optional("best")
observed = read_optional("observed")
eco_function = read_optional("eco_function")
step5_support = read_optional("step5_support")

mode_main = mode_impl if mode_impl is not None else mode_map
if mode_main is None:
    raise FileNotFoundError("缺少 ModeImplementation 或 ModeMapReady。")


# ============================================================
# 5. 数值字段转换
# ============================================================

numeric_cols = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "Area_km2",
    "area_km2",
    "Area",
    "Pixel_n",
    "GDE_frac_mean",
    "GDE_stability_mean",
    "GDE_persistence_mean",
    "GDE_persistence_count_mean",
    "GDE_trajectory_code_mean",
    "NaturalFrac_mean",
    "LAI_current_3yr",
    "LAI_current_3yr_mean",
    "LAI_max_total",
    "LAImax_total",
    "LAImaxTotal_q25",
    "LAI_safe_max",
    "LAI_safe_max_mean",
    "R_gde",
    "safe_ratio",
    "safe_ratio_final",
    "target_LAI",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
    "TreeFrac_obs",
    "ShrubFrac_obs",
    "GrassFrac_obs",
    "WoodyFrac_obs",
    "CurrentDensityProxy",
    "NDVI",
    "NDVI_mean",
    "NDVI_std",
    "NPP",
    "NPP_mean",
    "WUE",
    "WUE_mean",
    "WUE_raw_std",
    "P",
    "P_mean",
    "PET",
    "PET_mean",
    "AET",
    "AET_mean",
    "AI",
    "Soil",
    "Soil_mean",
    "Runoff",
    "Runoff_mean",
    "GWSA_trend",
    "GWSA_mean_period_mean",
    "SW_access",
    "SW_access_mean",
    "SW_occurrence",
    "SW_occurrence_mean",
    "Elevation",
    "Elevation_mean",
    "Slope",
    "Slope_mean",
    "BareSoilFrac",
    "BareSoilFrac_mean",
    "BareSoilFrac_std",
    "WindErosionRisk",
    "Wind10_spring_mean",
    "SandConnectivity",
    "SandConnectivity_mean",
    "lon",
    "lat",
    "x",
    "y",
    "centroid_x",
    "centroid_y",
]

for df0 in [hydro, lai_total, lai_safe, mode_main, best, observed, eco_function, step5_support]:
    to_numeric_if_exists(df0, numeric_cols)


# ============================================================
# 6. 构建分析表
# ============================================================

area_col_hydro = pick_col(hydro, ["Area_km2", "area_km2", "Area"], required=False)
area_col_mode = pick_col(mode_main, ["Area_km2", "area_km2", "Area"], required=False)

base_cols = ["ClassCode"]

for c in ["BaseZone", "GDE_Level"]:
    if c in hydro.columns:
        base_cols.append(c)

if area_col_hydro is not None:
    base_cols.append(area_col_hydro)

# 只先带 HydroSupport 中可能用到的原始/半原始解释变量
raw_keywords_for_hydro = [
    "pixel",
    "gde",
    "natural",
    "lai_current",
    "ndvi",
    "npp",
    "wue",
    "p_mean",
    "pet",
    "aet",
    "soil",
    "runoff",
    "gws",
    "sw_",
    "elev",
    "slope",
    "soil",
    "sand",
    "wind",
    "bare",
    "treefrac",
    "shrubfrac",
    "grassfrac",
    "woodyfrac",
    "densityproxy",
    "lon",
    "lat",
    "centroid",
    "x",
    "y",
]

for c in hydro.columns:
    if c not in base_cols and c != "ClassCode":
        if any(k in c.lower() for k in raw_keywords_for_hydro):
            base_cols.append(c)

base_cols = list(dict.fromkeys([c for c in base_cols if c in hydro.columns]))
df = hydro[base_cols].copy()

rename = {}
if area_col_hydro is not None and area_col_hydro in df.columns:
    rename[area_col_hydro] = "Area_km2"

df = df.rename(columns=rename)

if "Area_km2" not in df.columns:
    if area_col_mode is not None:
        df = df.merge(
            mode_main[["ClassCode", area_col_mode]].rename(columns={area_col_mode: "Area_km2"}),
            on="ClassCode",
            how="left",
        )
    else:
        df["Area_km2"] = 1.0

# 合并 LAI total
lai_total_col = pick_col(
    lai_total,
    ["LAI_max_total", "LAImax_total", "LAImaxTotal_q25", "LAI_max_total_q25"],
    required=True,
)

df = df.merge(
    lai_total[["ClassCode", lai_total_col]].rename(columns={lai_total_col: "LAImaxTotal"}),
    on="ClassCode",
    how="left",
)

# 合并 LAI safe
lai_safe_col = pick_col(
    lai_safe,
    ["LAI_safe_max", "LAI_safe_max_mean"],
    required=True,
)

safe_cols = ["ClassCode", lai_safe_col]

for c in [
    "R_gde",
    "R_gde_raw",
    "safe_ratio",
    "safe_ratio_final",
    "GDE_frac_mean",
    "GDE_stability_mean",
    "GDE_persistence_mean",
    "GDE_persistence_count_mean",
    "GDE_trajectory_code_mean",
]:
    if c in lai_safe.columns and c not in safe_cols:
        safe_cols.append(c)

df = df.merge(
    lai_safe[safe_cols].rename(columns={lai_safe_col: "LAI_safe_max"}),
    on="ClassCode",
    how="left",
    suffixes=("", "_safe"),
)

# 合并 mode
final_mode_col = pick_col(mode_main, ["final_mode", "FinalMode", "mode"], required=True)

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
    "TreeFrac_obs",
    "ShrubFrac_obs",
    "GrassFrac_obs",
    "WoodyFrac_obs",
]:
    if c in mode_main.columns:
        mode_cols.append(c)

mode_cols = list(dict.fromkeys(mode_cols))

mode_use = mode_main[mode_cols].copy()
mode_use = mode_use.rename(columns={
    final_mode_col: "final_mode",
})

mode_use["final_mode"] = mode_use["final_mode"].map(normalize_mode)

df = df.merge(mode_use, on="ClassCode", how="left", suffixes=("", "_mode"))

# 用 BestCompromise 补优化单元 target / ratio / density
if best is not None:
    best_cols = ["ClassCode"]

    for c in [
        "target_LAI",
        "tree_ratio",
        "shrub_ratio",
        "grass_ratio",
        "tree_density_per_ha",
        "shrub_density_per_ha",
        "grass_cover_target_pct",
    ]:
        if c in best.columns:
            best_cols.append(c)

    best_use = best[best_cols].copy()
    best_use = best_use.rename(columns={
        "target_LAI": "target_LAI_best",
        "tree_ratio": "tree_ratio_best",
        "shrub_ratio": "shrub_ratio_best",
        "grass_ratio": "grass_ratio_best",
        "tree_density_per_ha": "tree_density_best",
        "shrub_density_per_ha": "shrub_density_best",
        "grass_cover_target_pct": "grass_cover_best",
    })

    df = df.merge(best_use, on="ClassCode", how="left")

    for a, b in [
        ("target_LAI", "target_LAI_best"),
        ("tree_ratio", "tree_ratio_best"),
        ("shrub_ratio", "shrub_ratio_best"),
        ("grass_ratio", "grass_ratio_best"),
        ("tree_density_per_ha", "tree_density_best"),
        ("shrub_density_per_ha", "shrub_density_best"),
        ("grass_cover_target_pct", "grass_cover_best"),
    ]:
        if a in df.columns and b in df.columns:
            df[a] = df[a].fillna(df[b])
        elif b in df.columns:
            df[a] = df[b]

# 合并 observed / eco / step5_support 中尚未存在的列
for extra_name, extra in [
    ("observed", observed),
    ("eco_function", eco_function),
    ("step5_support", step5_support),
]:
    if extra is not None and "ClassCode" in extra.columns:
        add_cols = [c for c in extra.columns if c != "ClassCode" and c not in df.columns]
        if add_cols:
            df = df.merge(extra[["ClassCode"] + add_cols], on="ClassCode", how="left")

# 响应变量
df["LAI_reduction_ratio"] = 1 - df["LAI_safe_max"] / df["LAImaxTotal"].replace(0, np.nan)
df["LAI_reduction_ratio"] = df["LAI_reduction_ratio"].clip(0, 1)

if "optimized_flag" not in df.columns:
    df["optimized_flag"] = np.where(df["tree_ratio"].notna(), 1, 0)

df.to_csv(OUT_FEATURE_TABLE, index=False, encoding="utf-8-sig")


# ============================================================
# 7. 严格机制变量和审计变量
# ============================================================

# 严格排除：决策、候选库、优化、模式派生变量
blocked_mechanism_keywords = [
    # 决策 / 安全阈值派生变量
    "LAI_safe",
    "LAImaxTotal",
    "LAI_max_total",
    "LAImax_total",
    "R_gde",
    "safe_ratio",
    "target_LAI",
    "LAI_margin",
    "Restoration_gap_safe",
    "Over_safe_capacity",
    "scheme_LAI_capacity",

    # 最终配置和密度阈值
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density",
    "shrub_density",
    "grass_cover",

    # 优化 / 候选库 / 打分变量
    "best",
    "score",
    "obj",
    "candidate",
    "Pareto",
    "scheme",
    "template",
    "species",
    "priority",
    "risk_level",
    "constraint_mode",
    "allowed_modes",
    "tree_allowed",
    "management",
    "obs_support",

    # 响应变量或模式结果
    "final_mode",
    "mode",
    "backfill",
    "optimized",
    "LAI_reduction_ratio",
    "veg_zone_primary",
]

# 严格允许：原始或半原始环境、水文、GDE、现状结构变量
candidate_feature_keywords = [
    "Pixel_n",

    "GDE_frac",
    "GDE_stability",
    "GDE_persistence",
    "GDE_trajectory",

    "NaturalFrac",
    "LAI_current",

    "NDVI",
    "NPP",
    "WUE",

    "P_mean",
    "PET",
    "AET",
    "AI",
    "Soil",
    "Runoff",
    "GWSA",
    "SW_",

    "Elevation",
    "Slope",
    "BareSoil",
    "Wind",
    "Sand",

    "TreeFrac_obs",
    "ShrubFrac_obs",
    "GrassFrac_obs",
    "WoodyFrac_obs",
    "CurrentDensityProxy",

    "lon",
    "lat",
    "centroid_x",
    "centroid_y",
    "x",
    "y",
]

mechanism_features = []

for c in df.columns:
    if c in ["ClassCode", "Area_km2", "BaseZone", "GDE_Level"]:
        continue

    if any(k.lower() in c.lower() for k in blocked_mechanism_keywords):
        continue

    if any(k.lower() in c.lower() for k in candidate_feature_keywords):
        mechanism_features.append(c)

# 审计变量：只用于 decision-audit，不作为生态归因
audit_features = []

for c in [
    "LAImaxTotal",
    "LAI_safe_max",
    "R_gde",
    "R_gde_raw",
    "safe_ratio",
    "safe_ratio_final",
    "target_LAI",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
]:
    if c in df.columns:
        audit_features.append(c)

mechanism_features = filter_features(df, mechanism_features)
audit_features = filter_features(df, audit_features)

priority = [
    "GDE_frac_mean",
    "GDE_stability_mean",
    "GDE_persistence_mean",
    "GDE_persistence_count_mean",
    "GDE_trajectory_code_mean",
    "NaturalFrac_mean",
    "LAI_current_3yr",
    "LAI_current_3yr_mean",
    "TreeFrac_obs",
    "ShrubFrac_obs",
    "GrassFrac_obs",
    "WoodyFrac_obs",
    "CurrentDensityProxy",
    "PET_mean",
    "AET_mean",
    "P_mean",
    "Soil_mean",
    "Runoff_mean",
    "GWSA_mean_period_mean",
    "SW_access_mean",
    "SW_occurrence_mean",
    "NDVI_mean",
    "NDVI_std",
    "NPP_mean",
    "WUE_mean",
    "WUE_raw_std",
    "BareSoilFrac_mean",
    "BareSoilFrac_std",
    "Wind10_spring_mean",
    "WindErosionRisk",
    "SandConnectivity",
    "SandConnectivity_mean",
    "Elevation_mean",
    "Slope_mean",
    "Pixel_n",
]

if mechanism_features:
    X_mech_raw = df[mechanism_features].copy()
    mechanism_features_kept, collinear_removed = remove_collinear_features(X_mech_raw, priority)
else:
    mechanism_features_kept = []
    collinear_removed = pd.DataFrame()

collinear_removed.to_csv(OUT_COLLINEARITY, index=False, encoding="utf-8-sig")

features_used = pd.DataFrame({
    "feature": mechanism_features_kept + audit_features,
    "feature_set": ["STRICT_mechanism_no_decision_variables"] * len(mechanism_features_kept)
                   + ["decision_audit_not_ecological_attribution"] * len(audit_features),
})
features_used.to_csv(OUT_FEATURES_USED, index=False, encoding="utf-8-sig")

print("[INFO] STRICT mechanism features:", mechanism_features_kept, flush=True)
print("[INFO] Audit features:", audit_features, flush=True)


# ============================================================
# 8. 模型函数
# ============================================================

def run_classification_model(
    data: pd.DataFrame,
    y_col: str,
    features: list[str],
    model_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data.loc[data[y_col].notna()].copy()
    d = d.loc[d[y_col].astype(str).ne("Unknown")].copy()

    if len(d) < 30 or d[y_col].nunique() < 2:
        perf = pd.DataFrame([{
            "model": model_name,
            "feature_set": feature_set,
            "task": "classification",
            "n": len(d),
            "status": "skipped_insufficient_data",
        }])
        return perf, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X = encode_categorical_to_numeric(d[features])
    y_raw = d[y_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    groups = d["BaseZone"] if "BaseZone" in d.columns else pd.Series(np.arange(len(d)))
    cv, group_array = safe_group_kfold(groups, len(d))

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])

    y_pred = np.full_like(y, fill_value=-1)
    fold_rows = []

    if group_array is not None:
        split_iter = cv.split(X, y, groups=group_array)
    else:
        split_iter = cv.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        y_pred[test_idx] = pred

        fold_rows.append({
            "model": model_name,
            "feature_set": feature_set,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, average="macro"),
            "accuracy": accuracy_score(y[test_idx], pred),
        })

    perf = pd.DataFrame(fold_rows)

    perf_summary = pd.DataFrame([{
        "model": model_name,
        "feature_set": feature_set,
        "task": "classification",
        "n": len(d),
        "n_classes": len(le.classes_),
        "status": "ok",
        "balanced_accuracy_mean": perf["balanced_accuracy"].mean(),
        "balanced_accuracy_sd": perf["balanced_accuracy"].std(),
        "macro_f1_mean": perf["macro_f1"].mean(),
        "macro_f1_sd": perf["macro_f1"].std(),
        "accuracy_mean": perf["accuracy"].mean(),
        "accuracy_sd": perf["accuracy"].std(),
    }])

    pipe.fit(X, y)

    perm = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="balanced_accuracy",
    )

    imp = pd.DataFrame({
        "model": model_name,
        "feature_set": feature_set,
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_sd": perm.importances_std,
        "importance_metric": "permutation_balanced_accuracy_decrease",
    }).sort_values("importance_mean", ascending=False)

    pred_df = d[["ClassCode", "Area_km2", "BaseZone"]].copy()
    pred_df["model"] = model_name
    pred_df["feature_set"] = feature_set
    pred_df["observed"] = y_raw.to_numpy()
    pred_df["predicted"] = le.inverse_transform(y_pred)
    pred_df["correct"] = pred_df["observed"].eq(pred_df["predicted"]).astype(int)
    pred_df["residual"] = 1 - pred_df["correct"]

    cm = confusion_matrix(y_raw, pred_df["predicted"], labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_long = cm_df.reset_index().melt(id_vars="index", var_name="predicted", value_name="n")
    cm_long = cm_long.rename(columns={"index": "observed"})
    cm_long["model"] = model_name
    cm_long["feature_set"] = feature_set

    return perf_summary, imp, pred_df, cm_long


def run_regression_model(
    data: pd.DataFrame,
    y_col: str,
    features: list[str],
    model_name: str,
    feature_set: str,
    subset_expr: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data.copy()

    if subset_expr is not None:
        d = d.query(subset_expr).copy()

    d = d.loc[d[y_col].notna()].copy()

    if len(d) < 30:
        perf = pd.DataFrame([{
            "model": model_name,
            "feature_set": feature_set,
            "task": "regression",
            "n": len(d),
            "status": "skipped_insufficient_data",
        }])
        return perf, pd.DataFrame(), pd.DataFrame()

    X = encode_categorical_to_numeric(d[features])
    y = pd.to_numeric(d[y_col], errors="coerce").to_numpy()

    mask = np.isfinite(y)
    X = X.loc[mask]
    d = d.loc[mask].copy()
    y = y[mask]

    groups = d["BaseZone"] if "BaseZone" in d.columns else pd.Series(np.arange(len(d)))
    cv, group_array = safe_group_kfold(groups, len(d))

    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", reg),
    ])

    y_pred = np.full(len(y), np.nan)
    fold_rows = []

    if group_array is not None:
        split_iter = cv.split(X, y, groups=group_array)
    else:
        split_iter = cv.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        y_pred[test_idx] = pred

        fold_rows.append({
            "model": model_name,
            "feature_set": feature_set,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "R2": r2_score(y[test_idx], pred),
            "RMSE": rmse(y[test_idx], pred),
            "MAE": mean_absolute_error(y[test_idx], pred),
        })

    fold_perf = pd.DataFrame(fold_rows)

    perf_summary = pd.DataFrame([{
        "model": model_name,
        "feature_set": feature_set,
        "task": "regression",
        "response": y_col,
        "n": len(d),
        "status": "ok",
        "R2_mean": fold_perf["R2"].mean(),
        "R2_sd": fold_perf["R2"].std(),
        "RMSE_mean": fold_perf["RMSE"].mean(),
        "RMSE_sd": fold_perf["RMSE"].std(),
        "MAE_mean": fold_perf["MAE"].mean(),
        "MAE_sd": fold_perf["MAE"].std(),
    }])

    pipe.fit(X, y)

    perm = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="r2",
    )

    imp = pd.DataFrame({
        "model": model_name,
        "feature_set": feature_set,
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_sd": perm.importances_std,
        "importance_metric": "permutation_R2_decrease",
    }).sort_values("importance_mean", ascending=False)

    pred_df = d[["ClassCode", "Area_km2", "BaseZone"]].copy()
    pred_df["model"] = model_name
    pred_df["feature_set"] = feature_set
    pred_df["observed"] = y
    pred_df["predicted"] = y_pred
    pred_df["residual"] = y - y_pred

    return perf_summary, imp, pred_df


# ============================================================
# 9. 运行模型
# ============================================================

performance_list = []
importance_list = []
prediction_list = []
confusion_list = []

if len(mechanism_features_kept) == 0:
    raise ValueError("严格机制归因模型没有可用解释变量，请检查输入表字段。")

print("[RUN] Model A: final_mode STRICT mechanism classification", flush=True)
perf, imp, pred, cm = run_classification_model(
    data=df,
    y_col="final_mode",
    features=mechanism_features_kept,
    model_name="Model_A_final_mode_STRICT_mechanism",
    feature_set="STRICT_mechanism_no_decision_variables",
)
performance_list.append(perf)
importance_list.append(imp)
prediction_list.append(pred)
confusion_list.append(cm)

print("[RUN] Model B: LAI_reduction STRICT mechanism regression", flush=True)
perf, imp, pred = run_regression_model(
    data=df,
    y_col="LAI_reduction_ratio",
    features=mechanism_features_kept,
    model_name="Model_B_LAI_reduction_STRICT_mechanism",
    feature_set="STRICT_mechanism_no_decision_variables",
)
performance_list.append(perf)
importance_list.append(imp)
prediction_list.append(pred)

print("[RUN] Model C: tree_ratio STRICT mechanism regression", flush=True)
if "tree_ratio" in df.columns:
    perf, imp, pred = run_regression_model(
        data=df,
        y_col="tree_ratio",
        features=mechanism_features_kept,
        model_name="Model_C_tree_ratio_STRICT_mechanism",
        feature_set="STRICT_mechanism_no_decision_variables",
        subset_expr="optimized_flag == 1",
    )
    performance_list.append(perf)
    importance_list.append(imp)
    prediction_list.append(pred)

print("[RUN] Audit model: final_mode with decision variables", flush=True)
if len(audit_features) >= 2:
    perf, imp, pred, cm = run_classification_model(
        data=df,
        y_col="final_mode",
        features=audit_features,
        model_name="Audit_final_mode_decision_variables",
        feature_set="decision_audit_not_ecological_attribution",
    )
    performance_list.append(perf)
    importance_list.append(imp)
    prediction_list.append(pred)
    confusion_list.append(cm)


performance = pd.concat(performance_list, ignore_index=True)

importance_nonempty = [x for x in importance_list if len(x) > 0]
importance = pd.concat(importance_nonempty, ignore_index=True) if importance_nonempty else pd.DataFrame()

prediction_nonempty = [x for x in prediction_list if len(x) > 0]
predictions = pd.concat(prediction_nonempty, ignore_index=True) if prediction_nonempty else pd.DataFrame()

confusion_nonempty = [x for x in confusion_list if len(x) > 0]
confusion_all = pd.concat(confusion_nonempty, ignore_index=True) if confusion_nonempty else pd.DataFrame()


# ============================================================
# 10. 残差空间自相关 Moran's I
# ============================================================

moran_rows = []

xcol, ycol = get_coord_columns(df)

if xcol is not None and ycol is not None and len(predictions) > 0:
    coord_df = df[["ClassCode", xcol, ycol]].copy()
    pred_coord = predictions.merge(coord_df, on="ClassCode", how="left")

    for model_name, g in pred_coord.groupby("model"):
        coords = g[[xcol, ycol]].to_numpy()
        residual = pd.to_numeric(g["residual"], errors="coerce").to_numpy()

        m = moran_i_knn(residual, coords, k=8, permutations=MORAN_PERMUTATIONS)

        moran_rows.append({
            "model": model_name,
            "coord_x": xcol,
            "coord_y": ycol,
            "moran_I": m["moran_I"],
            "p_value": m["p_value"],
            "note": m["note"],
        })
else:
    moran_rows.append({
        "model": "all",
        "coord_x": "",
        "coord_y": "",
        "moran_I": np.nan,
        "p_value": np.nan,
        "note": "skipped: no coordinate columns found",
    })

moran_df = pd.DataFrame(moran_rows)


# ============================================================
# 11. 可选 SHAP
# 默认关闭，避免运行过慢
# ============================================================

if ENABLE_SHAP:
    try:
        import shap

        shap_rows = []

        for model_name, y_col, subset_expr in [
            ("Model_B_LAI_reduction_STRICT_mechanism", "LAI_reduction_ratio", None),
            ("Model_C_tree_ratio_STRICT_mechanism", "tree_ratio", "optimized_flag == 1"),
        ]:
            d = df.copy()
            if subset_expr is not None:
                d = d.query(subset_expr).copy()
            d = d.loc[d[y_col].notna()].copy()

            if len(d) >= 30:
                X = encode_categorical_to_numeric(d[mechanism_features_kept])
                y = pd.to_numeric(d[y_col], errors="coerce")

                mask = y.notna()
                X = X.loc[mask]
                y = y.loc[mask]

                imputer = SimpleImputer(strategy="median")
                X_imp = pd.DataFrame(
                    imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index,
                )

                reg = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                reg.fit(X_imp, y)

                explainer = shap.TreeExplainer(reg)
                sv = explainer.shap_values(X_imp)
                mean_abs = np.abs(sv).mean(axis=0)

                for f, val in zip(X_imp.columns, mean_abs):
                    shap_rows.append({
                        "model": model_name,
                        "feature_set": "STRICT_mechanism_no_decision_variables",
                        "feature": f,
                        "mean_abs_shap": val,
                    })

        shap_df = pd.DataFrame(shap_rows).sort_values(
            ["model", "mean_abs_shap"],
            ascending=[True, False],
        )

    except Exception as e:
        shap_df = pd.DataFrame([{
            "model": "SHAP_optional",
            "feature_set": "STRICT_mechanism_no_decision_variables",
            "feature": "",
            "mean_abs_shap": np.nan,
            "note": f"skipped: {e}",
        }])
else:
    shap_df = pd.DataFrame([{
        "model": "SHAP_optional",
        "feature_set": "STRICT_mechanism_no_decision_variables",
        "feature": "",
        "mean_abs_shap": np.nan,
        "note": "skipped: SHAP was not generated; check shap installation or ENABLE_SHAP setting",
    }])


# ============================================================
# 12. 输出
# ============================================================

performance.to_csv(OUT_PERFORMANCE, index=False, encoding="utf-8-sig")
importance.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")
predictions.to_csv(OUT_PREDICTIONS, index=False, encoding="utf-8-sig")
confusion_all.to_csv(OUT_CONFUSION, index=False, encoding="utf-8-sig")
moran_df.to_csv(OUT_MORAN, index=False, encoding="utf-8-sig")
shap_df.to_csv(OUT_SHAP, index=False, encoding="utf-8-sig")


# ============================================================
# 13. Markdown 报告
# ============================================================

lines = []

lines.append("# S8 STRICT attribution model diagnostics report")
lines.append("")
lines.append("## Definition")
lines.append("")
lines.append(
    "STRICT mechanism attribution models used only raw or semi-raw environmental, hydroclimatic, vegetation and GDE descriptors. "
    "Decision-derived variables, including LAI_safe_max, R_gde, safe_ratio, target_LAI, tree_ratio, density thresholds, candidate-library descriptors, management-priority labels, template groups, risk-level labels and optimization scores, were excluded from mechanism models to avoid circular attribution."
)
lines.append("")
lines.append(
    "A separate audit model was allowed to use decision-derived variables, but it is interpreted only as a decision-consistency check and not as ecological attribution."
)
lines.append("")
lines.append(
    f"Random forest settings: n_estimators={N_ESTIMATORS}, permutation repeats={PERMUTATION_N_REPEATS}, spatial grouping by BaseZone where available."
)
lines.append("")

lines.append("## Features used")
lines.append("")
lines.append("STRICT mechanism features:")
lines.append(", ".join(mechanism_features_kept))
lines.append("")
lines.append("Audit features:")
lines.append(", ".join(audit_features))
lines.append("")

lines.append("## Collinearity filtering")
lines.append("")
if len(collinear_removed) > 0:
    lines.append(collinear_removed.to_string(index=False))
else:
    lines.append("No features removed by collinearity filtering.")
lines.append("")

lines.append("## Model performance")
lines.append("")
lines.append(performance.to_string(index=False))
lines.append("")

lines.append("## Top permutation importance")
lines.append("")
if len(importance) > 0:
    top_imp = importance.groupby("model").head(10)
    lines.append(top_imp.to_string(index=False))
else:
    lines.append("No permutation importance results.")
lines.append("")

lines.append("## Residual spatial autocorrelation")
lines.append("")
lines.append(moran_df.to_string(index=False))
lines.append("")

lines.append("## Interpretation guidance")
lines.append("")
lines.append(
    "For the manuscript, STRICT mechanism-model results should be described as robust associations rather than causal effects. "
    "Because decision-derived variables were removed, the STRICT models are suitable for ecological attribution. "
    "The audit model should be reported separately as a decision-chain consistency check."
)
lines.append("")
lines.append(
    "If the STRICT mechanism models have lower performance than the decision-aware diagnostic model, this is expected and should be interpreted as evidence that decision variables encode part of the restoration workflow."
)

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


print("STRICT 归因模型诊断完成。")
print(f"特征表: {OUT_FEATURE_TABLE}")
print(f"使用特征: {OUT_FEATURES_USED}")
print(f"共线性剔除: {OUT_COLLINEARITY}")
print(f"模型性能: {OUT_PERFORMANCE}")
print(f"变量重要性: {OUT_IMPORTANCE}")
print(f"预测残差: {OUT_PREDICTIONS}")
print(f"混淆矩阵: {OUT_CONFUSION}")
print(f"Moran's I: {OUT_MORAN}")
print(f"SHAP 可选结果: {OUT_SHAP}")
print(f"报告: {OUT_REPORT}")

print("\n模型性能：")
print(performance.to_string(index=False))

print("\nTop permutation importance：")
if len(importance) > 0:
    print(importance.groupby("model").head(10).to_string(index=False))
