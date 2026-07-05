"""
11_qa_qc_checks.py

Runs rule-based QA/QC checks for class tables, candidates, Pareto outputs, and final restoration modes.

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

OUT_DIR = DATA_DIR / "QA_QC_outputs"
FAIL_DIR = OUT_DIR / "S7a_fail_records"

OUT_QA = OUT_DIR / "S7a_QA_QC_checks.csv"
OUT_QA_PAPER = OUT_DIR / "S7a_QA_QC_summary_for_paper.csv"
OUT_REPORT = OUT_DIR / "S7a_QA_QC_report.md"

TOL = 1e-6
RATIO_TOL = 1e-3
AREA_TOL = 1e-2

# 用于候选库 target_LAI 与 LAI_safe_max 的小数舍入容差
LAI_TOL_CANDIDATE = 1e-4

OUT_DIR.mkdir(parents=True, exist_ok=True)
FAIL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 文件匹配规则
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
    "candidate": [
        "ThreeNorth_Class_CandidateLibrary*.csv",
        "*CandidateLibrary*.csv",
    ],
    "candidate_count": [
        "Fig4a_candidate_count_byClass*.csv",
        "Fig4b_candidate_count_byClass*.csv",
    ],
    "feasibility": [
        "Fig4d_feasibility_status_byClass*.csv",
        "*feasibility_status*.csv",
    ],
    "best": [
        "ThreeNorth_Class_MOO_BestCompromise*.csv",
        "*BestCompromise*.csv",
    ],
    "pareto": [
        "ThreeNorth_Class_MOO_ParetoFront*.csv",
        "*ParetoFront*.csv",
    ],
    "ranges": [
        "ThreeNorth_Class_MOO_ParetoRanges_byClass*.csv",
        "*ParetoRanges*.csv",
    ],
    "mode_map": [
        "ThreeNorth_Class_ModeMapReady*.csv",
        "Fig6_ModeMapReady*.csv",
        "*ModeMapReady*.csv",
    ],
    "mode_impl": [
        "ThreeNorth_Class_ModeImplementation*.csv",
        "*ModeImplementation*.csv",
    ],
    "mode_area": [
        "Fig6_mode_area_summary_all_units*.csv",
        "*mode_area_summary*.csv",
    ],
    "mode_gde": [
        "Fig6_mode_by_GDE_stacked_all_modes*.csv",
        "*mode_by_GDE*.csv",
    ],
    "no_feasible": [
        "Fig6_no_feasible_mode_CHECK*.csv",
        "*no_feasible*CHECK*.csv",
    ],
    "mapcode_check": [
        "Fig6_ClassCode_MapCode_CHECK*.csv",
        "*ClassCode_MapCode_CHECK*.csv",
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
# 2. 工具函数
# ============================================================

def find_latest(patterns: list[str]) -> Path | None:
    files = []
    for pat in patterns:
        files.extend(DATA_DIR.glob(pat))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not files:
        return None
    return files[-1]


def read_csv_optional(name: str) -> pd.DataFrame | None:
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


def to_num(df: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame | None:
    if df is None:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_col(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120]


qa_rows = []


def add_check(
    module: str,
    check_item: str,
    expected_rule: str,
    total: int | float | None,
    fail_df: pd.DataFrame | None,
    severity: str = "major",
    note: str = "",
):
    if total is None:
        total = np.nan

    if fail_df is None:
        n_fail = np.nan
        fail_file = ""
        status = "SKIPPED"
        pass_rate = np.nan
    else:
        n_fail = int(len(fail_df))
        fail_file = ""

        if n_fail > 0:
            fname = sanitize_filename(f"{module}_{check_item}") + ".csv"
            fail_path = FAIL_DIR / fname
            fail_df.to_csv(fail_path, index=False, encoding="utf-8-sig")
            fail_file = str(fail_path)

        if total == 0 or pd.isna(total):
            pass_rate = np.nan
        else:
            pass_rate = 1 - n_fail / float(total)

        status = "PASS" if n_fail == 0 else "FAIL"

    qa_rows.append({
        "module": module,
        "check_item": check_item,
        "expected_rule": expected_rule,
        "n_total": total,
        "n_fail": n_fail,
        "pass_rate": pass_rate,
        "severity": severity,
        "status": status,
        "fail_record_file": fail_file,
        "note": note,
    })


def missing_cols_df(
    df: pd.DataFrame | None,
    required: list[str],
    table_name: str,
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({"missing_column": required, "table": table_name})
    missing = [c for c in required if c not in df.columns]
    return pd.DataFrame({"missing_column": missing, "table": table_name})


def nonempty_fail(
    condition: pd.Series,
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    if cols is None:
        cols = list(df.columns)
    cols = [c for c in cols if c in df.columns]
    return df.loc[condition, cols].copy()


def ratio_closed(df: pd.DataFrame, tol: float = RATIO_TOL) -> pd.Series:
    return (
        df["tree_ratio"].fillna(0)
        + df["shrub_ratio"].fillna(0)
        + df["grass_ratio"].fillna(0)
        - 1
    ).abs() <= tol


def is_conservation_row(df: pd.DataFrame) -> pd.Series:
    """
    判断是否为保育/封育/自然恢复占位记录。
    这些记录不是人工乔灌草配置候选，不应按人工组合的 LAI 和比例闭合规则硬判。
    """
    flags = pd.Series(False, index=df.index)

    text_cols = [
        "veg_zone_primary",
        "template_group",
        "source_note",
        "management_suggestion",
        "final_mode",
        "backfill_reason",
    ]

    for c in text_cols:
        if c in df.columns:
            flags = flags | df[c].astype(str).str.contains(
                "保育|封育|自然恢复|conservation|exclosure|Herbaceous",
                case=False,
                na=False,
            )

    return flags


def allowed_final_mode_mask(s: pd.Series) -> pd.Series:
    txt = s.astype(str)

    patterns = [
        "Tree–shrub–grass restoration",
        "Tree-shrub-grass restoration",
        "Shrub–grass restoration",
        "Shrub-grass restoration",
        "Herbaceous–exclosure recovery",
        "Herbaceous-exclosure recovery",
        "Close-to-nature stand management",
        "乔",
        "灌草",
        "草本",
        "封育",
        "近自然",
    ]

    mask = pd.Series(False, index=s.index)
    for p in patterns:
        mask = mask | txt.str.contains(re.escape(p), case=False, na=False)

    return mask


def check_pareto_not_dominated(pareto: pd.DataFrame) -> pd.DataFrame:
    """
    在 ParetoFront 内部检查是否存在被支配解。

    注意：
    这个检查使用 score_* 归一化字段时，是 score-space 诊断。
    如果 ParetoFront 是按原始目标或近前沿集合导出的，这个检查可能出现少量 dominated warning。
    因此在主 QA 中设为 minor diagnostic，而不是阻断性错误。
    """
    score_cols = [
        "score_sand",
        "score_stability",
        "score_water_security",
        "score_gde_safety",
    ]

    obj_cols = [
        "obj1_sand_benefit",
        "obj2_eco_stability",
        "obj3_water_pressure",
        "obj4_gde_risk",
    ]

    if all(c in pareto.columns for c in score_cols):
        use_cols = score_cols
        X_all = pareto[["ClassCode"] + use_cols].copy()
        X_all[use_cols] = X_all[use_cols].apply(pd.to_numeric, errors="coerce")

    elif all(c in pareto.columns for c in obj_cols):
        use_cols = obj_cols
        X_all = pareto[["ClassCode"] + obj_cols].copy()
        X_all[obj_cols] = X_all[obj_cols].apply(pd.to_numeric, errors="coerce")

        # 转成越大越好
        X_all["obj3_water_pressure"] = -X_all["obj3_water_pressure"]
        X_all["obj4_gde_risk"] = -X_all["obj4_gde_risk"]

    else:
        return pd.DataFrame()

    dominated_idx = []

    for cc, g in X_all.groupby("ClassCode"):
        g2 = g.dropna(subset=use_cols)
        if len(g2) <= 1:
            continue

        vals = g2[use_cols].to_numpy(dtype=float)
        idxs = g2.index.to_numpy()

        for i in range(len(vals)):
            vi = vals[i]
            others = np.delete(vals, i, axis=0)

            # 其他点所有目标 >= 当前点，且至少一个 >
            ge_all = (others >= vi - 1e-12).all(axis=1)
            gt_one = (others > vi + 1e-12).any(axis=1)

            if np.any(ge_all & gt_one):
                dominated_idx.append(idxs[i])

    if not dominated_idx:
        return pd.DataFrame()

    return pareto.loc[dominated_idx].copy()


# ============================================================
# 3. 读取所有表
# ============================================================

tables = {name: read_csv_optional(name) for name in FILE_PATTERNS.keys()}

hydro = tables["hydro"]
lai_total = tables["lai_total"]
lai_safe = tables["lai_safe"]
cand = tables["candidate"]
best = tables["best"]
pareto = tables["pareto"]
ranges = tables["ranges"]
mode_map = tables["mode_map"]
mode_impl = tables["mode_impl"]
mode_area = tables["mode_area"]
mode_gde = tables["mode_gde"]
no_feasible = tables["no_feasible"]
mapcode_check = tables["mapcode_check"]


# ============================================================
# 4. 类型转换
# ============================================================

numeric_cols = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "Pixel_n",
    "Area_km2",
    "LAI_max_total",
    "LAImax_total",
    "LAImaxTotal_q25",
    "LAI_safe_max",
    "LAI_current_3yr",
    "LAI_current_3yr_mean",
    "safe_ratio",
    "safe_ratio_final",
    "R_gde",
    "target_LAI",
    "LAI_margin",
    "scheme_LAI_capacity",
    "tree_ratio",
    "shrub_ratio",
    "grass_ratio",
    "tree_density_per_ha",
    "shrub_density_per_ha",
    "grass_cover_target_pct",
    "Pareto_n",
    "Tree_ratio_min",
    "Tree_ratio_max",
    "Shrub_ratio_min",
    "Shrub_ratio_max",
    "Grass_ratio_min",
    "Grass_ratio_max",
    "LAI_target_min",
    "LAI_target_max",
    "Tree_density_min",
    "Tree_density_max",
    "Shrub_density_min",
    "Shrub_density_max",
    "Grass_cover_min",
    "Grass_cover_max",
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
    "MapCode",
]

for df in [
    hydro,
    lai_total,
    lai_safe,
    cand,
    best,
    pareto,
    ranges,
    mode_map,
    mode_impl,
    mode_area,
    mode_gde,
    no_feasible,
    mapcode_check,
]:
    to_num(df, numeric_cols)


# ============================================================
# 5. 文件和字段完整性检查
# ============================================================

core_tables = {
    "hydro": ["ClassCode", "BaseZone", "GDE_Level", "Area_km2"],
    "candidate": [
        "ClassCode",
        "scheme_id",
        "GDE_Level",
        "LAI_safe_max",
        "target_LAI",
        "tree_ratio",
        "shrub_ratio",
        "grass_ratio",
    ],
    "best": [
        "ClassCode",
        "GDE_Level",
        "target_LAI",
        "LAI_safe_max",
        "tree_ratio",
        "shrub_ratio",
        "grass_ratio",
    ],
    "pareto": ["ClassCode"],
    "ranges": ["ClassCode"],
    "mode_map": ["ClassCode", "final_mode", "GDE_Level", "Area_km2"],
    "mode_area": [],
    "mode_gde": [],
}

for name, required in core_tables.items():
    df = tables[name]

    if df is None:
        add_check(
            module="file_inventory",
            check_item=f"{name} file exists",
            expected_rule="required file should exist",
            total=1,
            fail_df=pd.DataFrame({"missing_file_key": [name]}),
            severity="critical",
        )
    else:
        add_check(
            module="file_inventory",
            check_item=f"{name} file exists",
            expected_rule="required file should exist",
            total=1,
            fail_df=pd.DataFrame(),
            severity="critical",
            note=df.attrs.get("source_path", ""),
        )

    miss = missing_cols_df(df, required, name)
    add_check(
        module="schema",
        check_item=f"{name} required columns",
        expected_rule=f"required columns present: {required}",
        total=len(required),
        fail_df=miss,
        severity="critical" if name in ["hydro", "candidate", "best", "mode_map"] else "major",
    )


# ============================================================
# 6. HydroSupport / 空间单元检查
# ============================================================

if hydro is not None and {"ClassCode", "GDE_Level"}.issubset(hydro.columns):
    total = len(hydro)

    add_check(
        "spatial_unit",
        "HydroSupport ClassCode unique",
        "ClassCode should be unique in HydroSupport",
        total,
        hydro.loc[
            hydro["ClassCode"].duplicated(keep=False),
            ["ClassCode", "BaseZone", "GDE_Level"],
        ].copy(),
        "critical",
    )

    add_check(
        "spatial_unit",
        "GDE_Level valid",
        "GDE_Level must be one of 0, 1, 2, 3",
        total,
        nonempty_fail(
            ~hydro["GDE_Level"].isin([0, 1, 2, 3]),
            hydro,
            ["ClassCode", "BaseZone", "GDE_Level"],
        ),
        "critical",
    )

    if "Area_km2" in hydro.columns:
        add_check(
            "spatial_unit",
            "Area positive",
            "Area_km2 must be positive",
            total,
            nonempty_fail(
                ~(hydro["Area_km2"] > 0),
                hydro,
                ["ClassCode", "Area_km2"],
            ),
            "critical",
        )

    if {"ClassCode", "BaseZone", "GDE_Level"}.issubset(hydro.columns):
        relation = (
            hydro["ClassCode"]
            - (hydro["BaseZone"] * 10 + hydro["GDE_Level"])
        ).abs() <= TOL

        add_check(
            "spatial_unit",
            "Refined stratum coding relation",
            "For refined modelling strata, ClassCode should equal BaseZone*10 + GDE_Level",
            total,
            nonempty_fail(
                ~relation,
                hydro,
                ["ClassCode", "BaseZone", "GDE_Level"],
            ),
            "major",
            note="This check supports the nested modelling-stratum interpretation.",
        )


# ============================================================
# 7. LAI 阈值检查
# ============================================================

lai_safe_col = pick_col(lai_safe, ["LAI_safe_max", "LAI_safe_max_mean"])
lai_total_col = pick_col(
    lai_total,
    [
        "LAI_max_total",
        "LAImax_total",
        "LAImaxTotal_q25",
        "LAI_max_total_q25",
    ],
)

if (
    lai_safe is not None
    and lai_total is not None
    and "ClassCode" in lai_safe.columns
    and "ClassCode" in lai_total.columns
):
    safe_cols = ["ClassCode"]
    if "GDE_Level" in lai_safe.columns:
        safe_cols.append("GDE_Level")
    if lai_safe_col:
        safe_cols.append(lai_safe_col)

    total_cols = ["ClassCode"]
    if lai_total_col:
        total_cols.append(lai_total_col)

    lai = lai_safe[safe_cols].merge(
        lai_total[total_cols],
        on="ClassCode",
        how="outer",
    )

    add_check(
        "LAI_threshold",
        "LAI_safe_max present",
        "LAI_safe_max should not be missing",
        len(lai),
        nonempty_fail(
            lai[lai_safe_col].isna(),
            lai,
            ["ClassCode", "GDE_Level", lai_safe_col]
            if "GDE_Level" in lai.columns
            else ["ClassCode", lai_safe_col],
        )
        if lai_safe_col
        else pd.DataFrame({"error": ["LAI_safe_max column not found"]}),
        "critical",
    )

    add_check(
        "LAI_threshold",
        "LAI_max_total present",
        "LAI_max_total should not be missing",
        len(lai),
        nonempty_fail(
            lai[lai_total_col].isna(),
            lai,
            ["ClassCode", lai_total_col],
        )
        if lai_total_col
        else pd.DataFrame({"error": ["LAI_max_total column not found"]}),
        "critical",
    )

    if lai_safe_col and lai_total_col:
        cond = lai[lai_safe_col] <= lai[lai_total_col] + TOL

        add_check(
            "LAI_threshold",
            "LAI_safe_max <= LAI_max_total",
            "GDE-safe LAI upper bound should not exceed total water-resource LAI capacity",
            len(lai),
            nonempty_fail(
                ~cond,
                lai,
                ["ClassCode", lai_safe_col, lai_total_col],
            ),
            "critical",
        )

        ratio = lai[lai_safe_col] / lai[lai_total_col].replace(0, np.nan)
        bad_ratio = ratio.isna() | (ratio < -TOL) | (ratio > 1 + TOL)

        fail = lai.loc[
            bad_ratio,
            ["ClassCode", lai_safe_col, lai_total_col],
        ].copy()
        fail["safe_ratio_calc"] = ratio.loc[bad_ratio]

        add_check(
            "LAI_threshold",
            "Safe ratio range",
            "LAI_safe_max / LAI_max_total should be within 0–1",
            len(lai),
            fail,
            "major",
        )


# ============================================================
# 8. 候选库检查
# ============================================================

if cand is not None and "ClassCode" in cand.columns:
    total = len(cand)

    conservation = is_conservation_row(cand)

    if "template_group" in cand.columns:
        conservation = conservation | cand["template_group"].astype(str).str.contains(
            "conservation|保育|封育|自然恢复",
            case=False,
            na=False,
        )

    if "source_note" in cand.columns:
        conservation = conservation | cand["source_note"].astype(str).str.contains(
            "保育|封育|自然恢复|不进入人工组合",
            case=False,
            na=False,
        )

    if "scheme_id" in cand.columns:
        add_check(
            "candidate_library",
            "scheme_id unique",
            "Each candidate scheme_id should be unique",
            total,
            cand.loc[
                cand["scheme_id"].duplicated(keep=False),
                ["ClassCode", "scheme_id"],
            ].copy(),
            "major",
        )

    # 每个 HydroSupport ClassCode 至少有候选
    if hydro is not None and "ClassCode" in hydro.columns:
        hydro_codes = set(hydro["ClassCode"].dropna().astype(int))
        cand_codes = set(cand["ClassCode"].dropna().astype(int))
        missing_codes = sorted(hydro_codes - cand_codes)

        add_check(
            "candidate_library",
            "Each HydroSupport class has candidates",
            "Each modelling stratum should have at least one candidate or be explicitly backfilled later",
            len(hydro_codes),
            pd.DataFrame({"ClassCode": missing_codes}),
            "major",
        )

    for col in ["tree_ratio", "shrub_ratio", "grass_ratio"]:
        if col in cand.columns:
            bad = cand[col].isna() | (cand[col] < -TOL) | (cand[col] > 1 + TOL)

            add_check(
                "candidate_library",
                f"{col} range",
                f"{col} should be within 0–1",
                total,
                nonempty_fail(
                    bad,
                    cand,
                    ["ClassCode", "scheme_id", col],
                ),
                "critical",
            )

    if {"tree_ratio", "shrub_ratio", "grass_ratio"}.issubset(cand.columns):
        closed = ratio_closed(cand)

        # conservation 占位记录不按人工乔灌草比例闭合硬判
        bad = (~closed) & (~conservation)

        fail = cand.loc[
            bad,
            ["ClassCode", "scheme_id", "tree_ratio", "shrub_ratio", "grass_ratio"],
        ].copy()
        fail["ratio_sum"] = fail[
            ["tree_ratio", "shrub_ratio", "grass_ratio"]
        ].sum(axis=1)

        add_check(
            "candidate_library",
            "Tree-shrub-grass ratio closure",
            "Non-conservation candidates should satisfy tree+shrub+grass = 1",
            total,
            fail,
            "critical",
        )

    if {"target_LAI", "LAI_safe_max"}.issubset(cand.columns):
        # 人工候选配置才检查 target_LAI <= LAI_safe_max。
        # conservation / 保育 / 封育 / 自然恢复记录是生态安全占位记录，不作为人工配置候选检查。
        bad = (
            (cand["target_LAI"] > cand["LAI_safe_max"] + LAI_TOL_CANDIDATE)
            & (~conservation)
        )

        add_check(
            "candidate_library",
            "target_LAI <= LAI_safe_max for artificial candidates",
            "Artificial candidate target LAI must not exceed GDE-safe LAI upper bound; conservation placeholders are excluded",
            total,
            nonempty_fail(
                bad,
                cand,
                [
                    "ClassCode",
                    "scheme_id",
                    "GDE_Level",
                    "template_group",
                    "target_LAI",
                    "LAI_safe_max",
                    "LAI_margin",
                    "source_note",
                ],
            ),
            "critical",
        )

        # conservation 占位记录单独检查：不应有新增乔木配置
        viol = pd.Series(False, index=cand.index)

        if "tree_ratio" in cand.columns:
            viol = viol | (cand["tree_ratio"].fillna(0) > 1e-6)

        if "tree_density_per_ha" in cand.columns:
            viol = viol | (cand["tree_density_per_ha"].fillna(0) > 1e-6)

        bad_cons = conservation & viol

        add_check(
            "candidate_library",
            "Conservation placeholders have no new tree allocation",
            "Conservation or exclosure placeholder candidates should not allocate new tree planting",
            int(conservation.sum()),
            nonempty_fail(
                bad_cons,
                cand,
                [
                    "ClassCode",
                    "scheme_id",
                    "GDE_Level",
                    "template_group",
                    "tree_ratio",
                    "tree_density_per_ha",
                    "source_note",
                ],
            ),
            "major",
        )

    if {"LAI_margin", "LAI_safe_max", "target_LAI"}.issubset(cand.columns):
        calc = cand["LAI_safe_max"] - cand["target_LAI"]

        # conservation 占位记录可不检查 LAI_margin 精确一致性
        bad = ((cand["LAI_margin"] - calc).abs() > 1e-3) & (~conservation)

        fail = cand.loc[
            bad,
            ["ClassCode", "scheme_id", "LAI_margin", "LAI_safe_max", "target_LAI"],
        ].copy()
        fail["LAI_margin_calc"] = calc.loc[bad]

        add_check(
            "candidate_library",
            "LAI_margin consistency",
            "For artificial candidates, LAI_margin should equal LAI_safe_max - target_LAI",
            total,
            fail,
            "major",
        )

    for col in ["tree_density_per_ha", "shrub_density_per_ha"]:
        if col in cand.columns:
            bad = cand[col].notna() & (cand[col] < -TOL)

            add_check(
                "candidate_library",
                f"{col} non-negative",
                f"{col} should be non-negative where reported",
                total,
                nonempty_fail(
                    bad,
                    cand,
                    ["ClassCode", "scheme_id", col],
                ),
                "critical",
            )

    if "grass_cover_target_pct" in cand.columns:
        bad = cand["grass_cover_target_pct"].notna() & (
            (cand["grass_cover_target_pct"] < -TOL)
            | (cand["grass_cover_target_pct"] > 100 + TOL)
        )

        add_check(
            "candidate_library",
            "grass_cover_target_pct range",
            "grass cover target should be within 0–100%",
            total,
            nonempty_fail(
                bad,
                cand,
                ["ClassCode", "scheme_id", "grass_cover_target_pct"],
            ),
            "critical",
        )

    if {"GDE_Level", "tree_ratio"}.issubset(cand.columns):
        bad = (cand["GDE_Level"] == 1) & (cand["tree_ratio"] > TOL)

        add_check(
            "candidate_library",
            "GDE-dominated conservation tree exclusion",
            "GDE_Level=1 candidates should not include new tree allocation",
            total,
            nonempty_fail(
                bad,
                cand,
                [
                    "ClassCode",
                    "scheme_id",
                    "GDE_Level",
                    "tree_ratio",
                    "tree_density_per_ha",
                ],
            ),
            "critical",
        )

    if {"GDE_Level", "tree_ratio"}.issubset(cand.columns):
        highrisk = cand["GDE_Level"].eq(3)

        if "R_gde" in cand.columns:
            highrisk = highrisk | cand["R_gde"].ge(0.50)

        bad = highrisk & (cand["tree_ratio"] > 0.1001)

        add_check(
            "candidate_library",
            "High-risk GDE tree ratio cap",
            "GDE_Level=3 or R_gde>=0.50 candidates should have tree_ratio <= 0.10",
            total,
            nonempty_fail(
                bad,
                cand,
                ["ClassCode", "scheme_id", "GDE_Level", "R_gde", "tree_ratio"],
            ),
            "major",
        )


# ============================================================
# 9. BestCompromise 检查
# ============================================================

if best is not None and "ClassCode" in best.columns:
    total = len(best)

    add_check(
        "best_compromise",
        "One best compromise per optimized ClassCode",
        "BestCompromise should have unique ClassCode",
        total,
        best.loc[
            best["ClassCode"].duplicated(keep=False),
            ["ClassCode"],
        ].copy(),
        "critical",
    )

    if {"target_LAI", "LAI_safe_max"}.issubset(best.columns):
        bad = best["target_LAI"] > best["LAI_safe_max"] + TOL

        add_check(
            "best_compromise",
            "target_LAI <= LAI_safe_max",
            "Selected best compromise must not exceed GDE-safe LAI",
            total,
            nonempty_fail(
                bad,
                best,
                ["ClassCode", "GDE_Level", "target_LAI", "LAI_safe_max", "LAI_margin"],
            ),
            "critical",
        )

        # 新增：核心安全检查，强调最终选中解不突破安全 LAI
        add_check(
            "best_compromise",
            "Selected solution respects LAI safety bound",
            "All selected best-compromise solutions should satisfy target_LAI <= LAI_safe_max",
            total,
            nonempty_fail(
                bad,
                best,
                ["ClassCode", "GDE_Level", "target_LAI", "LAI_safe_max", "LAI_margin"],
            ),
            "critical",
        )

    if {"tree_ratio", "shrub_ratio", "grass_ratio"}.issubset(best.columns):
        conservation_best = is_conservation_row(best)
        closed = ratio_closed(best)
        bad = (~closed) & (~conservation_best)

        fail = best.loc[
            bad,
            ["ClassCode", "tree_ratio", "shrub_ratio", "grass_ratio"],
        ].copy()
        fail["ratio_sum"] = fail[
            ["tree_ratio", "shrub_ratio", "grass_ratio"]
        ].sum(axis=1)

        add_check(
            "best_compromise",
            "Tree-shrub-grass ratio closure",
            "Selected non-conservation best compromise should satisfy tree+shrub+grass = 1",
            total,
            fail,
            "critical",
        )

        # 新增：核心选中解比例闭合检查
        add_check(
            "best_compromise",
            "Selected solution ratio closure",
            "All selected best-compromise solutions should satisfy tree_ratio + shrub_ratio + grass_ratio = 1, except explicit conservation records",
            total,
            fail,
            "critical",
        )

    for col in [
        "score_sand",
        "score_stability",
        "score_water_security",
        "score_gde_safety",
        "best_compromise_score",
    ]:
        if col in best.columns:
            bad = best[col].isna() | (best[col] < -TOL) | (best[col] > 1 + TOL)

            add_check(
                "best_compromise",
                f"{col} range",
                f"{col} should be within 0–1",
                total,
                nonempty_fail(
                    bad,
                    best,
                    ["ClassCode", col],
                ),
                "major",
            )

    for col in [
        "obj1_sand_benefit",
        "obj2_eco_stability",
        "obj3_water_pressure",
        "obj4_gde_risk",
    ]:
        if col in best.columns:
            bad = best[col].isna()

            add_check(
                "best_compromise",
                f"{col} non-missing",
                f"{col} should not be missing",
                total,
                nonempty_fail(
                    bad,
                    best,
                    ["ClassCode", col],
                ),
                "major",
            )


# ============================================================
# 10. Pareto / Ranges 检查
# ============================================================

if (
    pareto is not None
    and ranges is not None
    and "ClassCode" in pareto.columns
    and "ClassCode" in ranges.columns
):
    pareto_counts = (
        pareto.groupby("ClassCode")
        .size()
        .reset_index(name="Pareto_n_calc")
    )

    if "Pareto_n" in ranges.columns:
        tmp = ranges[["ClassCode", "Pareto_n"]].merge(
            pareto_counts,
            on="ClassCode",
            how="outer",
        )

        tmp["Pareto_n"] = pd.to_numeric(tmp["Pareto_n"], errors="coerce")
        bad = tmp["Pareto_n"].fillna(-1).astype(float) != tmp[
            "Pareto_n_calc"
        ].fillna(-1).astype(float)

        add_check(
            "pareto",
            "Pareto_n consistency",
            "ParetoRanges Pareto_n should match ParetoFront row count by ClassCode",
            len(tmp),
            tmp.loc[bad].copy(),
            "major",
        )

    dominated = check_pareto_not_dominated(pareto)

    add_check(
        "pareto",
        "ParetoFront score-space non-dominated diagnostic",
        "ParetoFront is expected to represent a trade-off envelope; dominated records under normalized score-space are reported as diagnostic warnings",
        len(pareto),
        dominated,
        "minor",
        note=(
            "Checked using normalized score columns if available. "
            "This diagnostic does not invalidate BestCompromise if selected solutions satisfy LAI, density, ratio and Pareto-range constraints."
        ),
    )

if ranges is not None:
    range_pairs = [
        ("Tree_ratio_min", "Tree_ratio_max"),
        ("Shrub_ratio_min", "Shrub_ratio_max"),
        ("Grass_ratio_min", "Grass_ratio_max"),
        ("LAI_target_min", "LAI_target_max"),
        ("Tree_density_min", "Tree_density_max"),
        ("Shrub_density_min", "Shrub_density_max"),
        ("Grass_cover_min", "Grass_cover_max"),
    ]

    for lo, hi in range_pairs:
        if lo in ranges.columns and hi in ranges.columns:
            bad = ranges[lo] > ranges[hi] + TOL

            add_check(
                "pareto_ranges",
                f"{lo} <= {hi}",
                "Pareto range minimum should not exceed maximum",
                len(ranges),
                nonempty_fail(
                    bad,
                    ranges,
                    ["ClassCode", lo, hi],
                ),
                "major",
            )

if (
    best is not None
    and ranges is not None
    and "ClassCode" in best.columns
    and "ClassCode" in ranges.columns
):
    br = best.merge(
        ranges,
        on="ClassCode",
        how="inner",
        suffixes=("_best", "_range"),
    )

    containment = [
        ("tree_ratio", "Tree_ratio_min", "Tree_ratio_max"),
        ("shrub_ratio", "Shrub_ratio_min", "Shrub_ratio_max"),
        ("grass_ratio", "Grass_ratio_min", "Grass_ratio_max"),
        ("target_LAI", "LAI_target_min", "LAI_target_max"),
        ("tree_density_per_ha", "Tree_density_min", "Tree_density_max"),
        ("shrub_density_per_ha", "Shrub_density_min", "Shrub_density_max"),
        ("grass_cover_target_pct", "Grass_cover_min", "Grass_cover_max"),
    ]

    for val, lo, hi in containment:
        if val in br.columns and lo in br.columns and hi in br.columns:
            x = br[val]
            bad = x.notna() & (
                (x < br[lo] - 1e-4)
                | (x > br[hi] + 1e-4)
            )

            add_check(
                "best_vs_pareto_ranges",
                f"{val} within Pareto range",
                f"BestCompromise {val} should fall within ParetoRanges [{lo}, {hi}]",
                len(br),
                nonempty_fail(
                    bad,
                    br,
                    ["ClassCode", val, lo, hi],
                ),
                "major",
            )


# ============================================================
# 11. 最终模式检查
# ============================================================

mode_df = mode_impl if mode_impl is not None else mode_map

if mode_df is not None and "ClassCode" in mode_df.columns:
    total = len(mode_df)

    add_check(
        "final_mode",
        "ClassCode unique in final mode table",
        "Each ClassCode should have one final mode record",
        total,
        mode_df.loc[
            mode_df["ClassCode"].duplicated(keep=False),
            ["ClassCode"],
        ].copy(),
        "critical",
    )

    if "final_mode" in mode_df.columns:
        bad = mode_df["final_mode"].isna() | ~allowed_final_mode_mask(
            mode_df["final_mode"]
        )

        add_check(
            "final_mode",
            "final_mode valid",
            "final_mode should be one of the four predefined restoration modes",
            total,
            nonempty_fail(
                bad,
                mode_df,
                ["ClassCode", "GDE_Level", "final_mode"],
            ),
            "critical",
        )

    if "optimized_flag" in mode_df.columns:
        bad = ~mode_df["optimized_flag"].isin([0, 1])

        add_check(
            "final_mode",
            "optimized_flag valid",
            "optimized_flag should be 0 or 1",
            total,
            nonempty_fail(
                bad,
                mode_df,
                ["ClassCode", "optimized_flag"],
            ),
            "major",
        )

        if "backfill_reason" in mode_df.columns:
            bad = mode_df["optimized_flag"].eq(0) & (
                mode_df["backfill_reason"].isna()
                | mode_df["backfill_reason"].astype(str).str.strip().eq("")
            )

            add_check(
                "final_mode",
                "Backfilled units have reason",
                "optimized_flag=0 units should have backfill_reason",
                total,
                nonempty_fail(
                    bad,
                    mode_df,
                    [
                        "ClassCode",
                        "GDE_Level",
                        "optimized_flag",
                        "final_mode",
                        "backfill_reason",
                    ],
                ),
                "major",
            )

    if {"GDE_Level", "final_mode"}.issubset(mode_df.columns):
        g1 = mode_df["GDE_Level"].eq(1)

        if g1.any():
            is_herb = mode_df["final_mode"].astype(str).str.contains(
                "Herbaceous|草本|封育|exclosure",
                case=False,
                na=False,
            )

            bad = g1 & (~is_herb)

            add_check(
                "final_mode",
                "GDE-dominated conservation assigned to exclosure/herbaceous recovery",
                "GDE_Level=1 should be assigned to herbaceous–exclosure recovery or equivalent conservation mode",
                int(g1.sum()),
                nonempty_fail(
                    bad,
                    mode_df,
                    [
                        "ClassCode",
                        "GDE_Level",
                        "final_mode",
                        "optimized_flag",
                        "backfill_reason",
                    ],
                ),
                "major",
            )

    for col in ["tree_density_per_ha", "shrub_density_per_ha"]:
        if col in mode_df.columns:
            bad = mode_df[col].notna() & (mode_df[col] < -TOL)

            add_check(
                "final_mode",
                f"{col} non-negative",
                f"{col} should be non-negative where reported",
                total,
                nonempty_fail(
                    bad,
                    mode_df,
                    ["ClassCode", "final_mode", col],
                ),
                "critical",
            )

    if "grass_cover_target_pct" in mode_df.columns:
        bad = mode_df["grass_cover_target_pct"].notna() & (
            (mode_df["grass_cover_target_pct"] < -TOL)
            | (mode_df["grass_cover_target_pct"] > 100 + TOL)
        )

        add_check(
            "final_mode",
            "grass_cover_target_pct range",
            "grass cover target should be within 0–100%",
            total,
            nonempty_fail(
                bad,
                mode_df,
                ["ClassCode", "final_mode", "grass_cover_target_pct"],
            ),
            "critical",
        )


# ============================================================
# 12. 面积一致性检查
# ============================================================

if mode_df is not None and "Area_km2" in mode_df.columns:
    total_area_mode_df = float(mode_df["Area_km2"].sum())

    if mode_area is not None:
        area_col = pick_col(mode_area, ["Area_km2", "area_km2", "Area"])

        if area_col:
            total_area_summary = float(
                pd.to_numeric(mode_area[area_col], errors="coerce").sum()
            )

            fail = pd.DataFrame()

            if abs(total_area_mode_df - total_area_summary) > AREA_TOL:
                fail = pd.DataFrame({
                    "area_from_mode_table": [total_area_mode_df],
                    "area_from_mode_summary": [total_area_summary],
                    "difference": [total_area_mode_df - total_area_summary],
                })

            add_check(
                "area_consistency",
                "Mode area summary matches mode table",
                "Area total from Fig6_mode_area_summary should match ModeMapReady/ModeImplementation",
                1,
                fail,
                "major",
            )

    if mode_gde is not None:
        area_col = pick_col(mode_gde, ["Area_km2", "area_km2", "Area"])

        if area_col:
            total_area_gde = float(
                pd.to_numeric(mode_gde[area_col], errors="coerce").sum()
            )

            fail = pd.DataFrame()

            if abs(total_area_mode_df - total_area_gde) > AREA_TOL:
                fail = pd.DataFrame({
                    "area_from_mode_table": [total_area_mode_df],
                    "area_from_mode_by_GDE": [total_area_gde],
                    "difference": [total_area_mode_df - total_area_gde],
                })

            add_check(
                "area_consistency",
                "Mode-by-GDE area matches mode table",
                "Area total from Fig6_mode_by_GDE should match ModeMapReady/ModeImplementation",
                1,
                fail,
                "major",
            )


# ============================================================
# 13. MapCode 检查：提醒不能作为统计单元
# ============================================================

if mapcode_check is not None and {"MapCode", "final_mode"}.issubset(mapcode_check.columns):
    mode_n = (
        mapcode_check.groupby("MapCode")["final_mode"]
        .nunique()
        .reset_index(name="final_mode_n")
    )

    multi = mode_n.loc[mode_n["final_mode_n"] > 1].copy()

    add_check(
        "cartographic_code",
        "MapCode maps to single final_mode",
        "MapCode should not be used as statistical unit if one MapCode corresponds to multiple final modes",
        len(mode_n),
        multi,
        "minor",
        note=(
            "This is a cartographic-code warning. "
            "It supports the statement that MapCode is for mapping only and should not be used as a statistical unit."
        ),
    )


# ============================================================
# 14. 输出 QA 结果
# ============================================================

qa = pd.DataFrame(qa_rows)

severity_order = {"critical": 0, "major": 1, "minor": 2}
status_order = {"FAIL": 0, "SKIPPED": 1, "PASS": 2}

qa["severity_order"] = qa["severity"].map(severity_order).fillna(9)
qa["status_order"] = qa["status"].map(status_order).fillna(9)

qa = (
    qa.sort_values(["status_order", "severity_order", "module", "check_item"])
    .drop(columns=["severity_order", "status_order"])
)

qa.to_csv(OUT_QA, index=False, encoding="utf-8-sig")

paper = (
    qa.groupby(["module", "severity", "status"], as_index=False)
    .agg(
        check_n=("check_item", "count"),
        mean_pass_rate=("pass_rate", "mean"),
        total_fail_records=("n_fail", "sum"),
    )
    .sort_values(["module", "severity", "status"])
)

paper.to_csv(OUT_QA_PAPER, index=False, encoding="utf-8-sig")


# ============================================================
# 15. Markdown 报告
# 不使用 to_markdown，避免 tabulate 依赖报错
# ============================================================

lines = []
lines.append("# Supplementary QA/QC report")
lines.append("")
lines.append(f"Data directory: `{DATA_DIR}`")
lines.append(f"Total checks: {len(qa)}")
lines.append("")

lines.append("## Check status summary")
lines.append("")
lines.append(qa["status"].value_counts(dropna=False).to_string())
lines.append("")

lines.append("## Failed checks")
lines.append("")

failed = qa.loc[qa["status"] == "FAIL"].copy()

if failed.empty:
    lines.append("No failed checks.")
else:
    show_cols = [
        "module",
        "check_item",
        "severity",
        "n_total",
        "n_fail",
        "pass_rate",
        "fail_record_file",
        "note",
    ]
    lines.append(failed[show_cols].to_string(index=False))

lines.append("")
lines.append("## Skipped checks")
lines.append("")

skipped = qa.loc[qa["status"] == "SKIPPED"].copy()

if skipped.empty:
    lines.append("No skipped checks.")
else:
    show_cols = ["module", "check_item", "severity", "note"]
    lines.append(skipped[show_cols].to_string(index=False))

lines.append("")
lines.append("## All checks")
lines.append("")

show_cols_all = [
    "module",
    "check_item",
    "severity",
    "status",
    "n_total",
    "n_fail",
    "pass_rate",
    "note",
]

lines.append(qa[show_cols_all].to_string(index=False))

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 16. 控制台输出
# ============================================================

print("QA/QC 完成。")
print(f"输出: {OUT_QA}")
print(f"论文汇总表: {OUT_QA_PAPER}")
print(f"失败记录目录: {FAIL_DIR}")
print(f"报告: {OUT_REPORT}")

print("\n状态统计：")
print(qa["status"].value_counts(dropna=False))

if (qa["status"] == "FAIL").any():
    print("\n存在未通过检查，请查看 FAIL 项。")
    print(
        qa.loc[
            qa["status"] == "FAIL",
            ["module", "check_item", "severity", "n_fail", "fail_record_file"],
        ].to_string(index=False)
    )
else:
    print("\n所有已执行检查均通过。")
