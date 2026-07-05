"""
01_extract_class_year_table.py

Aggregates static and annual GeoTIFF stacks to a ClassCode x Year table and year-summary table.

This script was organized from the project process notes for the manuscript:
"Groundwater-dependent ecosystems decouple climatic suitability from safe woody
restoration capacity in dryland shelterbelts".

Before running, place the required processed CSV/TIF inputs in
../data/processed_class_tables or edit DATA_DIR/ROOT below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# =============================================================================
# 0. 参数区
# =============================================================================

# 改成你的真实目录
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_class_tables"

STATIC_TIF = DATA_DIR / "ThreeNorth_StaticStack_Formal_V1.tif"
ANNUAL_TEMPLATE = "ThreeNorth_AnnualStack_{year}.tif"

START_YEAR = 2005
END_YEAR = 2024

OUT_CSV = DATA_DIR / "ThreeNorth_ClassYear_Table_Full_2005_2024.csv"
OUT_YEAR_SUMMARY_CSV = DATA_DIR / "ThreeNorth_ClassYear_YearSummary_2005_2024.csv"

BLOCK_SIZE = 1024


# =============================================================================
# 1. 波段定义：必须和 GEE 导出顺序一致
# =============================================================================

STATIC_BANDS = [
    "ClassCode",
    "BaseZone",
    "GDE_Level",
    "GDE_frac",
    "GDE_stability",
    "GDE_persistence",
    "GDE_trajectory_code",
    "NaturalFrac",
    "NaturalMask_1km",
    "LAI_current_3yr",
    "SW_occurrence",
    "SW_access",
    "Elevation",
    "Slope",
    "GWSA_trend",
]

ANNUAL_BANDS = [
    "P",
    "PET",
    "AET",
    "Runoff",
    "Soil",
    "Tmean",
    "AI_prelim_noGW",
    "AI_with_GWSA_proxy",
    "GWSA_mean_period",
    "LAI_mean",
    "LAI_max",
    "LAI_nat_mean",
    "LAI_nat_max",
]


# =============================================================================
# 2. 工具函数
# =============================================================================

def band_index_map(names: List[str]) -> Dict[str, int]:
    return {name: i + 1 for i, name in enumerate(names)}


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            win_width = min(block_size, width - col_off)
            win_height = min(block_size, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)


def safe_mean(sum_val: float, count_val: int) -> float:
    if count_val <= 0:
        return np.nan
    return sum_val / count_val


def percentile_from_concat(arrays: List[np.ndarray], q: float) -> float:
    if not arrays:
        return np.nan
    arr = np.concatenate(arrays)
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def per_row_pixel_area_km2(transform, row_start: int, nrows: int) -> np.ndarray:
    """
    对经纬度栅格，按纬度近似计算每一行像元面积（km²）
    """
    dlon_deg = abs(transform.a)
    dlat_deg = abs(transform.e)

    rows = np.arange(row_start, row_start + nrows)
    _, lats = rasterio.transform.xy(transform, rows, np.zeros_like(rows), offset="center")
    lats = np.asarray(lats, dtype=np.float64)

    pixel_height_km = 110.574 * dlat_deg
    pixel_width_km = 111.320 * dlon_deg * np.cos(np.deg2rad(lats))

    return pixel_width_km * pixel_height_km


# =============================================================================
# 3. 静态表：按 ClassCode 汇总静态字段
# =============================================================================

def build_static_table(static_tif: Path) -> pd.DataFrame:
    idx = band_index_map(STATIC_BANDS)

    with rasterio.open(static_tif) as src:
        if src.count != len(STATIC_BANDS):
            raise ValueError(
                f"静态栅格 band 数不对：期望 {len(STATIC_BANDS)}，实际 {src.count}"
            )

        transform = src.transform
        acc: Dict[int, Dict[str, object]] = {}

        for window in iter_windows(src.width, src.height, BLOCK_SIZE):
            arr = src.read(window=window).astype(np.float32)

            row_area_km2 = per_row_pixel_area_km2(transform, window.row_off, window.height)
            area2d = np.repeat(row_area_km2[:, None], window.width, axis=1).astype(np.float32)

            class_code = arr[idx["ClassCode"] - 1]
            valid = np.isfinite(class_code) & (class_code > 0)

            if not np.any(valid):
                continue

            class_code_int = class_code[valid].astype(np.int32)
            pixel_area_vals = area2d[valid]

            values = {
                "BaseZone": arr[idx["BaseZone"] - 1][valid],
                "GDE_Level": arr[idx["GDE_Level"] - 1][valid],
                "GDE_frac": arr[idx["GDE_frac"] - 1][valid],
                "GDE_stability": arr[idx["GDE_stability"] - 1][valid],
                "GDE_persistence": arr[idx["GDE_persistence"] - 1][valid],
                "GDE_trajectory_code": arr[idx["GDE_trajectory_code"] - 1][valid],
                "NaturalFrac": arr[idx["NaturalFrac"] - 1][valid],
                "NaturalMask_1km": arr[idx["NaturalMask_1km"] - 1][valid],
                "LAI_current_3yr": arr[idx["LAI_current_3yr"] - 1][valid],
                "SW_occurrence": arr[idx["SW_occurrence"] - 1][valid],
                "SW_access": arr[idx["SW_access"] - 1][valid],
                "Elevation": arr[idx["Elevation"] - 1][valid],
                "Slope": arr[idx["Slope"] - 1][valid],
                "GWSA_trend": arr[idx["GWSA_trend"] - 1][valid],
            }

            unique_classes = np.unique(class_code_int).astype(np.int32).tolist()

            for c in unique_classes:
                c = int(c)
                m = class_code_int == c

                if c not in acc:
                    acc[c] = {
                        "count": 0,
                        "Area_km2_sum": 0.0,
                        "BaseZone_sum": 0.0,
                        "GDE_Level_sum": 0.0,
                        "GDE_frac_sum": 0.0,
                        "GDE_stability_sum": 0.0,
                        "GDE_persistence_sum": 0.0,
                        "GDE_trajectory_code_sum": 0.0,
                        "NaturalFrac_sum": 0.0,
                        "LAI_current_3yr_sum": 0.0,
                        "SW_occurrence_sum": 0.0,
                        "SW_access_sum": 0.0,
                        "Elevation_sum": 0.0,
                        "Slope_sum": 0.0,
                        "GWSA_trend_sum": 0.0,
                    }

                acc[c]["count"] += int(np.sum(m))
                acc[c]["Area_km2_sum"] += float(np.nansum(pixel_area_vals[m]))
                acc[c]["BaseZone_sum"] += float(np.nansum(values["BaseZone"][m]))
                acc[c]["GDE_Level_sum"] += float(np.nansum(values["GDE_Level"][m]))
                acc[c]["GDE_frac_sum"] += float(np.nansum(values["GDE_frac"][m]))
                acc[c]["GDE_stability_sum"] += float(np.nansum(values["GDE_stability"][m]))
                acc[c]["GDE_persistence_sum"] += float(np.nansum(values["GDE_persistence"][m]))
                acc[c]["GDE_trajectory_code_sum"] += float(np.nansum(values["GDE_trajectory_code"][m]))
                acc[c]["NaturalFrac_sum"] += float(np.nansum(values["NaturalFrac"][m]))
                acc[c]["LAI_current_3yr_sum"] += float(np.nansum(values["LAI_current_3yr"][m]))
                acc[c]["SW_occurrence_sum"] += float(np.nansum(values["SW_occurrence"][m]))
                acc[c]["SW_access_sum"] += float(np.nansum(values["SW_access"][m]))
                acc[c]["Elevation_sum"] += float(np.nansum(values["Elevation"][m]))
                acc[c]["Slope_sum"] += float(np.nansum(values["Slope"][m]))
                acc[c]["GWSA_trend_sum"] += float(np.nansum(values["GWSA_trend"][m]))

    rows = []
    for c, a in acc.items():
        cnt = int(a["count"])
        rows.append({
            "ClassCode": int(c),
            "BaseZone": int(round(safe_mean(a["BaseZone_sum"], cnt))),
            "GDE_Level": int(round(safe_mean(a["GDE_Level_sum"], cnt))),
            "Area_km2": float(a["Area_km2_sum"]),
            "GDE_frac_mean": safe_mean(a["GDE_frac_sum"], cnt),
            "GDE_stability_mean": safe_mean(a["GDE_stability_sum"], cnt),
            "GDE_persistence_mean": safe_mean(a["GDE_persistence_sum"], cnt),
            "GDE_trajectory_code_mean": safe_mean(a["GDE_trajectory_code_sum"], cnt),
            "NaturalFrac_mean": safe_mean(a["NaturalFrac_sum"], cnt),
            "LAI_current_3yr_mean": safe_mean(a["LAI_current_3yr_sum"], cnt),
            "SW_occurrence_mean": safe_mean(a["SW_occurrence_sum"], cnt),
            "SW_access_mean": safe_mean(a["SW_access_sum"], cnt),
            "Elevation_mean": safe_mean(a["Elevation_sum"], cnt),
            "Slope_mean": safe_mean(a["Slope_sum"], cnt),
            "GWSA_trend_mean": safe_mean(a["GWSA_trend_sum"], cnt),
        })

    df = pd.DataFrame(rows).sort_values("ClassCode").reset_index(drop=True)
    return df


# =============================================================================
# 4. 年度表：按 ClassCode 汇总年度变量
# =============================================================================

def build_annual_table(static_tif: Path, annual_tif: Path, year: int) -> pd.DataFrame:
    sidx = band_index_map(STATIC_BANDS)
    aidx = band_index_map(ANNUAL_BANDS)

    with rasterio.open(static_tif) as ssrc, rasterio.open(annual_tif) as asrc:
        if asrc.count != len(ANNUAL_BANDS):
            raise ValueError(
                f"{annual_tif.name} band 数不对：期望 {len(ANNUAL_BANDS)}，实际 {asrc.count}"
            )

        if ssrc.width != asrc.width or ssrc.height != asrc.height:
            raise ValueError(f"{annual_tif.name} 与静态栅格尺寸不一致")
        if ssrc.transform != asrc.transform:
            raise ValueError(f"{annual_tif.name} 与静态栅格坐标变换不一致")

        acc: Dict[int, Dict[str, object]] = {}
        nat_arrays: Dict[int, List[np.ndarray]] = {}

        for window in iter_windows(ssrc.width, ssrc.height, BLOCK_SIZE):
            sarr = ssrc.read(window=window).astype(np.float32)
            aarr = asrc.read(window=window).astype(np.float32)

            class_code = sarr[sidx["ClassCode"] - 1]
            natural_mask = sarr[sidx["NaturalMask_1km"] - 1]

            valid = np.isfinite(class_code) & (class_code > 0)
            if not np.any(valid):
                continue

            class_code_int = class_code[valid].astype(np.int32)

            values = {
                "P": aarr[aidx["P"] - 1][valid],
                "PET": aarr[aidx["PET"] - 1][valid],
                "AET": aarr[aidx["AET"] - 1][valid],
                "Runoff": aarr[aidx["Runoff"] - 1][valid],
                "Soil": aarr[aidx["Soil"] - 1][valid],
                "Tmean": aarr[aidx["Tmean"] - 1][valid],
                "AI_prelim_noGW": aarr[aidx["AI_prelim_noGW"] - 1][valid],
                "AI_with_GWSA_proxy": aarr[aidx["AI_with_GWSA_proxy"] - 1][valid],
                "GWSA_mean_period": aarr[aidx["GWSA_mean_period"] - 1][valid],
                "LAI_mean": aarr[aidx["LAI_mean"] - 1][valid],
                "LAI_max": aarr[aidx["LAI_max"] - 1][valid],
                "LAI_nat_mean": aarr[aidx["LAI_nat_mean"] - 1][valid],
            }

            unique_classes = np.unique(class_code_int).astype(np.int32).tolist()

            for c in unique_classes:
                c = int(c)
                m = class_code_int == c

                if c not in acc:
                    acc[c] = {
                        "count": 0,
                        "P_sum": 0.0,
                        "PET_sum": 0.0,
                        "AET_sum": 0.0,
                        "Runoff_sum": 0.0,
                        "Soil_sum": 0.0,
                        "Tmean_sum": 0.0,
                        "AI_prelim_noGW_sum": 0.0,
                        "AI_with_GWSA_proxy_sum": 0.0,
                        "GWSA_mean_period_sum": 0.0,
                        "LAI_mean_sum": 0.0,
                        "LAI_max_sum": 0.0,
                        "LAI_nat_mean_sum": 0.0,
                    }

                acc[c]["count"] += int(np.sum(m))
                acc[c]["P_sum"] += float(np.nansum(values["P"][m]))
                acc[c]["PET_sum"] += float(np.nansum(values["PET"][m]))
                acc[c]["AET_sum"] += float(np.nansum(values["AET"][m]))
                acc[c]["Runoff_sum"] += float(np.nansum(values["Runoff"][m]))
                acc[c]["Soil_sum"] += float(np.nansum(values["Soil"][m]))
                acc[c]["Tmean_sum"] += float(np.nansum(values["Tmean"][m]))
                acc[c]["AI_prelim_noGW_sum"] += float(np.nansum(values["AI_prelim_noGW"][m]))
                acc[c]["AI_with_GWSA_proxy_sum"] += float(np.nansum(values["AI_with_GWSA_proxy"][m]))
                acc[c]["GWSA_mean_period_sum"] += float(np.nansum(values["GWSA_mean_period"][m]))
                acc[c]["LAI_mean_sum"] += float(np.nansum(values["LAI_mean"][m]))
                acc[c]["LAI_max_sum"] += float(np.nansum(values["LAI_max"][m]))
                acc[c]["LAI_nat_mean_sum"] += float(np.nansum(values["LAI_nat_mean"][m]))

            # 自然植被 LAI_nat_max 的 p90 / p95 / count
            lai_nat_max_full = aarr[aidx["LAI_nat_max"] - 1]
            valid_nat = valid & (natural_mask >= 0.5) & np.isfinite(lai_nat_max_full)

            if np.any(valid_nat):
                nat_classes = class_code[valid_nat].astype(np.int32)
                nat_vals = lai_nat_max_full[valid_nat].astype(np.float32)

                unique_nat_classes = np.unique(nat_classes).astype(np.int32).tolist()
                for c in unique_nat_classes:
                    c = int(c)
                    m = nat_classes == c
                    if c not in nat_arrays:
                        nat_arrays[c] = []
                    nat_arrays[c].append(nat_vals[m])

    rows = []
    for c, a in acc.items():
        cnt = int(a["count"])
        rows.append({
            "Year": year,
            "ClassCode": int(c),
            "P_mean": safe_mean(a["P_sum"], cnt),
            "PET_mean": safe_mean(a["PET_sum"], cnt),
            "AET_mean": safe_mean(a["AET_sum"], cnt),
            "Runoff_mean": safe_mean(a["Runoff_sum"], cnt),
            "Soil_mean": safe_mean(a["Soil_sum"], cnt),
            "Tmean_mean": safe_mean(a["Tmean_sum"], cnt),
            "AI_prelim_noGW_mean": safe_mean(a["AI_prelim_noGW_sum"], cnt),
            "AI_with_GWSA_proxy_mean": safe_mean(a["AI_with_GWSA_proxy_sum"], cnt),
            "GWSA_mean_period": safe_mean(a["GWSA_mean_period_sum"], cnt),
            "LAI_mean": safe_mean(a["LAI_mean_sum"], cnt),
            "LAI_max_mean": safe_mean(a["LAI_max_sum"], cnt),
            "LAI_nat_mean": safe_mean(a["LAI_nat_mean_sum"], cnt),
            "LAI_nat_max_p90": percentile_from_concat(nat_arrays.get(c, []), 90),
            "LAI_nat_max_p95": percentile_from_concat(nat_arrays.get(c, []), 95),
            "Nat_pixel_count": int(sum(len(x) for x in nat_arrays.get(c, []))),
        })

    df = pd.DataFrame(rows).sort_values("ClassCode").reset_index(drop=True)
    return df


# =============================================================================
# 5. 主程序
# =============================================================================

def main():
    print("Step 1: 读取静态栅格并汇总...")
    static_df = build_static_table(STATIC_TIF)
    print(f"静态表完成，ClassCode 数量: {len(static_df)}")
    print("静态面积前5行：")
    print(static_df[["ClassCode", "Area_km2"]].head())

    all_years = []
    year_logs = []

    for year in range(START_YEAR, END_YEAR + 1):
        annual_tif = DATA_DIR / ANNUAL_TEMPLATE.format(year=year)

        if not annual_tif.exists():
            print(f"[跳过] 缺少年度栅格: {annual_tif.name}")
            year_logs.append({
                "Year": year,
                "Status": "missing_file",
                "Rows": 0,
            })
            continue

        print(f"Step 2: 处理年度栅格 {year} ...")
        annual_df = build_annual_table(STATIC_TIF, annual_tif, year)

        merged = static_df.merge(annual_df, on="ClassCode", how="left")

        front_cols = ["Year", "ClassCode", "BaseZone", "GDE_Level"]
        other_cols = [c for c in merged.columns if c not in front_cols]
        merged = merged[front_cols + other_cols]

        all_years.append(merged)

        year_logs.append({
            "Year": year,
            "Status": "ok",
            "Rows": len(merged),
        })

        print(f"  年份 {year} 完成，输出行数: {len(merged)}")

    if not all_years:
        raise RuntimeError("没有任何年度栅格成功处理，请检查文件路径。")

    result = pd.concat(all_years, ignore_index=True)
    result = result.sort_values(["Year", "ClassCode"]).reset_index(drop=True)

    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(year_logs).to_csv(OUT_YEAR_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n全部完成。")
    print(f"主表输出: {OUT_CSV}")
    print(f"年份汇总输出: {OUT_YEAR_SUMMARY_CSV}")
    print(f"总行数: {len(result)}")
    print(f"总列数: {result.shape[1]}")
    print("前5行预览：")
    print(result.head())

    print("\nArea_km2 描述统计：")
    print(result["Area_km2"].describe())

    print("\nNat_pixel_count 描述统计：")
    print(result["Nat_pixel_count"].describe())


if __name__ == "__main__":
    main()
