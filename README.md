
# GDE-aware safe woody restoration capacity in Three-North shelterbelts

This repository contains the analysis code organized for the manuscript **"Groundwater-dependent ecosystems decouple climatic suitability from safe woody restoration capacity in dryland shelterbelts"**.

The workflow estimates hydroclimatic LAI carrying capacity, applies GDE and groundwater-safety reductions to derive GDE-constrained safe LAI, optimizes tree-shrub-grass restoration configurations, and evaluates hydroclimatic/VPD and GDE controls using GAM-style marginal responses, random forests, SHAP, and sensitivity diagnostics.

## Repository structure

```text
gee/      Google Earth Engine scripts for spatial layers and annual stacks
src/      Python scripts for class-level aggregation, modelling, optimization, QA/QC, attribution, and figures
data/     Placeholder folders and metadata for processed inputs deposited separately
outputs/  Placeholder folder for generated tables and figures
docs/     Data availability text, run order, and upload notes
```

## Main run order

1. Run `gee/01_build_ecohydro_gde_zones.js` in Google Earth Engine to create the ecohydrological-GDE ClassCode layer.
2. Run `gee/02_export_static_annual_stacks.js` in Google Earth Engine to export static and annual GeoTIFF stacks.
3. Place exported GeoTIFFs and processed CSV tables in `data/processed_class_tables/`.
4. Run the Python scripts in numeric order from `src/01_...py` to `src/15_...py`.

The scripts were organized from the original project notes. Some input datasets are derived products from GEE assets and should be deposited in Zenodo/Dryad/Figshare with a DOI rather than stored directly in GitHub.

## Python environment

```bash
pip install -r requirements.txt
```

The code was organized for Python 3.10+.

## Data availability

Large raster layers and processed class-level tables are not committed to this repository. Deposit them in a public data repository that issues DOIs, then update the DOI placeholders in `docs/data_availability_statement.md` and `CITATION.cff`.

## Citation

Please cite the manuscript and the archived software/data DOI once available.
