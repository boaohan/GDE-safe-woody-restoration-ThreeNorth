
# Data Folder

This folder is a lightweight placeholder for GitHub. Large spatial layers and full processed tables should be deposited in a DOI-issuing repository such as Zenodo, Dryad, or Figshare.

## Public products used through Google Earth Engine

- `IDAHO_EPSCOR/TERRACLIMATE`: Precipitation, PET, AET, runoff, soil moisture, temperature
- `MODIS/061/MCD15A3H`: LAI
- `MODIS/061/MOD13Q1`: NDVI
- `MODIS/061/MOD17A3HGF`: NPP
- `MODIS/061/MCD12Q1`: Land cover
- `ECMWF/ERA5_LAND/MONTHLY_AGGR`: VPD/wind-related climate variables
- `USGS/SRTMGL1_003`: Elevation and slope
- `JRC/GSW1_4/GlobalSurfaceWater`: Permanent surface-water mask

## Derived data to archive with DOI

Archive the following processed/derived datasets with the final paper:

- `ThreeNorth_EcoHydro_Zones_1km.tif`
- `ThreeNorth_StaticStack_Formal_V2.tif`
- `ThreeNorth_AnnualStack_2005.tif` to `ThreeNorth_AnnualStack_2024.tif`
- GDE binary period layers P1-P4, GDE union, GDE trajectory, GDE fraction/stability/persistence layers
- GWSA period means and 2005-2024 trend layers
- `ThreeNorth_ClassYear_FullMetrics_2005_2024.csv`
- `ThreeNorth_Class_LAImaxTotal_q25.csv`
- `ThreeNorth_Class_LAI_safe_max.csv`
- `ThreeNorth_Class_HydroSupport_2005_2024.csv`
- `ThreeNorth_Class_EcoFunction.csv`
- `ThreeNorth_Class_ObservedStructure.csv`
- `ThreeNorth_Class_Step5Support_RealMetrics.csv`
- `ThreeNorth_Class_CandidateLibrary.csv`
- `ThreeNorth_Class_MOO_BestCompromise.csv`
- `ThreeNorth_Class_MOO_ParetoFront.csv`
- GAM input/output tables and RF/SHAP attribution tables
- Aggregated field-support tables for the Horqin-Otindag Sandy Lands

Do not rely on private GEE asset IDs alone. Reviewers need either public access to the derived layers or archived copies in the data repository.
