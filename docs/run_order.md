
# Run Order

1. `gee/01_build_ecohydro_gde_zones.js`
2. `gee/02_export_static_annual_stacks.js`
3. `src/01_extract_class_year_table.py`
4. `src/02_model_hydroclimatic_lai_capacity.py`
5. `src/03_compute_gde_safe_lai.py`
6. `src/04_classify_vegetation_zone.py`
7. `src/05_fix_hydro_support_table.py`
8. `src/06_fix_eco_function_table.py`
9. `src/07_merge_step5_support.py`
10. `src/08_build_candidate_library.py`
11. `src/09_multiobjective_optimization.py`
12. `src/10_translate_best_solution_to_maps.py`
13. `src/11_qa_qc_checks.py`
14. `src/12_gde_threshold_sensitivity.py`
15. `src/13_weight_sensitivity.py`
16. `src/14_rf_shap_attribution.py`
17. `src/15_gam_vpd_marginal_response.py`

The exact run order can be shortened if precomputed class-level tables are provided in the DOI-archived data package.
