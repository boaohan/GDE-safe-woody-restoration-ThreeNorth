
# GitHub and DOI Upload Checklist

## GitHub

1. Create a new GitHub repository, for example `GDE-safe-woody-restoration-ThreeNorth`.
2. Upload the full contents of this folder.
3. Edit `CITATION.cff` and replace `USERNAME` and DOI placeholders.
4. Create a release named `v1.0.0`.

## DOI archive

Recommended route:

1. Upload code release to Zenodo as `Software` or connect GitHub release to Zenodo.
2. Upload processed data package to Zenodo/Dryad/Figshare as `Dataset`.
3. Use separate DOI records for code and data if the data are large.
4. Replace the DOI placeholders in `docs/data_availability_statement.md`.

## Before submission

- Confirm all private or sensitive field information has been aggregated or anonymized.
- Confirm any third-party data redistribution rules are respected.
- Confirm GEE asset-derived layers are available through the DOI archive, not only private GEE paths.
