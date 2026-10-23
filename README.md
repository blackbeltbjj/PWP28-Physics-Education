# PWP28-Physics-Education

Reproducibility package in preparation for the manuscript:
**Time-Series Analysis with Python and Authentic Climate Data: From Spherical Geometry to Variability in the Pacific Warm Pool Area**.

## Included validated derived dataset

`data/derived/RBEF_PWP28_daily_area_19810901_20260930.csv`

- Daily Pacific Warm Pool canonical area at SST >= 28.0 deg C.
- 1981-09-01 to 2026-09-30 (16,466 dates).
- Area units: million square kilometres.
- Extracted from `JPO04_daily_pwp_geometry.csv`, filtering threshold 28.0 deg C and selecting the canonical area (not LCC area).
- Dataset SHA-256 and upstream SHA-256 are documented in `provenance/SHA256SUMS.txt` and `data/metadata/PWP28_DATA_METADATA.json`.

## Citation and archive status

**Zenodo deposit and DOI have not yet been verified.** Do not invent a DOI or imply the dataset is already archived. When a verified DOI is available, update this README, the dataset metadata and `CITATION.cff` before public release. The derived dataset must be cited separately when used independently.

## Scope

This initial staged package contains only the validated daily area series and provenance records. It **does not yet reproduce the nine manuscript figures**. The spatial figure needs additional spatial inputs; source scripts, tests, requirements, and figure outputs must be audited and added before claiming full reproducibility.

## Data origin

Underlying SST: NOAA OISST v2.1 (daily, 0.25-degree grid). The source SST NetCDF files are not included in this package.
