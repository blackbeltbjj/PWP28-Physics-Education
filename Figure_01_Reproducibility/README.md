# RBEF Figure 01 — PWP28 spatial snapshots

This package preserves the supplied Python source and the supplied final grayscale Figure 1 image **without modifying either file**.

## Figure configuration

- Six panels in a 3 × 2 layout.
- Panel A: 1995-09-01; B: 1983-04-30; C: 1989-09-01; D: 1998-04-01; E: 1999-09-01; F: 2016-04-01.
- Canonical Pacific Warm Pool: valid cells in the fixed Pacific mask with SST >= 28.0 °C. No largest-connected-component (LCC) filtering.
- Spherical cell areas and area-weighted three-dimensional spherical centroids.
- Shared grayscale SST scale and PWP outlines.

## Contents

- `src/JPO011B_six_date_final_pwp_comparison_grayscale.py` — original supplied script.
- `figures/RBEF_Figure_01_PWP28_spatial_snapshots.png` — original supplied manuscript image.
- `requirements.txt` — direct scientific Python dependencies plus NetCDF backend.
- `docs/INPUTS_AND_RUN.md` — project dependencies, execution, and verification.
- `SHA256_MANIFEST.csv` — file integrity checksums.

## Provenance and verification status

The six dates, panel arrangement, grayscale styling, threshold, centroid annotation, and geographic layout match the supplied manuscript image. **An independent reproduction from the original NOAA OISST NetCDF inputs has not been executed as part of this package assembly.** Consequently the exact historical execution that produced the renamed manuscript PNG is not independently proven. The preserved source and figure are not altered.

The Python script expects the complete original project structure and is **not standalone**. See `docs/INPUTS_AND_RUN.md`. Its native output names begin with `JPO_011B_six_date_canonical_pwp_grayscale_`; the supplied manuscript image has a different name. Do not treat the differing filenames as evidence of byte-for-byte reproducibility.
