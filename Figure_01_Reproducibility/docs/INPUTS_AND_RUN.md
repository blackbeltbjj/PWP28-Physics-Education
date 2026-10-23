# Inputs and execution

## Original project structure required

The source locates `src/config/config.py` by walking upward from its own directory and imports `config.config` and optionally `plotting_utils`. The full original project is required, including:

```text
pwp_mascara/
  src/
    config/config.py
    plotting_utils.py                 # optional; script has fallback
    JPO/JPO011B_six_date_final_pwp_comparison_grayscale.py
  data/raw/**/*.nc                   # daily absolute NOAA OISST, including all six dates
  data/processed/pacific_mask_oisst.npy
  data/processed/grid_lat.npy
  data/processed/grid_lon.npy
```

`RAW_DIR` and `PROCESSED_DIR` are resolved by the project's existing `config/config.py`. Verify that they refer to the correct audited datasets. No source NetCDF, grid, mask, or project configuration files are included in this package.

## Suggested installation and run (Windows PowerShell)

Install dependencies in the appropriate existing Python environment:

```powershell
python -m pip install -r requirements.txt
```

Copy the supplied source **as a new file** into the original project's `src/JPO/` directory only if it is not already present. Avoid overwriting the scientific source. Then run from the original project:

```powershell
python .\src\JPO\JPO011B_six_date_final_pwp_comparison_grayscale.py
```

The script writes under `outputs/JPO/RBEF/` in `figures/`, `tables/`, and `reports/`. It deliberately refuses to overwrite existing output products; if it reports existing outputs, preserve them and choose a separate verified run location rather than deleting archival products.

## Verification checklist

1. Check input dataset versions, exact six dates, Pacific mask and reference grid.
2. Run in the original project environment without altering archived scientific code.
3. Compare the regenerated figure visually with `figures/RBEF_Figure_01_PWP28_spatial_snapshots.png`.
4. Compare all six centroid coordinates and PWP areas against the displayed labels and generated CSV.
5. Record NetCDF source hashes from the script report, output hashes, Python/package versions, and any reason for differences.

This package's SHA-256 manifest verifies **preservation of packaged bytes**, not successful numerical reproduction.
