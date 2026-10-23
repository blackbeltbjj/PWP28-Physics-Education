# PWP28-Physics-Education — GitHub code distribution (draft)

This is a **distribution adaptation**, not a new scientific analysis. Original RBEF04–RBEF10 scientific scripts remain untouched in the scientific project. The adapted RBEF04–07 and RBEF09 read the audited two-column PWP28 daily area CSV; RBEF08 reads RBEF07 outputs. Output folders remain `outputs/RBEF/{figures,tables,reports}`.

## Required data

Copy `RBEF_PWP28_daily_area_19810901_20260930.csv` to `data/derived/`. SHA256 (uppercase): `4DA00D092E9B94EF182838499582C21DC34475D9F0EAA9E34A3E6C3C137C29FB`. Period: 1981-09-01 through 2026-09-30, 16,466 daily observations. The CSV is not included in this code-only archive because it was not uploaded in this conversation. It has already been staged on the user's computer.

## Run (Windows PowerShell, from repository root)

```powershell
python -m pip install -r requirements.txt
(Get-FileHash .\data\derived\RBEF_PWP28_daily_area_19810901_20260930.csv -Algorithm SHA256).Hash
python .\src\RBEF\RBEF04_exploratory_pwp28_v1.0.1.py
python .\src\RBEF\RBEF05_trend_pwp28_v1.0.0.py
python .\src\RBEF\RBEF06_seasonality_pwp28_v1.0.0.py
python .\src\RBEF\RBEF07_anomalies_residuals_pwp28_v1.0.0.py
python .\src\RBEF\RBEF08_residual_diagnostics_pwp28_v1.0.1.py
python .\src\RBEF\RBEF09_forecast_hindcast_pwp28_v1.0.2.py
```

## Scope and limitations

The temporal analyses do not reproduce spatial Figure 1; that requires separately documented SST/mask/spatial inputs. RBEF10 is retained as an audit/provenance utility and requires the original full JPO04 source. No Zenodo DOI has been verified. Before publishing, compare numerical outputs and figures against original programs using the original JPO04 input, and verify the Zenodo citation/deposit.
