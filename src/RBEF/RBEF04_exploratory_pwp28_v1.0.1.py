#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""
RBEF04_exploratory_pwp28_v1.0.1.py

RBEF Article 1 — exploratory analysis of the canonical Pacific Warm Pool
defined by SST >= 28.0 °C.

Scientific scope
----------------
- Uses ONLY the canonical PWP at 28.0 °C.
- Does NOT use LCC geometry.
- Reads the already-audited JPO04 daily geometry table; it does not recompute SST.
- Preserves the complete daily record present in that table.
- Computes descriptive statistics.
- Computes a centered 365-observation moving mean (full windows only).
- Fits a descriptive OLS linear trend to ALL ORIGINAL DAILY AREA VALUES.
  The moving mean is NOT used to estimate the trend.
- Time for the regression is elapsed days / 365.2425 since the first observation.
- Reports the slope in million km²/year and million km²/decade, the fitted
  change over the observed interval, and R².
- No significance claim, serial-correlation correction, seasonal removal,
  ENSO adjustment, interpolation, forecasting, or LCC analysis is performed.

Expected installation
---------------------
<project>/src/RBEF/RBEF04_exploratory_pwp28_v1.0.1.py

Input
-----
<project>/outputs/JPO/tables/JPO04_daily_pwp_geometry.csv

Outputs
-------
<project>/outputs/RBEF/tables/
<project>/outputs/RBEF/figures/
<project>/outputs/RBEF/reports/

Plot convention
---------------
- Daily canonical area: light gray.
- Centered 365-day mean: black.
- Linear trend fitted to the daily data by OLS: blue.
- X-axis major ticks: 1982, 1986, 1990, ..., every four years.
- Plot border follows the observed date range exactly.
- Trend equation uses t = years since the first observation and
  A = area in million km².
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROGRAM = "RBEF04"
VERSION = "1.0.1"
THRESHOLD_C = 28.0
EXPECTED_START_DATE = pd.Timestamp("1981-09-01")
YEAR_DAYS = 365.2425
MOVING_WINDOW = 365


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "rbef" and here.parent.parent.name.lower() == "src":
        return here.parents[2]
    raise RuntimeError(
        "Place this program in <project>/src/RBEF/ so the project root is unambiguous."
    )


def load_canonical_pwp28(p):
    d = pd.read_csv(p, usecols=["date", "pwp28_area_million_km2"])
    d = d.rename(columns={"pwp28_area_million_km2": "canonical_area_million_km2"})
    d["threshold_c"] = 28.0
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d", errors="raise")
    d = d.sort_values("date").reset_index(drop=True)
    if d.empty or d.date.duplicated().any():
        raise ValueError("Invalid PWP28 record")
    if not pd.DatetimeIndex(d.date).equals(pd.date_range(d.date.min(), d.date.max(), freq="D")):
        raise ValueError("Incomplete daily record")
    if d.date.iloc[0] != pd.Timestamp("1981-09-01") or d.date.iloc[-1] != pd.Timestamp("2026-09-30") or len(d) != 16466:
        raise ValueError("Unexpected date coverage")
    y = d.canonical_area_million_km2.to_numpy(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid area values")
    return d


def analyse(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = g.copy()

    y = out["canonical_area_million_km2"].to_numpy(dtype=float)
    t_years = (
        (out["date"] - out["date"].iloc[0]).dt.total_seconds().to_numpy()
        / (86400.0 * YEAR_DAYS)
    )

    # Centered 365-observation mean. Full windows only.
    out["canonical_moving_mean_365d"] = (
        out["canonical_area_million_km2"]
        .rolling(MOVING_WINDOW, center=True, min_periods=MOVING_WINDOW)
        .mean()
    )

    # OLS: A(t) = intercept + slope * t
    X = np.column_stack((np.ones(len(t_years)), t_years))
    intercept, slope = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = intercept + slope * t_years
    out["canonical_ols_trend"] = fitted
    out["elapsed_years_since_start"] = t_years

    residual = y - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot)

    elapsed_years = float(t_years[-1])
    fitted_change = float(slope * elapsed_years)

    stats = pd.DataFrame([{
        "geometry": "canonical",
        "threshold_c": THRESHOLD_C,
        "start": out["date"].iloc[0].date().isoformat(),
        "end": out["date"].iloc[-1].date().isoformat(),
        "n_days": int(len(out)),
        "mean_million_km2": float(np.mean(y)),
        "std_million_km2": float(np.std(y, ddof=1)),
        "min_million_km2": float(np.min(y)),
        "q05_million_km2": float(np.quantile(y, 0.05)),
        "q25_million_km2": float(np.quantile(y, 0.25)),
        "median_million_km2": float(np.median(y)),
        "q75_million_km2": float(np.quantile(y, 0.75)),
        "q95_million_km2": float(np.quantile(y, 0.95)),
        "max_million_km2": float(np.max(y)),
        "ols_intercept_million_km2": float(intercept),
        "ols_slope_million_km2_per_year": float(slope),
        "ols_slope_million_km2_per_decade": float(10.0 * slope),
        "ols_fitted_change_million_km2": fitted_change,
        "ols_r_squared": r_squared,
        "ols_elapsed_years": elapsed_years,
    }])

    return out, stats


def make_figure(g: pd.DataFrame, stats: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    fit = stats.iloc[0]
    start = g["date"].iloc[0]
    end = g["date"].iloc[-1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(12.0, 5.8), layout="constrained")

    ax.plot(
        g["date"],
        g["canonical_area_million_km2"],
        color="0.72",
        linewidth=0.50,
        label="Observed daily area",
        zorder=1,
    )
    ax.plot(
        g["date"],
        g["canonical_moving_mean_365d"],
        color="black",
        linewidth=1.55,
        label="Centered 365-day mean",
        zorder=3,
    )
    ax.plot(
        g["date"],
        g["canonical_ols_trend"],
        color="#0072B2",
        linewidth=1.55,
        label="Linear trend (daily data)",
        zorder=2,
    )

    ax.set_title("Canonical Pacific Warm Pool area — 28°C", fontsize=14)
    ax.set_ylabel("Area (million km²)")
    ax.set_xlabel(f"Year | Observed {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    # Exact observed temporal borders: no artificial 1980/2027 padding.
    ax.set_xlim(start, end)
    ax.margins(x=0)

    # Major ticks explicitly anchored at 1982 and repeated every four years.
    tick_years = list(range(1982, end.year + 1, 4))
    tick_dates = [pd.Timestamp(year=y, month=1, day=1) for y in tick_years]
    tick_dates = [d for d in tick_dates if start <= d <= end]
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Data-driven vertical limits with a small, symmetric visual margin.
    values = g["canonical_area_million_km2"].to_numpy(dtype=float)
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    yrange = ymax - ymin
    ypad = 0.035 * yrange if yrange > 0 else 1.0
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.grid(axis="y", color="0.88", linewidth=0.55)

    # Equation is expressed in pedagogically meaningful units.
    intercept = float(fit["ols_intercept_million_km2"])
    slope_year = float(fit["ols_slope_million_km2_per_year"])
    slope_decade = float(fit["ols_slope_million_km2_per_decade"])
    r2 = float(fit["ols_r_squared"])

    sign = "+" if slope_year >= 0 else "−"
    equation = (
        rf"$A(t)={intercept:.2f}\ {sign}\ {abs(slope_year):.3f}t$"
        "\n"
        rf"Linear trend = {slope_decade:+.2f} million km$^2$/decade"
        "\n"
        rf"$R^2={r2:.3f}$;  $t$ = years since {start:%Y-%m-%d}"
    )

    ax.text(
        0.015,
        0.975,
        equation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#0072B2",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, boxstyle="round,pad=0.35"),
        zorder=5,
    )

    ax.legend(loc="upper right", frameon=False)

    fig.savefig(path_png, dpi=300, facecolor="white")
    fig.savefig(path_pdf, facecolor="white")
    plt.close(fig)


def main() -> None:
    root = project_root_from_script()
    here = Path(__file__).resolve()

    source = root / "data" / "derived" / "RBEF_PWP28_daily_area_19810901_20260930.csv"
    figures = root / "outputs" / "RBEF" / "figures"
    tables = root / "outputs" / "RBEF" / "tables"
    reports = root / "outputs" / "RBEF" / "reports"

    for directory in (figures, tables, reports):
        directory.mkdir(parents=True, exist_ok=True)

    stem = "RBEF04_canonical_pwp28_exploratory_v1p0p1"

    daily_csv = tables / f"{stem}_daily_and_moving_mean.csv"
    stats_csv = tables / f"{stem}_descriptive_statistics_and_trend.csv"
    figure_png = figures / f"{stem}.png"
    figure_pdf = figures / f"{stem}.pdf"
    metadata_json = reports / f"{stem}_metadata.json"

    print("=" * 78)
    print("RBEF04 | EXPLORATORY CANONICAL PWP28 AREA")
    print("=" * 78)
    print(f"Project root : {root}")
    print(f"Input        : {source}")
    print(f"RBEF figures : {figures}")
    print(f"RBEF tables  : {tables}")
    print(f"RBEF reports : {reports}")
    print("Loading and validating audited JPO04 input...")

    initial_input_hash = sha256(source)
    g = load_canonical_pwp28(source)
    analysed, stats = analyse(g)

    analysed.to_csv(daily_csv, index=False, date_format="%Y-%m-%d")
    stats.to_csv(stats_csv, index=False)

    make_figure(analysed, stats, figure_png, figure_pdf)

    if sha256(source) != initial_input_hash:
        raise RuntimeError("Input changed during RBEF04 analysis.")

    outputs = [daily_csv, stats_csv, figure_png, figure_pdf]

    metadata = {
        "program": here.name,
        "program_id": PROGRAM,
        "version": VERSION,
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root.resolve()),
        "scientific_scope": {
            "geometry": "canonical",
            "threshold_c": THRESHOLD_C,
            "lcc_used": False,
            "sst_recomputed": False,
        },
        "input": {
            "path": str(source.resolve()),
            "sha256": initial_input_hash,
            "source_product": "JPO04 audited daily PWP geometry table",
        },
        "record": {
            "start": str(analysed["date"].min().date()),
            "end": str(analysed["date"].max().date()),
            "n_days": int(len(analysed)),
            "daily_continuity": True,
        },
        "moving_mean": {
            "window_observations": MOVING_WINDOW,
            "centered": True,
            "min_periods": MOVING_WINDOW,
            "edge_values": "missing",
            "interpolation": False,
            "used_for_trend_fit": False,
        },
        "trend": {
            "method": "ordinary least squares",
            "input": "all original daily canonical PWP28 area values",
            "time_definition": f"elapsed days / {YEAR_DAYS}",
            "time_origin": str(analysed["date"].iloc[0].date()),
            "seasonal_removal": False,
            "serial_correlation_correction": False,
            "significance_test": False,
            "intercept_million_km2": float(stats.iloc[0]["ols_intercept_million_km2"]),
            "slope_million_km2_per_year": float(stats.iloc[0]["ols_slope_million_km2_per_year"]),
            "slope_million_km2_per_decade": float(stats.iloc[0]["ols_slope_million_km2_per_decade"]),
            "fitted_change_million_km2": float(stats.iloc[0]["ols_fitted_change_million_km2"]),
            "r_squared": float(stats.iloc[0]["ols_r_squared"]),
        },
        "figure": {
            "x_limits": [
                str(analysed["date"].min().date()),
                str(analysed["date"].max().date()),
            ],
            "major_tick_start_year": 1982,
            "major_tick_interval_years": 4,
            "png_dpi": 300,
            "pdf_vector": True,
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "script_sha256": sha256(here),
        "output_sha256": {str(p.resolve()): sha256(p) for p in outputs},
    }

    metadata_json.write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nDESCRIPTIVE STATISTICS AND TREND")
    print(stats.to_string(index=False))

    print("\nOUTPUTS")
    for p in outputs:
        print(f"  {p}")
    print(f"  {metadata_json}")

    print(
        f"\nSTATUS: PASS | {len(analysed):,} continuous daily observations | "
        f"{analysed['date'].min():%Y-%m-%d} to {analysed['date'].max():%Y-%m-%d}"
    )
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STATUS: FAIL | {type(exc).__name__}: {exc}")
        raise
