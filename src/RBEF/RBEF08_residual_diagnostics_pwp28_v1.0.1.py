#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""
RBEF08_residual_diagnostics_pwp28_v1.0.1.py

RBEF Article 1 — temporal diagnostics of canonical PWP28 residuals.

Reads the RBEF07 daily product and asks three pedagogical questions:
1. What is the distribution of the residuals?
2. How large are the most extreme residuals?
3. Do consecutive residuals remain temporally dependent?

Outputs:
- residual descriptive statistics and selected quantiles;
- largest positive/negative residual events;
- autocorrelation function (ACF) at selected daily lags;
- two publication figures:
    Figure A: residual distribution + normal reference;
    Figure B: ACF at pedagogically relevant lags.
No Fourier analysis, wavelets, ENSO classification, forecasting, or LCC.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "1.0.1"
THRESHOLD = 28.0
LAGS = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365]

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def root():
    p = Path(__file__).resolve()
    if p.parent.name.lower() == "rbef" and p.parent.parent.name.lower() == "src":
        return p.parents[2]
    raise RuntimeError("Install in <project>/src/RBEF/")

def load(p):
    if not p.is_file():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    required = {"date", "threshold_c", "residual_million_km2"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    d = d.loc[np.isclose(d["threshold_c"].astype(float), THRESHOLD)].copy()
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d", errors="raise")
    d = d.sort_values("date").reset_index(drop=True)
    if d.empty or d["date"].duplicated().any():
        raise ValueError("Invalid RBEF07 PWP28 daily product")
    expected = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    if not pd.DatetimeIndex(d["date"]).equals(expected):
        raise ValueError("RBEF07 daily record is not continuous")
    r = d["residual_million_km2"].to_numpy(float)
    if not np.isfinite(r).all():
        raise ValueError("Non-finite residual values")
    return d

def acf_at_lag(x, lag):
    x = np.asarray(x, float)
    if lag <= 0 or lag >= len(x):
        raise ValueError("Invalid lag")
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1])

def analyse(d):
    r = d["residual_million_km2"].to_numpy(float)
    n = len(r)
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    centered = r - mean

    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / (m2**1.5)
    excess_kurtosis = m4 / (m2**2) - 3.0

    qlevels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    qvals = np.quantile(r, qlevels)

    summary = pd.DataFrame([{
        "geometry": "canonical",
        "threshold_c": THRESHOLD,
        "start": str(d["date"].min().date()),
        "end": str(d["date"].max().date()),
        "n_days": n,
        "mean_million_km2": mean,
        "std_million_km2": sd,
        "min_million_km2": float(np.min(r)),
        "q01_million_km2": float(qvals[0]),
        "q05_million_km2": float(qvals[1]),
        "q25_million_km2": float(qvals[2]),
        "median_million_km2": float(qvals[3]),
        "q75_million_km2": float(qvals[4]),
        "q95_million_km2": float(qvals[5]),
        "q99_million_km2": float(qvals[6]),
        "max_million_km2": float(np.max(r)),
        "skewness": float(skew),
        "excess_kurtosis": float(excess_kurtosis),
    }])

    acf = pd.DataFrame({
        "lag_days": LAGS,
        "autocorrelation": [acf_at_lag(r, lag) for lag in LAGS]
    })
    # Approximate 95% white-noise reference bounds, for visual teaching only.
    acf["white_noise_95_bound"] = 1.96 / np.sqrt(n)

    k = 10
    neg = d.nsmallest(k, "residual_million_km2")[["date", "residual_million_km2"]].copy()
    neg["extreme_type"] = "negative"
    pos = d.nlargest(k, "residual_million_km2")[["date", "residual_million_km2"]].copy()
    pos["extreme_type"] = "positive"
    extremes = pd.concat([neg, pos], ignore_index=True)
    extremes = extremes[["extreme_type", "date", "residual_million_km2"]]

    return summary, acf, extremes

def normal_pdf(x, mean, sd):
    return np.exp(-0.5*((x-mean)/sd)**2) / (sd*np.sqrt(2*np.pi))

def distribution_figure(d, summary, png, pdf):
    r = d["residual_million_km2"].to_numpy(float)
    s = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    ax.hist(r, bins=60, density=True, color="0.78", edgecolor="0.35",
            linewidth=0.45, label="Observed residual distribution")
    xx = np.linspace(r.min(), r.max(), 500)
    ax.plot(xx, normal_pdf(xx, s.mean_million_km2, s.std_million_km2),
            color="black", linewidth=1.6,
            label="Normal reference with same mean and SD")
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: residual distribution")
    ax.set_xlabel("Residual (million km²)")
    ax.set_ylabel("Probability density")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend(frameon=False)
    note = (f"n = {int(s.n_days):,}\n"
            f"Mean = {s.mean_million_km2:+.3f} million km²\n"
            f"SD = {s.std_million_km2:.2f} million km²\n"
            f"Skewness = {s.skewness:.2f}\n"
            f"Excess kurtosis = {s.excess_kurtosis:.2f}")
    ax.text(.015, .97, note, transform=ax.transAxes, va="top", fontsize=9.3,
            bbox=dict(facecolor="white", edgecolor=".75", alpha=.93,
                      boxstyle="round,pad=.35"))
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

def acf_figure(acf, png, pdf):
    bound = float(acf["white_noise_95_bound"].iloc[0])
    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(bound, color="0.55", linewidth=0.9, linestyle="--",
               label="Approx. 95% white-noise bound")
    ax.axhline(-bound, color="0.55", linewidth=0.9, linestyle="--")
    ax.plot(acf["lag_days"], acf["autocorrelation"], color="black",
            marker="o", linewidth=1.4, markersize=5,
            label="Residual autocorrelation")
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: residual persistence")
    ax.set_xlabel("Lag (days; logarithmic scale)")
    ax.set_ylabel("Autocorrelation")
    # Logarithmic x-axis preserves the physical lag values while separating
    # the short lags (1, 2, 3, 7 days) that overlap on a 0–365 linear axis.
    ax.set_xscale("log")
    ax.set_xticks(LAGS)
    ax.set_xticklabels([str(v) for v in LAGS])
    ax.set_xlim(0.9, 400)
    ax.set_ylim(-1.0, 1.0)
    ax.minorticks_off()
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

def main():
    r = root()
    here = Path(__file__).resolve()
    source = r/"outputs"/"RBEF"/"tables"/"RBEF07_canonical_pwp28_anomalies_residuals_v1p0p0_daily.csv"
    out = r/"outputs"/"RBEF"
    for z in ("figures", "tables", "reports"):
        (out/z).mkdir(parents=True, exist_ok=True)

    stem = "RBEF08_canonical_pwp28_residual_diagnostics_v1p0p1"
    summary_csv = out/"tables"/f"{stem}_summary.csv"
    acf_csv = out/"tables"/f"{stem}_acf.csv"
    extremes_csv = out/"tables"/f"{stem}_extremes.csv"
    dist_png = out/"figures"/f"{stem}_distribution.png"
    dist_pdf = out/"figures"/f"{stem}_distribution.pdf"
    acf_png = out/"figures"/f"{stem}_acf.png"
    acf_pdf = out/"figures"/f"{stem}_acf.pdf"
    meta = out/"reports"/f"{stem}_metadata.json"

    print("="*78)
    print("RBEF08 | CANONICAL PWP28 RESIDUAL DIAGNOSTICS")
    print("="*78)
    print(f"Project root : {r}")
    print(f"Input        : {source}")
    print("Loading and validating RBEF07 daily residual product...")

    h = sha256(source)
    d = load(source)
    summary, acf, extremes = analyse(d)

    summary.to_csv(summary_csv, index=False)
    acf.to_csv(acf_csv, index=False)
    extremes.to_csv(extremes_csv, index=False, date_format="%Y-%m-%d")
    distribution_figure(d, summary, dist_png, dist_pdf)
    acf_figure(acf, acf_png, acf_pdf)

    if sha256(source) != h:
        raise RuntimeError("Input changed during RBEF08 analysis")

    outs=[summary_csv,acf_csv,extremes_csv,dist_png,dist_pdf,acf_png,acf_pdf]
    md = {
        "program": here.name, "version": VERSION, "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"geometry":"canonical","threshold_c":28.0,
                  "domain":"time","fourier_used":False,"wavelets_used":False,
                  "forecasting_used":False,"enso_classification_used":False},
        "input": {"path":str(source.resolve()),"sha256":h,
                  "start":str(d.date.min().date()),"end":str(d.date.max().date()),
                  "n_days":len(d)},
        "diagnostics": {
            "distribution":"histogram and normal reference with same mean and SD",
            "extremes":"10 largest positive and 10 largest negative daily residuals",
            "acf_lags_days":LAGS,
            "acf_x_axis":"logarithmic; physical lag values unchanged",
            "acf_definition":"Pearson correlation between residual(t) and residual(t-lag)",
            "white_noise_reference":"plus/minus 1.96/sqrt(N); descriptive visual reference only"
        },
        "results": summary.iloc[0].to_dict(),
        "script_sha256": sha256(here),
        "output_sha256": {str(p.resolve()):sha256(p) for p in outs}
    }
    meta.write_text(json.dumps(md, indent=2, allow_nan=False), encoding="utf-8")

    print("\nRESIDUAL DISTRIBUTION")
    print(summary.to_string(index=False))
    print("\nAUTOCORRELATION")
    print(acf.to_string(index=False))
    print("\nTEN MOST NEGATIVE / TEN MOST POSITIVE RESIDUALS")
    print(extremes.to_string(index=False))
    print("\nOUTPUTS")
    for p in outs: print(" ", p)
    print(" ", meta)
    print(f"\nSTATUS: PASS | canonical PWP28 | {len(d):,} daily residuals diagnosed")
    print("="*78)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"STATUS: FAIL | {type(e).__name__}: {e}")
        raise
