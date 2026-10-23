#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""
RBEF07_anomalies_residuals_pwp28_v1.0.0.py

RBEF Article 1 — anomalies and residuals of canonical Pacific Warm Pool area
(SST >= 28.0 °C).

Purpose
-------
Teach and quantify the distinction between:
  1) observed area A(t);
  2) seasonal anomaly: A(t) minus the climatological seasonal component;
  3) residual: A(t) minus a simple additive model containing linear trend
     plus the climatological seasonal component.

Scientific choices
------------------
- Canonical PWP28 only; no LCC.
- Input is the audited JPO04 daily geometry table.
- Linear trend is fitted to the complete daily record, with time in years
  since 1981-09-01.
- Seasonal component is estimated independently from COMPLETE calendar years
  (1982–2025), after removing each year's own annual mean, following RBEF06.
- Feb 29 seasonal component is the mean of Feb 28 and Mar 01 climatological
  anomalies; this is an explicit deterministic convention, not interpolation
  of missing observations.
- No Fourier, wavelets, ENSO adjustment, forecasting, or causal attribution.

Definitions
-----------
Let S(d) be the 365-day climatological seasonal anomaly for calendar day d.
Let T(t) = a + b*t be the daily-data linear trend.

Deseasonalized anomaly:
    D(t) = A(t) - S(d)

Residual:
    R(t) = A(t) - [T(t) + S(d)]

Thus:
    A(t) = T(t) + S(d) + R(t)

The term "residual" is reserved for R(t), not for the deseasonalized anomaly.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "1.0.0"
THRESHOLD = 28.0
START = pd.Timestamp("1981-09-01")
DAYS_PER_YEAR = 365.2425

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def root():
    p=Path(__file__).resolve()
    if p.parent.name.lower()=="rbef" and p.parent.parent.name.lower()=="src":
        return p.parents[2]
    raise RuntimeError("Install in <project>/src/RBEF/")

def load(p):
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


def complete_years(d):
    x=d.assign(year=d.date.dt.year,month=d.date.dt.month,day=d.date.dt.day)
    good=[]; partial=[]
    for yr,z in x.groupby("year"):
        n=366 if pd.Timestamp(int(yr),12,31).is_leap_year else 365
        ok=(len(z)==n and z.date.min()==pd.Timestamp(int(yr),1,1)
            and z.date.max()==pd.Timestamp(int(yr),12,31))
        (good if ok else partial).append(int(yr))
    return x,good,partial

def seasonal_cycle(d):
    x,good,partial=complete_years(d)
    c=x[x.year.isin(good)].copy()
    c["annual_mean"]=c.groupby("year").canonical_area_million_km2.transform("mean")
    c["within_year_anomaly"]=c.canonical_area_million_km2-c.annual_mean
    q=c[~((c.month==2)&(c.day==29))].copy()
    q["month_day"]=q.date.dt.strftime("%m-%d")
    s=q.groupby("month_day",as_index=False).agg(
        seasonal_component_million_km2=("within_year_anomaly","mean"),
        seasonal_component_std_million_km2=("within_year_anomaly","std"),
        n_years=("year","nunique"))
    return s,good,partial

def analyse(d):
    seasonal,good,partial=seasonal_cycle(d)
    smap=dict(zip(seasonal.month_day,seasonal.seasonal_component_million_km2))
    feb29=0.5*(smap["02-28"]+smap["03-01"])

    x=d.copy()
    x["month_day"]=x.date.dt.strftime("%m-%d")
    x["seasonal_component_million_km2"]=x.month_day.map(smap)
    x.loc[x.month_day=="02-29","seasonal_component_million_km2"]=feb29
    if x.seasonal_component_million_km2.isna().any():
        raise ValueError("Missing seasonal component for one or more dates")

    t=(x.date-START).dt.total_seconds().to_numpy()/86400.0/DAYS_PER_YEAR
    y=x.canonical_area_million_km2.to_numpy(float)
    X=np.column_stack((np.ones(len(t)),t))
    intercept,slope=np.linalg.lstsq(X,y,rcond=None)[0]
    trend=intercept+slope*t

    x["time_years_since_start"]=t
    x["linear_trend_million_km2"]=trend
    x["deseasonalized_area_million_km2"]=y-x.seasonal_component_million_km2.to_numpy()
    x["residual_million_km2"]=y-trend-x.seasonal_component_million_km2.to_numpy()

    resid=x.residual_million_km2.to_numpy()
    deseason=x.deseasonalized_area_million_km2.to_numpy()
    model=trend+x.seasonal_component_million_km2.to_numpy()

    ss_tot=float(np.sum((y-y.mean())**2))
    r2_additive=1-float(np.sum((y-model)**2))/ss_tot

    summary=pd.DataFrame([{
      "geometry":"canonical","threshold_c":28.0,
      "start":str(x.date.min().date()),"end":str(x.date.max().date()),"n_days":len(x),
      "trend_intercept_million_km2":float(intercept),
      "trend_slope_million_km2_per_year":float(slope),
      "trend_slope_million_km2_per_decade":float(10*slope),
      "seasonal_peak_to_trough_daily_climatology_million_km2":
          float(seasonal.seasonal_component_million_km2.max()-seasonal.seasonal_component_million_km2.min()),
      "deseasonalized_mean_million_km2":float(deseason.mean()),
      "residual_mean_million_km2":float(resid.mean()),
      "residual_std_million_km2":float(resid.std(ddof=1)),
      "residual_rmse_million_km2":float(np.sqrt(np.mean(resid**2))),
      "residual_mae_million_km2":float(np.mean(np.abs(resid))),
      "additive_model_r_squared":float(r2_additive),
      "first_complete_year":min(good),"last_complete_year":max(good),
      "n_complete_years_for_seasonality":len(good),
      "partial_years_excluded_from_seasonality":",".join(map(str,partial)),
      "feb29_seasonal_component_million_km2":float(feb29)
    }])
    return x,seasonal,summary,good,partial

def figure(x,summary,png,pdf):
    s=summary.iloc[0]
    ticks=pd.to_datetime([f"{yr}-01-01" for yr in range(1982,int(x.date.max().year)+1,4)])
    labels=[str(d.year) for d in ticks]

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained")

    ax=axs[0]
    ax.plot(x.date,x.deseasonalized_area_million_km2,color=".45",lw=.65,
            label="Deseasonalized daily area")
    ax.plot(x.date,x.linear_trend_million_km2,color="#0072B2",lw=1.7,
            label="Linear trend")
    ax.set_ylabel("Area (million km²)")
    ax.set_title("A) Deseasonalized canonical Pacific Warm Pool area — 28°C",loc="left",fontsize=12)
    ax.grid(axis="y",color=".88",lw=.5)
    ax.legend(loc="upper right",frameon=False)

    ax=axs[1]
    ax.plot(x.date,x.residual_million_km2,color=".35",lw=.65)
    ax.axhline(0,color="black",lw=.8)
    ax.set_ylabel("Residual (million km²)")
    ax.set_title("B) Residual after removing linear trend and climatological seasonality",
                 loc="left",fontsize=12)
    ax.grid(axis="y",color=".88",lw=.5)
    ax.set_xlabel(f"Year | Observed {x.date.min().date()} to {x.date.max().date()}")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)

    for ax in axs:
        ax.set_xlim(x.date.min(),x.date.max())
        ax.margins(x=0)

    note=(f"Residual mean = {s.residual_mean_million_km2:+.3f} million km²\n"
          f"Residual SD = {s.residual_std_million_km2:.2f} million km²\n"
          f"Additive model $R^2$ = {s.additive_model_r_squared:.3f}")
    axs[1].text(.015,.97,note,transform=axs[1].transAxes,va="top",fontsize=9.3,
                bbox=dict(facecolor="white",edgecolor=".75",alpha=.93,
                          boxstyle="round,pad=.35"))

    fig.suptitle("Canonical Pacific Warm Pool area — 28°C: anomalies and residuals",fontsize=14)
    fig.savefig(png,dpi=300,facecolor="white")
    fig.savefig(pdf,facecolor="white")
    plt.close(fig)

def main():
    r=root(); here=Path(__file__).resolve()
    source=r/"data"/"derived"/"RBEF_PWP28_daily_area_19810901_20260930.csv"
    out=r/"outputs"/"RBEF"
    for z in ("figures","tables","reports"): (out/z).mkdir(parents=True,exist_ok=True)

    stem="RBEF07_canonical_pwp28_anomalies_residuals_v1p0p0"
    daily_csv=out/"tables"/f"{stem}_daily.csv"
    seasonal_csv=out/"tables"/f"{stem}_seasonal_component.csv"
    summary_csv=out/"tables"/f"{stem}_summary.csv"
    png=out/"figures"/f"{stem}.png"; pdf=out/"figures"/f"{stem}.pdf"
    meta=out/"reports"/f"{stem}_metadata.json"

    print("="*78); print("RBEF07 | CANONICAL PWP28 ANOMALIES AND RESIDUALS"); print("="*78)
    print(f"Project root : {r}\nInput        : {source}")
    print("Loading and validating audited JPO04 input...")

    h=sha256(source)
    x,seasonal,summary,good,partial=analyse(load(source))
    x.to_csv(daily_csv,index=False,date_format="%Y-%m-%d")
    seasonal.to_csv(seasonal_csv,index=False)
    summary.to_csv(summary_csv,index=False)
    figure(x,summary,png,pdf)
    if sha256(source)!=h: raise RuntimeError("Input changed during analysis")

    outs=[daily_csv,seasonal_csv,summary_csv,png,pdf]
    md={
      "program":here.name,"version":VERSION,"status":"PASS",
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "scope":{"geometry":"canonical","threshold_c":28.0,"lcc_used":False,
               "fourier_used":False,"wavelets_used":False,"forecasting_used":False},
      "definitions":{
        "deseasonalized_area":"observed daily area minus 365-day climatological seasonal anomaly",
        "residual":"observed daily area minus [daily linear trend + climatological seasonal anomaly]",
        "trend":"OLS linear trend on all daily PWP28 observations",
        "seasonality":"365-day climatology estimated from complete years after removing each year's annual mean",
        "feb29":"mean of Feb 28 and Mar 01 seasonal climatological components"},
      "input":{"path":str(source.resolve()),"sha256":h,
               "start":str(x.date.min().date()),"end":str(x.date.max().date()),"n_days":len(x)},
      "results":summary.iloc[0].to_dict(),
      "script_sha256":sha256(here),
      "output_sha256":{str(p.resolve()):sha256(p) for p in outs}}
    meta.write_text(json.dumps(md,indent=2,allow_nan=False),encoding="utf-8")

    print(f"\nSeasonal climatology years : {min(good)}–{max(good)} ({len(good)} complete years)")
    print("Partial years excluded     : "+", ".join(map(str,partial)))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nOUTPUTS")
    for p in outs: print(" ",p)
    print(" ",meta)
    print(f"\nSTATUS: PASS | canonical PWP28 | {len(x):,} daily observations")
    print("="*78)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"STATUS: FAIL | {type(e).__name__}: {e}")
        raise
