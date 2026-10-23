#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""
RBEF09_forecast_hindcast_pwp28_v1.0.0.py

RBEF Article 1 — simple, interpretable forecasting of canonical PWP28 area.

Pedagogical model
-----------------
A(t) = T(t) + S(d) + R(t)

T(t): linear trend fitted only to the training data.
S(d): 365-day climatological seasonal component estimated only from complete
      training years after removing each year's own annual mean.
R(t): remaining residual. Its short-term memory is represented by AR(1):
      R(t) = phi * R(t-1) + epsilon(t).

Two forecasts are compared:
  Baseline: trend + seasonality
  Memory model: trend + seasonality + recursively decaying AR(1) residual

Hindcast
--------
Training: all observations through 2024-12-31
Test:     2025-01-01 through 2025-12-31
No 2025 observations are used to fit trend, seasonality, phi, or initialize
the 2025 forecast. This is a fixed-origin hindcast, not a fitted reconstruction.

Forward forecast
----------------
After hindcast evaluation, the same model is refitted using all observations
through 2026-09-30 and projected for 6 calendar months through 2027-03-31.

No Fourier analysis, wavelets, ENSO predictors, machine learning, or LCC.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "1.0.2"
THRESHOLD = 28.0
DAYS_PER_YEAR = 365.2425
HINDCAST_TRAIN_END = pd.Timestamp("2024-12-31")
HINDCAST_START = pd.Timestamp("2025-01-01")
HINDCAST_END = pd.Timestamp("2025-12-31")
FORECAST_END = pd.Timestamp("2027-03-31")

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
    good=[]
    for yr,z in d.groupby(d.date.dt.year):
        expected=366 if pd.Timestamp(int(yr),12,31).is_leap_year else 365
        if (len(z)==expected and z.date.min()==pd.Timestamp(int(yr),1,1)
            and z.date.max()==pd.Timestamp(int(yr),12,31)):
            good.append(int(yr))
    return good

def fit_model(train):
    train=train.copy()
    origin=train.date.iloc[0]
    t=(train.date-origin).dt.total_seconds().to_numpy()/86400.0/DAYS_PER_YEAR
    y=train.canonical_area_million_km2.to_numpy(float)
    X=np.column_stack([np.ones(len(t)),t])
    intercept,slope=np.linalg.lstsq(X,y,rcond=None)[0]
    trend=intercept+slope*t

    # Seasonal climatology from complete training years, using within-year
    # anomalies so long-term changes do not become part of the seasonal cycle.
    years=complete_years(train)
    c=train[train.date.dt.year.isin(years)].copy()
    c["annual_mean"]=c.groupby(c.date.dt.year).canonical_area_million_km2.transform("mean")
    c["within_year_anomaly"]=c.canonical_area_million_km2-c.annual_mean
    c=c[~((c.date.dt.month==2)&(c.date.dt.day==29))].copy()
    c["month_day"]=c.date.dt.strftime("%m-%d")
    seas=c.groupby("month_day").within_year_anomaly.mean().to_dict()
    seas["02-29"]=0.5*(seas["02-28"]+seas["03-01"])

    s=np.array([seas[x] for x in train.date.dt.strftime("%m-%d")])
    resid=y-trend-s

    # Zero-intercept AR(1) coefficient: residual_t = phi*residual_(t-1)+epsilon.
    phi=float(np.dot(resid[:-1],resid[1:])/np.dot(resid[:-1],resid[:-1]))
    if not (-1 < phi < 1):
        raise ValueError(f"Estimated AR(1) phi is not stationary: {phi}")

    innovations=resid[1:]-phi*resid[:-1]
    return {
        "origin":origin,"intercept":float(intercept),"slope":float(slope),
        "seasonal":seas,"phi":phi,"last_residual":float(resid[-1]),
        "innovation_sd":float(np.std(innovations,ddof=1)),
        "complete_years":years
    }

def predict(model, dates, include_memory=True):
    dates=pd.DatetimeIndex(dates)
    t=(dates-model["origin"]).total_seconds()/86400.0/DAYS_PER_YEAR
    trend=model["intercept"]+model["slope"]*np.asarray(t,float)
    seasonal=np.array([model["seasonal"][d.strftime("%m-%d")] for d in dates])
    h=np.arange(1,len(dates)+1,dtype=float)
    memory=model["last_residual"]*(model["phi"]**h) if include_memory else np.zeros(len(dates))
    return trend+seasonal+memory,trend,seasonal,memory

def metrics(obs,pred):
    e=np.asarray(pred)-np.asarray(obs)
    return {
        "mae_million_km2":float(np.mean(np.abs(e))),
        "rmse_million_km2":float(np.sqrt(np.mean(e**2))),
        "bias_million_km2":float(np.mean(e))
    }

def figures(test, future, outstem):
    # Hindcast
    fig,ax=plt.subplots(figsize=(11,6),layout="constrained")
    ax.plot(test.date,test.observed_million_km2,color="0.35",lw=1.15,label="Observed 2025")
    ax.plot(test.date,test.baseline_forecast_million_km2,color="0.55",lw=1.25,
            linestyle="--",label="Baseline: trend + seasonality")
    ax.plot(test.date,test.memory_forecast_million_km2,color="black",lw=1.6,
            label="Trend + seasonality + AR(1) residual memory")
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: 2025 fixed-origin hindcast")
    ax.set_ylabel("Area (million km²)")
    ax.set_xlabel("Date | Model trained only through 2024-12-31")
    ax.grid(axis="y",color=".9",lw=.5); ax.legend(frameon=False)
    fig.savefig(str(outstem)+"_hindcast.png",dpi=300,facecolor="white")
    fig.savefig(str(outstem)+"_hindcast.pdf",facecolor="white"); plt.close(fig)

    # Forward forecast
    fig,ax=plt.subplots(figsize=(11,6),layout="constrained")
    ax.plot(future.date,future.baseline_forecast_million_km2,color="0.55",lw=1.25,
            linestyle="--",label="Baseline: trend + seasonality")
    ax.plot(future.date,future.memory_forecast_million_km2,color="black",lw=1.7,
            label="Forecast with AR(1) residual memory")
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: six-month forecast")
    ax.set_ylabel("Forecast area (million km²)")
    ax.set_xlabel("Date | Forecast initialized after 2026-09-30")
    ax.grid(axis="y",color=".9",lw=.5); ax.legend(frameon=False)
    fig.savefig(str(outstem)+"_forecast.png",dpi=300,facecolor="white")
    fig.savefig(str(outstem)+"_forecast.pdf",facecolor="white"); plt.close(fig)

def main():
    r=root(); here=Path(__file__).resolve()
    source=r/"data"/"derived"/"RBEF_PWP28_daily_area_19810901_20260930.csv"
    out=r/"outputs"/"RBEF"
    for z in ("figures","tables","reports"): (out/z).mkdir(parents=True,exist_ok=True)
    stem="RBEF09_canonical_pwp28_forecast_hindcast_v1p0p0"

    print("="*78); print("RBEF09 | CANONICAL PWP28 FORECAST AND HINDCAST"); print("="*78)
    print(f"Project root : {r}\nInput        : {source}")
    print("Loading and validating audited JPO04 input...")
    h=sha256(source); d=load(source)

    # Leakage-free 2025 hindcast.
    train=d[d.date<=HINDCAST_TRAIN_END].copy()
    test=d[(d.date>=HINDCAST_START)&(d.date<=HINDCAST_END)].copy()
    if len(test)!=365: raise ValueError("2025 test period is incomplete")
    m=fit_model(train)
    pb,*_=predict(m,test.date,False)
    pm,*_=predict(m,test.date,True)
    testout=pd.DataFrame({
        "date":test.date,
        "observed_million_km2":test.canonical_area_million_km2.to_numpy(),
        "baseline_forecast_million_km2":pb,
        "memory_forecast_million_km2":pm
    })
    mb=metrics(testout.observed_million_km2,pb)
    mm=metrics(testout.observed_million_km2,pm)

    # Refit to all observations, then forecast from next day to 2027-03-31.
    mall=fit_model(d)
    future_dates=pd.date_range(d.date.max()+pd.Timedelta(days=1),FORECAST_END,freq="D")
    fb,trend,seas,_=predict(mall,future_dates,False)
    fm,_,_,memory=predict(mall,future_dates,True)
    future=pd.DataFrame({
        "date":future_dates,
        "trend_million_km2":trend,
        "seasonal_component_million_km2":seas,
        "ar1_memory_component_million_km2":memory,
        "baseline_forecast_million_km2":fb,
        "memory_forecast_million_km2":fm
    })

    summary=pd.DataFrame([
      {"model":"Baseline: trend + seasonality",**mb},
      {"model":"Trend + seasonality + AR(1) residual memory",**mm}
    ])
    summary["mae_improvement_vs_baseline_percent"]=np.nan
    summary["rmse_improvement_vs_baseline_percent"]=np.nan
    summary.loc[1,"mae_improvement_vs_baseline_percent"]=100*(mb["mae_million_km2"]-mm["mae_million_km2"])/mb["mae_million_km2"]
    summary.loc[1,"rmse_improvement_vs_baseline_percent"]=100*(mb["rmse_million_km2"]-mm["rmse_million_km2"])/mb["rmse_million_km2"]

    modelpars=pd.DataFrame([
      {"fit":"hindcast_training","training_start":str(train.date.min().date()),
       "training_end":str(train.date.max().date()),"intercept_million_km2":m["intercept"],
       "slope_million_km2_per_year":m["slope"],
       "slope_million_km2_per_decade":10*m["slope"],"ar1_phi":m["phi"],
       "last_training_residual_million_km2":m["last_residual"],
       "innovation_sd_million_km2":m["innovation_sd"]},
      {"fit":"full_record_forecast","training_start":str(d.date.min().date()),
       "training_end":str(d.date.max().date()),"intercept_million_km2":mall["intercept"],
       "slope_million_km2_per_year":mall["slope"],
       "slope_million_km2_per_decade":10*mall["slope"],"ar1_phi":mall["phi"],
       "last_training_residual_million_km2":mall["last_residual"],
       "innovation_sd_million_km2":mall["innovation_sd"]}
    ])

    testcsv=out/"tables"/f"{stem}_2025_hindcast.csv"
    futurecsv=out/"tables"/f"{stem}_six_month_forecast.csv"
    sumcsv=out/"tables"/f"{stem}_evaluation.csv"
    parscsv=out/"tables"/f"{stem}_model_parameters.csv"
    testout.to_csv(testcsv,index=False,date_format="%Y-%m-%d")
    future.to_csv(futurecsv,index=False,date_format="%Y-%m-%d")
    summary.to_csv(sumcsv,index=False)
    modelpars.to_csv(parscsv,index=False)

    figstem=out/"figures"/stem
    figures(testout,future,figstem)
    figs=[Path(str(figstem)+"_hindcast.png"),Path(str(figstem)+"_hindcast.pdf"),
          Path(str(figstem)+"_forecast.png"),Path(str(figstem)+"_forecast.pdf")]
    outs=[testcsv,futurecsv,sumcsv,parscsv]+figs

    if sha256(source)!=h: raise RuntimeError("Input changed during analysis")
    meta=out/"reports"/f"{stem}_metadata.json"
    md={
      "program":here.name,"version":VERSION,"status":"PASS",
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "scope":{"geometry":"canonical","threshold_c":28.0,"fourier_used":False,
               "wavelets_used":False,"enso_predictors_used":False,"lcc_used":False},
      "hindcast":{"training_end":"2024-12-31","test_start":"2025-01-01",
                  "test_end":"2025-12-31","fixed_origin":True,
                  "future_test_observations_used_in_fitting":False},
      "forecast":{"training_end":str(d.date.max().date()),
                  "forecast_start":str(future_dates.min().date()),
                  "forecast_end":str(future_dates.max().date())},
      "model":{"equation":"Ahat(t+h)=T(t+h)+S(d)+phi^h*R(t)",
               "baseline":"Ahat(t+h)=T(t+h)+S(d)",
               "ar1":"zero-intercept AR(1) fitted to training residuals"},
      "input":{"path":str(source.resolve()),"sha256":h},
      # JSON has no NaN value.  The baseline row deliberately has no
      # "improvement versus baseline" statistic, so pandas stores NaN there.
      # Convert missing numeric values to JSON null before serialization.
      "evaluation":summary.replace({np.nan: None}).to_dict(orient="records"),
      "model_parameters":modelpars.replace({np.nan: None}).to_dict(orient="records"),
      "script_sha256":sha256(here),
      "output_sha256":{str(p.resolve()):sha256(p) for p in outs}
    }
    meta.write_text(json.dumps(md,indent=2,allow_nan=False),encoding="utf-8")

    print("\n2025 FIXED-ORIGIN HINDCAST EVALUATION")
    print(summary.to_string(index=False))
    print("\nMODEL PARAMETERS")
    print(modelpars.to_string(index=False))
    print(f"\nFORWARD FORECAST: {future_dates.min().date()} to {future_dates.max().date()} ({len(future_dates)} days)")
    print("\nOUTPUTS")
    for p in outs: print(" ",p)
    print(" ",meta)
    print("\nSTATUS: PASS | leakage-free 2025 hindcast + six-month forward forecast")
    print("="*78)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"STATUS: FAIL | {type(e).__name__}: {e}")
        raise
