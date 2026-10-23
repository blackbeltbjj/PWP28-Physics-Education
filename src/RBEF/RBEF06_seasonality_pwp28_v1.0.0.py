#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""RBEF06_seasonality_pwp28_v1.0.0.py
Seasonality of canonical Pacific Warm Pool area (SST >= 28.0 C).

Uses audited JPO04 daily geometry. Canonical PWP28 only.
Seasonal climatology uses complete calendar years only and removes each
year's own annual mean before estimating the seasonal anomaly, preventing
the long-term increase in area from contaminating the seasonal cycle.
Feb 29 is excluded only from the 365-day daily climatological calendar.
No Fourier, wavelets, ENSO adjustment, forecasting, interpolation, or LCC.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION="1.0.0"; THRESHOLD=28.0
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def root():
    p=Path(__file__).resolve()
    if p.parent.name.lower()=="rbef" and p.parent.parent.name.lower()=="src": return p.parents[2]
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


def analyse(d):
    x=d.assign(year=d.date.dt.year,month=d.date.dt.month,day=d.date.dt.day)
    complete=[]; partial=[]
    for yr,z in x.groupby("year"):
        n=366 if pd.Timestamp(int(yr),12,31).is_leap_year else 365
        ok=len(z)==n and z.date.min()==pd.Timestamp(int(yr),1,1) and z.date.max()==pd.Timestamp(int(yr),12,31)
        (complete if ok else partial).append(int(yr))
    c=x[x.year.isin(complete)].copy()
    c["annual_mean_million_km2"]=c.groupby("year").canonical_area_million_km2.transform("mean")
    c["within_year_anomaly_million_km2"]=c.canonical_area_million_km2-c.annual_mean_million_km2

    m=c.groupby("month",as_index=False).agg(
      climatological_mean_area_million_km2=("canonical_area_million_km2","mean"),
      seasonal_anomaly_million_km2=("within_year_anomaly_million_km2","mean"),
      daily_std_million_km2=("canonical_area_million_km2","std"),
      n_daily_observations=("canonical_area_million_km2","size"),n_years=("year","nunique"))
    m["month_name"]=[MONTHS[i-1] for i in m.month]

    q=c[~((c.month==2)&(c.day==29))].copy()
    q["month_day"]=q.date.dt.strftime("%m-%d")
    q["clim_day"]=pd.to_datetime("2001-"+q.month_day).dt.dayofyear
    dc=q.groupby(["clim_day","month_day"],as_index=False).agg(
      seasonal_anomaly_million_km2=("within_year_anomaly_million_km2","mean"),
      climatological_mean_area_million_km2=("canonical_area_million_km2","mean"),
      anomaly_std_million_km2=("within_year_anomaly_million_km2","std"),n_years=("year","nunique"))
    dc=dc.sort_values("clim_day").reset_index(drop=True)

    peak=m.loc[m.seasonal_anomaly_million_km2.idxmax()]
    trough=m.loc[m.seasonal_anomaly_million_km2.idxmin()]
    s=pd.DataFrame([{
      "geometry":"canonical","threshold_c":28.0,"first_complete_year":min(complete),
      "last_complete_year":max(complete),"n_complete_years":len(complete),
      "partial_years_excluded":",".join(map(str,partial)),
      "peak_month":peak.month_name,"peak_seasonal_anomaly_million_km2":peak.seasonal_anomaly_million_km2,
      "trough_month":trough.month_name,"trough_seasonal_anomaly_million_km2":trough.seasonal_anomaly_million_km2,
      "peak_to_trough_amplitude_million_km2":peak.seasonal_anomaly_million_km2-trough.seasonal_anomaly_million_km2}])
    return m,dc,s,complete,partial

def figure(m,dc,s,png,pdf):
    z=s.iloc[0]; mid=np.array([pd.Timestamp(2001,i,15).dayofyear for i in range(1,13)])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(11.5,5.8),layout="constrained")
    ax.plot(dc.clim_day,dc.seasonal_anomaly_million_km2,color=".65",lw=1,
            label="Daily climatological seasonal anomaly")
    ax.plot(mid,m.seasonal_anomaly_million_km2,color="black",lw=1.8,marker="o",ms=5,
            label="Monthly seasonal anomaly")
    ax.axhline(0,color=".35",lw=.8)
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: climatological seasonal cycle",fontsize=14)
    ax.set_ylabel("Seasonal area anomaly (million km²)")
    ax.set_xlabel(f"Calendar month | Complete years {int(z.first_complete_year)}–{int(z.last_complete_year)}")
    ax.set_xlim(1,365); ax.set_xticks(mid); ax.set_xticklabels(MONTHS); ax.grid(axis="y",color=".88",lw=.55)
    note=(f"Peak month: {z.peak_month}\nTrough month: {z.trough_month}\n"
          f"Peak-to-trough amplitude = {z.peak_to_trough_amplitude_million_km2:.2f} million km²")
    ax.text(.015,.975,note,transform=ax.transAxes,va="top",fontsize=9.5,
            bbox=dict(facecolor="white",edgecolor=".75",alpha=.93,boxstyle="round,pad=.35"))
    ax.legend(loc="upper right",frameon=False)
    fig.savefig(png,dpi=300,facecolor="white"); fig.savefig(pdf,facecolor="white"); plt.close(fig)

def main():
    r=root(); here=Path(__file__).resolve()
    source=r/"data"/"derived"/"RBEF_PWP28_daily_area_19810901_20260930.csv"; out=r/"outputs"/"RBEF"
    for z in ("figures","tables","reports"): (out/z).mkdir(parents=True,exist_ok=True)
    stem="RBEF06_canonical_pwp28_seasonality_v1p0p0"
    mcsv=out/"tables"/f"{stem}_monthly_climatology.csv"; dcsv=out/"tables"/f"{stem}_daily_climatology.csv"
    scsv=out/"tables"/f"{stem}_summary.csv"; png=out/"figures"/f"{stem}.png"; pdf=out/"figures"/f"{stem}.pdf"
    meta=out/"reports"/f"{stem}_metadata.json"
    print("="*78); print("RBEF06 | CANONICAL PWP28 SEASONALITY"); print("="*78)
    print(f"Project root : {r}\nInput        : {source}\nLoading and validating audited JPO04 input...")
    h=sha256(source); d=load(source); m,dc,s,complete,partial=analyse(d)
    m.to_csv(mcsv,index=False); dc.to_csv(dcsv,index=False); s.to_csv(scsv,index=False); figure(m,dc,s,png,pdf)
    if sha256(source)!=h: raise RuntimeError("Input changed during analysis")
    outs=[mcsv,dcsv,scsv,png,pdf]
    md={"program":here.name,"version":VERSION,"status":"PASS","generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope":{"geometry":"canonical","threshold_c":28.0,"lcc_used":False,"fourier_used":False,"wavelets_used":False},
        "input":{"path":str(source.resolve()),"sha256":h,"start":str(d.date.min().date()),"end":str(d.date.max().date()),"n_days":len(d)},
        "seasonality":{"complete_years_only":True,"first_year":min(complete),"last_year":max(complete),
        "n_years":len(complete),"partial_years_excluded":partial,"annual_level_removed_before_climatology":True,
        "feb29_excluded_from_daily_365_climatology":True,"interpolation":False},
        "results":s.iloc[0].to_dict(),"script_sha256":sha256(here),
        "output_sha256":{str(x.resolve()):sha256(x) for x in outs}}
    meta.write_text(json.dumps(md,indent=2,allow_nan=False),encoding="utf-8")
    print(f"\nComplete calendar years : {min(complete)}–{max(complete)} ({len(complete)} years)")
    print("Partial years excluded  : "+", ".join(map(str,partial)))
    print("\nMONTHLY SEASONAL CLIMATOLOGY")
    print(m[["month","month_name","climatological_mean_area_million_km2","seasonal_anomaly_million_km2","n_years"]].to_string(index=False))
    print("\nSEASONAL SUMMARY\n"+s.to_string(index=False)); print("\nOUTPUTS")
    for x in outs: print(" ",x)
    print(" ",meta); print(f"\nSTATUS: PASS | canonical PWP28 | {len(complete)} complete years used for seasonality"); print("="*78)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"STATUS: FAIL | {type(e).__name__}: {e}"); raise
