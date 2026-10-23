#!/usr/bin/env python3
# GitHub distribution adaptation: input path/schema only; original analysis preserved.
# -*- coding: utf-8 -*-
"""RBEF05 — annual trend analysis, canonical PWP >= 28.0 C. Version 1.0.0.

Reads the audited JPO04 daily geometry table. Uses canonical PWP28 only.
Calendar-year means are calculated; only COMPLETE years are fitted.
OLS provides the descriptive linear trend and R². Theil–Sen provides a robust
slope comparison and its scipy 95% slope interval. No LCC, interpolation,
forecast, ENSO adjustment, or formal significance claim.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION="1.0.0"; THRESHOLD=28.0

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


def annual(d):
    x=d.assign(year=d.date.dt.year)
    a=x.groupby("year",as_index=False).agg(
        annual_mean_million_km2=("canonical_area_million_km2","mean"),
        annual_std_million_km2=("canonical_area_million_km2","std"),
        n_days=("canonical_area_million_km2","size"),
        first_date=("date","min"),last_date=("date","max"))
    a["expected_days"]=a.year.map(lambda y:366 if pd.Timestamp(int(y),12,31).is_leap_year else 365)
    a["complete_calendar_year"]=(a.n_days==a.expected_days)&(a.first_date.dt.dayofyear==1)&(
        a.last_date.dt.month.eq(12)&a.last_date.dt.day.eq(31))
    return a

def fit(a):
    f=a.loc[a.complete_calendar_year].copy().reset_index(drop=True)
    yr=f.year.to_numpy(float); y=f.annual_mean_million_km2.to_numpy(float); t=yr-yr[0]
    X=np.column_stack((np.ones(len(t)),t))
    ai,b=np.linalg.lstsq(X,y,rcond=None)[0]; yh=ai+b*t
    r2=1-float(np.sum((y-yh)**2))/float(np.sum((y-y.mean())**2))
    ts=theilslopes(y,t,alpha=.95)
    f["ols_fitted_million_km2"]=yh
    f["theil_sen_fitted_million_km2"]=ts.intercept+ts.slope*t
    span=t[-1]-t[0]
    s=pd.DataFrame([
      {"method":"OLS","first_year":int(yr[0]),"last_year":int(yr[-1]),"n_years":len(f),
       "intercept_million_km2":ai,"slope_million_km2_per_year":b,
       "slope_million_km2_per_decade":10*b,"slope_95ci_low_per_decade":np.nan,
       "slope_95ci_high_per_decade":np.nan,"fitted_change_million_km2":b*span,"r_squared":r2},
      {"method":"Theil-Sen","first_year":int(yr[0]),"last_year":int(yr[-1]),"n_years":len(f),
       "intercept_million_km2":ts.intercept,"slope_million_km2_per_year":ts.slope,
       "slope_million_km2_per_decade":10*ts.slope,"slope_95ci_low_per_decade":10*ts.low_slope,
       "slope_95ci_high_per_decade":10*ts.high_slope,"fitted_change_million_km2":ts.slope*span,
       "r_squared":np.nan}])
    return f,s

def figure(a,f,s,png,pdf):
    o=s[s.method=="OLS"].iloc[0]; ts=s[s.method=="Theil-Sen"].iloc[0]
    c=a[a.complete_calendar_year]; p=a[~a.complete_calendar_year]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(12,5.8),layout="constrained")
    if len(p): ax.scatter(p.year,p.annual_mean_million_km2,s=36,facecolors="white",edgecolors=".55",
                          label="Partial-year mean (not fitted)",zorder=3)
    ax.plot(c.year,c.annual_mean_million_km2,color=".45",lw=.9,marker="o",ms=4.2,mfc="white",
            mec=".35",label="Annual mean (complete years)",zorder=3)
    ax.plot(f.year,f.ols_fitted_million_km2,color="#0072B2",lw=1.8,label="Linear trend (OLS)")
    ax.plot(f.year,f.theil_sen_fitted_million_km2,color="#D55E00",lw=1.8,ls="--",
            label="Robust trend (Theil–Sen)")
    ax.set_title("Canonical Pacific Warm Pool area — 28°C: annual trend",fontsize=14)
    ax.set_ylabel("Annual mean area (million km²)")
    ax.set_xlabel(f"Year | Trend fits use complete calendar years {int(o.first_year)}–{int(o.last_year)}")
    ax.set_xlim(a.year.min()-.4,a.year.max()+.4); ax.set_xticks(range(1982,int(a.year.max())+1,4))
    v=a.annual_mean_million_km2.to_numpy(); pad=.08*(v.max()-v.min()); ax.set_ylim(v.min()-pad,v.max()+pad)
    ax.grid(axis="y",color=".88",lw=.55)
    note=(f"OLS trend = {o.slope_million_km2_per_decade:+.2f} million km²/decade\n"
          f"$R^2$ = {o.r_squared:.3f}\n"
          f"Theil–Sen trend = {ts.slope_million_km2_per_decade:+.2f} million km²/decade\n"
          f"95% slope interval = [{ts.slope_95ci_low_per_decade:+.2f}, {ts.slope_95ci_high_per_decade:+.2f}]")
    ax.text(.015,.975,note,transform=ax.transAxes,va="top",fontsize=9.5,
            bbox=dict(facecolor="white",edgecolor=".75",alpha=.93,boxstyle="round,pad=.35"))
    ax.legend(loc="lower right",frameon=False)
    fig.savefig(png,dpi=300,facecolor="white"); fig.savefig(pdf,facecolor="white"); plt.close(fig)

def main():
    r=root(); here=Path(__file__).resolve()
    source=r/"data"/"derived"/"RBEF_PWP28_daily_area_19810901_20260930.csv"
    out=r/"outputs"/"RBEF"
    for z in ("figures","tables","reports"): (out/z).mkdir(parents=True,exist_ok=True)
    stem="RBEF05_canonical_pwp28_trend_v1p0p0"
    acsv=out/"tables"/f"{stem}_annual_means.csv"; scsv=out/"tables"/f"{stem}_trend_summary.csv"
    png=out/"figures"/f"{stem}.png"; pdf=out/"figures"/f"{stem}.pdf"; meta=out/"reports"/f"{stem}_metadata.json"
    print("="*78); print("RBEF05 | CANONICAL PWP28 ANNUAL TREND ANALYSIS"); print("="*78)
    print(f"Project root : {r}\nInput        : {source}\nLoading and validating audited JPO04 input...")
    h=sha256(source); d=load(source); a=annual(d); f,s=fit(a)
    a.to_csv(acsv,index=False,date_format="%Y-%m-%d"); s.to_csv(scsv,index=False); figure(a,f,s,png,pdf)
    if sha256(source)!=h: raise RuntimeError("Input changed during analysis")
    c=a[a.complete_calendar_year]; p=a[~a.complete_calendar_year]; outs=[acsv,scsv,png,pdf]
    md={"program":here.name,"version":VERSION,"status":"PASS","generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope":{"geometry":"canonical","threshold_c":28.0,"lcc_used":False},
        "input":{"path":str(source.resolve()),"sha256":h,"start":str(d.date.min().date()),
                 "end":str(d.date.max().date()),"n_days":len(d)},
        "annual_analysis":{"complete_years_only":True,"first_year":int(c.year.min()),"last_year":int(c.year.max()),
                           "n_years":len(c),"partial_years_excluded":[int(x) for x in p.year]},
        "methods":{"OLS":"descriptive trend on complete-year annual means",
                   "Theil-Sen":"robust slope comparison; scipy 95% slope interval",
                   "formal_significance_claim":False},
        "results":s.replace({np.nan:None}).to_dict("records"),
        "dependencies":{"numpy":np.__version__,"pandas":pd.__version__,
                        "matplotlib":matplotlib.__version__,"scipy":scipy.__version__},
        "script_sha256":sha256(here),"output_sha256":{str(x.resolve()):sha256(x) for x in outs}}
    meta.write_text(json.dumps(md,indent=2,allow_nan=False),encoding="utf-8")
    print(f"\nComplete calendar years : {int(c.year.min())}–{int(c.year.max())} ({len(c)} years)")
    print("Partial years excluded  : "+", ".join(str(int(x)) for x in p.year))
    print("\nTREND RESULTS\n"+s.to_string(index=False)); print("\nOUTPUTS")
    for x in outs: print(" ",x)
    print(" ",meta); print(f"\nSTATUS: PASS | canonical PWP28 | {len(c)} complete years used"); print("="*78)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"STATUS: FAIL | {type(e).__name__}: {e}"); raise
