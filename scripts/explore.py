"""Go/no-go survey of the OMNI hourly record: coverage + storm event counts."""

import numpy as np
import pandas as pd

COLS = ["time", "bz_gsm", "n_p", "v_sw", "pressure", "e_field", "f107", "kp10", "dst", "ae"]
FILL = {
    "bz_gsm": 999.9, "n_p": 999.9, "v_sw": 9999.0, "pressure": 99.99,
    "e_field": 999.99, "f107": 999.9, "kp10": 99, "dst": 99999, "ae": 9999,
}

df = pd.read_csv("data/raw/omni_hourly.csv", names=COLS, header=None)
df["time"] = pd.to_datetime(df["time"], format="ISO8601")
for c, f in FILL.items():
    df.loc[df[c] >= f, c] = np.nan
df = df.set_index("time")

print(f"rows {len(df):,}   {df.index[0].date()} -> {df.index[-1].date()}")

print("\n=== channel coverage (% non-fill) ===")
era = df.assign(era=pd.cut(df.index.year, [1962, 1980, 1997, 2010, 2027],
                           labels=["63-80", "81-97", "98-10", "11-26"]))
cov = era.groupby("era", observed=True).apply(
    lambda g: (g[list(FILL)].notna().mean() * 100).round(1), include_groups=False)
print(cov.to_string())

print("\n=== storm EVENTS by Dst minimum (>=48h separation) ===")
dst = df["dst"]


def events(thresh, gap_h=48):
    below = dst < thresh
    grp = (below != below.shift()).cumsum()[below]
    runs = dst[below].groupby(grp)
    starts, mins = [], []
    for _, r in runs:
        if starts and (r.index[0] - starts[-1]).total_seconds() / 3600 < gap_h:
            mins[-1] = min(mins[-1], r.min())
        else:
            starts.append(r.index[-1])
            mins.append(r.min())
        starts[-1] = r.index[-1]
    return mins


for label, t in [("moderate  <-50", -50), ("intense   <-100", -100),
                 ("severe    <-200", -200), ("great     <-250", -250),
                 ("extreme   <-300", -300), ("           <-400", -400)]:
    m = events(t)
    print(f"{label:18} {len(m):5d} events   {len(m)/63.5:5.1f}/yr")

print("\n=== 10 deepest storms on record ===")
d = dst.dropna()
top = d.nsmallest(2000).sort_index()
seen = []
for ts, v in top.items():
    if not seen or (ts - seen[-1][0]).days > 5:
        seen.append((ts, v))
    elif v < seen[-1][1]:
        seen[-1] = (ts, v)
for ts, v in sorted(seen, key=lambda x: x[1])[:10]:
    print(f"  {ts.date()}  Dst = {v:.0f} nT")

print("\n=== driver availability during intense storms (Dst<-100) ===")
mask = dst < -100
print((df.loc[mask, ["bz_gsm", "v_sw", "n_p", "e_field"]].notna().mean() * 100).round(1).to_string())

print("\n=== longest continuous gap-free stretch (bz, v, dst all present) ===")
ok = df[["bz_gsm", "v_sw", "dst"]].notna().all(axis=1)
grp = (~ok).cumsum()[ok]
runs = ok.groupby(grp).size()
print(f"  longest {runs.max():,} h ({runs.max()/24:.0f} d)   "
      f"stretches >30d: {(runs > 720).sum()}   total usable {ok.sum():,} h ({ok.mean()*100:.1f}%)")
