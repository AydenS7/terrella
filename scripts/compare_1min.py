"""What the 1-minute record actually buys, measured rather than assumed.

Three questions before any model is retrained:

  1. How do SYM/H and Dst relate? They are different indices from different station sets,
     so v1 and v2 numbers are not comparable until the offset is measured, not assumed.
  2. How much do the nonlinear coupling terms change when computed at 1-minute cadence and
     then averaged, versus computed from hourly averages? mean(VBs) != VBs(mean Bz).
  3. Does coverage hold up at the finer cadence?
"""

import numpy as np
import pandas as pd

from terrella import data as D
from terrella import data1min as D1


def main():
    print("loading 1-minute record aggregated to 60min steps ...")
    m = D1.load_all("60min")
    h = D.add_features(D.load_raw())
    # the hourly product labels each hour at its half-hour midpoint (00:30), while a 60min
    # resample labels at the hour start (00:00) - they describe the same interval
    h = h.set_index(h.index.floor("h"))
    j = m.join(h[["dst", "bz_gsm", "v_sw", "vbs", "newell"]].rename(
        columns=lambda c: c + "_hr"), how="inner")
    print(f"  1-min steps: {len(m):,}   overlap with hourly: {len(j):,}"
          f"   {j.index[0].date()} -> {j.index[-1].date()}")

    print("\n=== 1. SYM/H vs Dst ===")
    k = j[["sym_h", "dst_hr"]].dropna()
    d = k["sym_h"] - k["dst_hr"]
    r = np.corrcoef(k["sym_h"], k["dst_hr"])[0, 1]
    slope, icept = np.polyfit(k["dst_hr"], k["sym_h"], 1)
    print(f"  n={len(k):,}  corr={r:.5f}   SYM/H = {slope:.4f}*Dst {icept:+.2f}")
    print(f"  mean(SYM/H - Dst) = {d.mean():+.2f} nT   sd {d.std():.2f}")
    for lo, hi, lab in [(-1e9, -200, "Dst < -200"), (-200, -100, "-200..-100"),
                        (-100, -50, "-100..-50"), (-50, 1e9, "Dst > -50")]:
        s = d[(k["dst_hr"] >= lo) & (k["dst_hr"] < hi)]
        if len(s) > 50:
            print(f"    {lab:12} n={len(s):>7,}  mean diff {s.mean():+7.2f} nT  sd {s.std():5.2f}")
    print("  (negative = SYM/H reads deeper than Dst)")

    print("\n=== 2. storm counts, same threshold, each index ===")
    for name, series in [("Dst (hourly)", k["dst_hr"]), ("SYM/H (1min->hourly)", k["sym_h"]),
                         ("SYM/H within-hour min", j["sym_h_min"].dropna())]:
        counts = [len(D.storm_events(series, t)) for t in (-50, -100, -200, -250)]
        print(f"  {name:24} <-50 {counts[0]:5d}  <-100 {counts[1]:4d}  "
              f"<-200 {counts[2]:3d}  <-250 {counts[3]:3d}   deepest {series.min():.0f}")

    print("\n=== 3. nonlinear coupling: 1-min-then-average vs average-then-compute ===")
    for col in ("vbs", "newell"):
        c = j[[col, col + "_hr"]].dropna()
        fine, coarse = c[col], c[col + "_hr"]
        rel = (fine - coarse) / coarse.replace(0, np.nan)
        big = c[coarse > coarse.quantile(0.99)]
        print(f"  {col:8} corr {np.corrcoef(fine, coarse)[0,1]:.4f}   "
              f"median rel. diff {rel.median()*100:+.1f}%   "
              f"top-1% driving: fine {big[col].mean():.3f} vs hourly {big[col+'_hr'].mean():.3f} "
              f"({(big[col].mean()/big[col+'_hr'].mean()-1)*100:+.1f}%)")

    print("\n=== 4. coverage ===")
    for lab, cols, frame in [("1-min -> 60min", D1.REQUIRED, m),
                             ("hourly product", D.REQUIRED, h[h.index >= m.index[0]])]:
        cov = frame[cols].notna().all(axis=1).mean() * 100
        print(f"  {lab:16} all-required present: {cov:5.1f}%")
    segs = D1.segments(m)
    print(f"  1-min segments >=72 steps: {len(segs)}   "
          f"usable steps {sum(len(s) for s in segs):,} "
          f"({sum(len(s) for s in segs)/8766:.1f} yr)   "
          f"longest {max(len(s) for s in segs)/24:.0f} d")


if __name__ == "__main__":
    main()
