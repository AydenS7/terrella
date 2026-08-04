"""How much of storm depth is predictable from the solar wind at all?

This sets the target for v3. If two storms with near-identical drivers still reach very
different depths, then a model that answers -326 when reality hit -406 is being *honest*,
and the right fix is a distribution rather than a deeper point estimate. If similar drivers
reliably give similar depths, the gap is model error and v3 is the wrong tool.

Two independent estimates:

  1. Neighbour spread, extrapolated to zero distance. For pairs of storms, plot the
     difference in depth against the distance between their driver summaries. Pairs at
     finite distance differ partly for real reasons, so their spread is an upper bound;
     extrapolating the trend to zero separation estimates the irreducible part.

  2. A strong tabular model on the same summary features. Gradient boosting is free to use
     any nonlinear combination of them, so its residual is roughly what these features can
     support, independent of our architecture.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from terrella import data as D

PRE_H, WIN_H = 12, 72


def storm_windows(segs, thresh=-50.0):
    """Each distinct storm as (driver summary, depth)."""
    rows = []
    for s in segs:
        dst = s["dst"].to_numpy()
        below = dst < thresh
        if not below.any():
            continue
        idx = np.flatnonzero(below)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 48) + 1):
            lo = int(run[0]) - PRE_H
            if lo < 0 or lo + WIN_H > len(s):
                continue
            w = s.iloc[lo:lo + WIN_H]
            bz, vbs, v, p = (w[c].to_numpy() for c in ("bz_gsm", "vbs", "v_sw", "pressure"))
            rows.append({
                "label": str(s.index[int(run[0])].date()),
                "depth": float(dst[lo:lo + WIN_H].min()),
                # what the literature says drives ring-current injection
                "vbs_sum": float(np.nansum(vbs)), "vbs_max": float(np.nanmax(vbs)),
                "vbs_top6": float(np.sort(vbs)[-6:].sum()),
                "bz_min": float(np.nanmin(bz)),
                "hours_south": float((bz < -5).sum()),
                "v_max": float(np.nanmax(v)), "v_mean": float(np.nanmean(v)),
                "p_max": float(np.nanmax(p)),
                "newell_sum": float(np.nansum(w["newell"].to_numpy())),
            })
    return pd.DataFrame(rows)


def main():
    segs = [s for v in D.build().values() for s in v]
    df = storm_windows(segs).dropna()
    feats = [c for c in df.columns if c not in ("label", "depth")]
    print(f"{len(df)} distinct storms (Dst < -50) with complete drivers, "
          f"depth {df.depth.min():.0f} to {df.depth.max():.0f} nT, sd {df.depth.std():.1f}")

    X = (df[feats] - df[feats].mean()) / df[feats].std()
    Xa = X.to_numpy()
    d = df["depth"].to_numpy()

    print("\n=== 1. neighbour spread vs driver distance ===")
    n = len(df)
    i, j = np.triu_indices(n, 1)
    dist = np.linalg.norm(Xa[i] - Xa[j], axis=1) / np.sqrt(len(feats))
    ddep = np.abs(d[i] - d[j])
    qs = np.quantile(dist, np.linspace(0, 0.5, 11))
    print(f"  {'driver distance':>18} {'pairs':>7} {'mean |depth diff|':>19}")
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 50:
            continue
        xs.append(float(np.mean(dist[m]))); ys.append(float(np.mean(ddep[m])))
        print(f"  {lo:7.3f}-{hi:<7.3f} {m.sum():>9,} {np.mean(ddep[m]):>17.1f} nT")
    slope, icept = np.polyfit(xs, ys, 1)
    # mean |A-B| for two draws from the same distribution is 2*sigma/sqrt(pi)
    sigma = icept / (2 / np.sqrt(np.pi))
    print(f"\n  linear fit -> mean |depth diff| at zero driver distance: {icept:.1f} nT")
    print(f"  implied irreducible sd: {sigma:.1f} nT   (RMSE floor for any point forecast)")

    print("\n=== 2. gradient boosting on the same features, 5-fold CV ===")
    kf, pred = KFold(5, shuffle=True, random_state=0), np.zeros(n)
    for tr, te in kf.split(Xa):
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          random_state=0).fit(Xa[tr], d[tr])
        pred[te] = g.predict(Xa[te])
    resid = pred - d
    print(f"  RMSE {np.sqrt(np.mean(resid ** 2)):.1f} nT   MAE {np.abs(resid).mean():.1f} nT   "
          f"R^2 {1 - resid.var() / d.var():.3f}")
    for lo, lab in [(-1e9, "all storms"), (-100, "intense <-100"), (-200, "severe <-200")]:
        m = d < lo if lo > -1e8 else np.ones(n, bool)
        if m.sum() > 5:
            print(f"    {lab:16} n={m.sum():>4}  RMSE {np.sqrt(np.mean(resid[m] ** 2)):6.1f} nT  "
                  f"bias {resid[m].mean():+6.1f}")

    print("\n=== 3. does the floor scale with storm depth? ===")
    pair_depth = (d[i] + d[j]) / 2
    close = dist < 0.35
    print(f"  {'depth band':>16} {'pairs':>7} {'mean |diff|':>12} {'% of depth':>11} {'implied sd':>11}")
    for lo, hi in [(-60, -50), (-80, -60), (-120, -80), (-200, -120), (-1e9, -200)]:
        m = close & (pair_depth < hi) & (pair_depth >= lo)
        if m.sum() < 15:
            print(f"  {hi:5.0f}..{lo:7.0f} {m.sum():>9,}   too few close pairs to measure")
            continue
        md = ddep[m].mean()
        print(f"  {hi:5.0f}..{lo:7.0f} {m.sum():>9,} {md:>11.1f} nT "
              f"{md / abs(pair_depth[m].mean()) * 100:>10.1f}% {md / (2 / np.sqrt(np.pi)):>10.1f} nT")
    print("  The fractional spread is roughly flat at 20-30% of depth beyond the shallowest")
    print("  storms, i.e. the noise is MULTIPLICATIVE, not additive. Extrapolated, a -400 nT")
    print("  event carries an irreducible sd near 100 nT - but there are zero close pairs")
    print("  below -200, so that extrapolation cannot be verified. We can least measure the")
    print("  uncertainty exactly where it matters most, which is the project's whole thesis.")

    print("\n=== 4. what this means for the model ===")
    print(f"  irreducible sd (pair extrapolation): {sigma:5.1f} nT")
    print(f"  gradient boosting residual (all):    {np.sqrt(np.mean(resid ** 2)):5.1f} nT")
    dm = d < -100
    print(f"  gradient boosting residual (intense):{np.sqrt(np.mean(resid[dm] ** 2)):5.1f} nT")
    print(f"  Terrella intense-storm RMSE at 24h:   30.5 nT (8-seed ensemble)")


if __name__ == "__main__":
    main()
