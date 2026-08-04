"""Does the magnetosphere's prior condition actually change how deep the next storm goes?

The whole project rests on the premise that the system stores energy invisibly, so that the
same solar wind can produce different storms depending on how loaded it already was. The
counterfactual tested a version of this inside the model and found no effect. This tests it
on the real record, with no model involved at all.

For each real storm: summarise the storm's own drivers, and separately summarise how
disturbed the magnetosphere was *before* it arrived. Then ask whether the prior state
explains any variance in depth that the drivers do not.

Two targets, because they answer different questions:
  absolute depth      - what matters for damage
  incremental depth   - depth minus starting level, i.e. what the storm itself added

If prior state adds nothing to either, the premise is in trouble.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from terrella import data as D

PRE_H, WIN_H, LOOKBACK = 12, 72, 24


def build(segs, thresh=-50.0):
    rows = []
    for s in segs:
        dst = s["dst"].to_numpy()
        vbs_all = s["vbs"].to_numpy()
        below = dst < thresh
        if not below.any():
            continue
        idx = np.flatnonzero(below)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 48) + 1):
            lo = int(run[0]) - PRE_H
            if lo - LOOKBACK < 0 or lo + WIN_H > len(s):
                continue
            w = s.iloc[lo:lo + WIN_H]
            bz, vbs, v, p = (w[c].to_numpy() for c in ("bz_gsm", "vbs", "v_sw", "pressure"))
            prior_dst = dst[lo - LOOKBACK:lo]
            prior_vbs = vbs_all[lo - LOOKBACK:lo]
            rows.append({
                "label": str(s.index[int(run[0])].date()),
                "depth": float(dst[lo:lo + WIN_H].min()),
                "start_dst": float(dst[lo]),
                # --- the storm's own drivers ---
                "vbs_sum": float(np.nansum(vbs)), "vbs_max": float(np.nanmax(vbs)),
                "vbs_top6": float(np.sort(vbs)[-6:].sum()),
                "bz_min": float(np.nanmin(bz)), "hours_south": float((bz < -5).sum()),
                "v_max": float(np.nanmax(v)), "v_mean": float(np.nanmean(v)),
                "p_max": float(np.nanmax(p)),
                "newell_sum": float(np.nansum(w["newell"].to_numpy())),
                # --- how loaded the system already was ---
                "pre_dst_mean": float(np.nanmean(prior_dst)),
                "pre_dst_min": float(np.nanmin(prior_dst)),
                "pre_dst_slope": float(prior_dst[-1] - prior_dst[0]),
                "pre_vbs_sum": float(np.nansum(prior_vbs)),
                "pre_vbs_max": float(np.nanmax(prior_vbs)),
            })
    df = pd.DataFrame(rows)
    df["incremental"] = df["depth"] - df["start_dst"]
    return df


DRIVER_F = ["vbs_sum", "vbs_max", "vbs_top6", "bz_min", "hours_south",
            "v_max", "v_mean", "p_max", "newell_sum"]
STATE_F = ["pre_dst_mean", "pre_dst_min", "pre_dst_slope", "pre_vbs_sum", "pre_vbs_max"]


def cv_r2(X, y, seed=0):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                          random_state=seed).fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pred


def main():
    df = build([s for v in D.build().values() for s in v]).dropna()
    print(f"{len(df)} storms.  prior-state range: pre_dst_mean "
          f"{df.pre_dst_mean.min():.0f} to {df.pre_dst_mean.max():.0f} nT "
          f"(sd {df.pre_dst_mean.std():.1f})")
    print(f"  {(df.pre_dst_mean < -30).sum()} storms arrived at an already-disturbed "
          f"magnetosphere (pre-storm mean Dst < -30)")

    for target in ("depth", "incremental"):
        y = df[target].to_numpy()
        print(f"\n=== target: {target} (sd {y.std():.1f} nT) ===")
        rows = []
        for name, feats in [("drivers only", DRIVER_F),
                            ("prior state only", STATE_F),
                            ("drivers + prior state", DRIVER_F + STATE_F)]:
            r2s = [cv_r2(df[feats].to_numpy(), y, seed=s)[0] for s in range(5)]
            rows.append((name, float(np.mean(r2s)), float(np.std(r2s))))
            print(f"  {name:24} CV R^2 {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
        base, both = rows[0][1], rows[2][1]
        gain = (both - base) / (1 - base) if base < 1 else float("nan")
        sd_both = rows[2][2]
        print(f"  --> prior state explains {gain * 100:+.1f}% of what the drivers leave over"
              f"   (seed sd on the combined fit alone is ±{sd_both:.3f} R^2)")

        # simple, assumption-free version of the same question
        r = np.corrcoef(df["pre_dst_mean"], y)[0, 1]
        resid = y - cv_r2(df[DRIVER_F].to_numpy(), y)[1]
        rp = np.corrcoef(df["pre_dst_mean"], resid)[0, 1]
        print(f"  raw corr(pre-storm Dst, {target}) = {r:+.3f};  "
              f"after removing what drivers explain = {rp:+.3f}")


if __name__ == "__main__":
    main()
