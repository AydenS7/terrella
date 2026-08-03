"""Aggregate seed runs into a stability report.

Only basis-invariant quantities are compared across seeds. Per-dimension numbers are
reported as sorted spectra, which survive permutation of the latent basis but not
rotation, so they are indicative rather than conclusive.
"""

import glob
import json

import numpy as np


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def band(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return f"{v.mean():7.3f} ± {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]" if len(v) else "  n/a"


def main():
    trained = [r for r in load("data/seeds/h4_s*.json") if not r.get("untrained")]
    null = [r for r in load("data/seeds/null_h4_s*.json") if r.get("untrained")]
    print(f"trained seeds: {len(trained)}    untrained null seeds: {len(null)}\n")

    print("=== skill (fully basis-invariant) ===")
    for k in ("val_rmse", "test_all", "test_storm", "test_intense"):
        print(f"  {k:14} {band([r[k] for r in trained])}")

    print("\n=== impulse response: 6h of Bz = -10 nT from quiet ===")
    for k in ("impulse_quiet_dst", "impulse_min_dst", "impulse_peak_h"):
        print(f"  {k:18} {band([r[k] for r in trained])}")
    taus = [t for r in trained for t in r["efold_sorted"]]
    nd = [r["n_nondecaying"] for r in trained]
    print(f"  e-fold decay (all dims, all seeds, n={len(taus)}): {band(taus)}")
    print(f"  non-decaying dims per seed: {band(nd)}")

    print("\n=== partial R^2 of unseen observables on the latent state ===")
    print("  (fraction of variance left by the controls that the latent explains)")
    print(f"  {'target / control':34} {'trained':>34}   {'untrained null':>22}")
    for target in ("ae", "kp"):
        for c, label in [("A_dst", "A: Dst"),
                         ("B_dst_drivers", "B: + instantaneous drivers"),
                         ("C_dst_drivers_lagged", "C: + lagged VBs (3/6/12/24h)")]:
            key = f"pr2_{target}_{c}"
            t = band([r[key] for r in trained])
            n = band([r[key] for r in null]) if null else "   not run"
            sig = ""
            if null:
                a = np.array([r[key] for r in trained])
                b = np.array([r[key] for r in null])
                se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                sig = f"   t={(a.mean() - b.mean()) / se:+5.2f}" if se > 0 else ""
            print(f"  {target.upper():4} {label:29} {t}   {n}{sig}")

    print("\n=== ablation: freeze one dimension, sorted by impact ===")
    A = np.array([r["ablation_sorted"] for r in trained])
    for i in range(A.shape[1]):
        print(f"  rank {i+1}  {band(A[:, i])}")
    print(f"  smallest-impact dimension across seeds: "
          f"min {A[:, -1].min():.3f}  (a dead dimension would sit near 0)")


if __name__ == "__main__":
    main()
