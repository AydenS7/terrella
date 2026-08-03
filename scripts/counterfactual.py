"""Does a storm hit harder when the magnetosphere is already loaded?

You cannot answer this observationally. There are a handful of sequential-CME events, no
controls, and no way to repeat them - you get one Earth and one history. A simulator with
an intervenable state is the only apparatus that can run the experiment.

Design. Both arms are structurally identical and differ only in the state the storm lands
on:

    [ 24 h encoder history, quiet ][ 24 h conditioning block ][ real storm drivers ]
                                     quiet  vs  Bz = -3/-5/-8 nT

Measured per arm:
    baseline     predicted Dst at the last step of the conditioning block
    minimum      deepest predicted Dst during the storm block
    incremental  minimum - baseline, i.e. what the storm itself added

The absolute minimum matters for damage; the incremental drop is the physics question.
Reporting only the absolute would confuse "started lower" with "hit harder".
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned
from terrella.probe import driver_row, quiet_baseline

torch.set_num_threads(1)

PRELOAD_BZ = [0.0, -3.0, -5.0, -8.0]   # 0 is the control arm
COND_H = 24
STORM_H = 72
VIZ_STORM = "2024-05-10"               # Gannon


def find_storms(segs, thresh=-100.0, pre_h=12, n_max=24):
    """Deepest distinct storms with complete drivers, as (label, drivers, observed Dst)."""
    found = []
    for s in segs:
        s = N._prep(s)
        dst = s["dst"].to_numpy()
        X = np.nan_to_num(s[N.DRIVERS].to_numpy(np.float32))
        below = dst < thresh
        if not below.any():
            continue
        runs = np.split(np.flatnonzero(below), np.flatnonzero(np.diff(np.flatnonzero(below)) > 48) + 1)
        for r in runs:
            if len(r) == 0:
                continue
            lo = int(r[0]) - pre_h
            if lo < 0 or lo + STORM_H > len(s):
                continue
            found.append({
                "label": str(s.index[int(r[0])].date()),
                "depth": float(dst[lo:lo + STORM_H].min()),
                "drivers": X[lo:lo + STORM_H],
                "observed": dst[lo:lo + STORM_H].tolist(),
            })
    found.sort(key=lambda d: d["depth"])
    return found[:n_max]


@torch.no_grad()
def run_arm(model, st, base, base_dst, storm_drivers, preload_bz):
    """Quiet history -> conditioning block -> the real storm. Returns Dst and latents."""
    quiet = driver_row(**base)
    cond = driver_row(**{**base, "bz_gsm": preload_bz}) if preload_bz else quiet
    hist = np.stack([quiet] * N.HISTORY)
    fwd = np.concatenate([np.stack([cond] * COND_H), storm_drivers])

    hx = torch.tensor(((hist - st.xm) / st.xs).astype(np.float32)).unsqueeze(0)
    hy = torch.full((1, N.HISTORY), (base_dst - st.ym) / st.ys, dtype=torch.float32)
    fx = torch.tensor(((fwd - st.xm) / st.xs).astype(np.float32)).unsqueeze(0)

    dst = model(hx, hy, fx)[0].numpy() * st.ys + st.ym
    lat = model.states(hx, hy, fx)[0].numpy()
    return dst, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", help="also dump a rich single-storm trace for the visualizer")
    args = ap.parse_args()

    splits = D.build()
    W = {k: aligned(v, stride=6) for k, v in splits.items()}
    Wg = {k: v[0] for k, v in W.items()}
    st = N.Stats(Wg["train"][0], Wg["train"][1])
    model, val = N.train(Wg["train"], Wg["val"], args.dim, st, epochs=args.epochs, seed=args.seed)

    base, base_dst = quiet_baseline(splits["train"])
    storms = find_storms([s for v in splits.values() for s in v])

    out = {"seed": args.seed, "dim": args.dim, "val_rmse": float(val),
           "n_storms": len(storms), "preload_bz": PRELOAD_BZ, "arms": {}}

    for bz in PRELOAD_BZ:
        rows = []
        for sm in storms:
            dst, _ = run_arm(model, st, base, base_dst, sm["drivers"], bz)
            baseline = float(dst[COND_H - 1])
            minimum = float(dst[COND_H:].min())
            rows.append({"label": sm["label"], "baseline": baseline, "minimum": minimum,
                         "incremental": minimum - baseline})
        out["arms"][str(bz)] = {
            "baseline_mean": float(np.mean([r["baseline"] for r in rows])),
            "minimum_mean": float(np.mean([r["minimum"] for r in rows])),
            "incremental_mean": float(np.mean([r["incremental"] for r in rows])),
            "per_storm": rows,
        }

    ctrl = out["arms"]["0.0"]
    for bz in PRELOAD_BZ[1:]:
        a = out["arms"][str(bz)]
        out[f"effect_abs_{bz}"] = a["minimum_mean"] - ctrl["minimum_mean"]
        out[f"effect_incremental_{bz}"] = a["incremental_mean"] - ctrl["incremental_mean"]

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    if args.viz:
        pick = next((s for s in storms if s["label"].startswith(VIZ_STORM[:7])), storms[0])
        traces = {}
        for bz in PRELOAD_BZ:
            dst, lat = run_arm(model, st, base, base_dst, pick["drivers"], bz)
            traces[str(bz)] = {"dst": dst.round(2).tolist(), "latent": lat.round(4).tolist()}
        bz_i, v_i, p_i = (N.DRIVERS.index(c) for c in ("bz_gsm", "v_sw", "pressure"))
        quiet_row = driver_row(**base)
        pathlib.Path(args.viz).write_text(json.dumps({
            "storm": pick["label"], "depth": pick["depth"],
            "cond_h": COND_H, "storm_h": STORM_H, "dim": args.dim,
            "preload_bz": PRELOAD_BZ,
            "observed": [None] * COND_H + [round(x, 1) for x in pick["observed"]],
            "bz": [float(quiet_row[bz_i])] * COND_H + pick["drivers"][:, bz_i].round(2).tolist(),
            "v": [float(quiet_row[v_i])] * COND_H + pick["drivers"][:, v_i].round(1).tolist(),
            "pressure": [float(quiet_row[p_i])] * COND_H + pick["drivers"][:, p_i].round(2).tolist(),
            "traces": traces,
        }))

    print(f"seed {args.seed}: {len(storms)} storms  "
          + "  ".join(f"Bz{bz:+.0f} abs {out[f'effect_abs_{bz}']:+6.1f} "
                      f"incr {out[f'effect_incremental_{bz}']:+6.1f}" for bz in PRELOAD_BZ[1:]))


if __name__ == "__main__":
    main()
