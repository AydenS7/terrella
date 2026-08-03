"""Three hourly-model improvements, measured against the frozen v1-hourly tag.

  1. Dipole tilt / season. Equinox months carry 2.4x the intense-storm rate of solstice
     months in this record (Russell-McPherron), and v1 had no time input whatsoever.
  2. Longer encoder history. 24 h was arbitrary; recurrent streams repeat on the 27-day
     solar rotation and storm recovery runs for days.
  3. Seed ensembling. We train many seeds anyway; averaging them is free skill, and the
     spread across them is a first crude uncertainty estimate.

None of these are resolution-dependent, so whatever wins here carries into the 1-minute
model unchanged. The hourly model is the cheap testbed - 40 s a run instead of minutes.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned

torch.set_num_threads(1)


def metrics(pred, truth):
    p, t = pred[:, -1], truth[:, -1]
    out = {"rmse_all": float(np.sqrt(np.mean((p - t) ** 2)))}
    for name, thr in [("storm", -50), ("intense", -100)]:
        m = t < thr
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))
    i = int(np.argmin(t))
    out["deepest_truth"] = float(t[i])
    out["deepest_pred"] = float(p[i])
    out["deepest_undershoot"] = float(p[i] - t[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["base", "tilt"], required=True)
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drivers = N.DRIVERS if args.features == "base" else N.DRIVERS_TILT
    splits = D.build()
    W = {k: aligned(v, stride=6, history=args.history, drivers=drivers)[0]
         for k, v in splits.items()}
    st = N.Stats(W["train"][0], W["train"][1])

    preds, per_seed = [], []
    for seed in range(args.seeds):
        model, _ = N.train(W["train"], W["val"], args.dim, st, epochs=args.epochs,
                           seed=seed, alpha=args.alpha)
        p = N.predict(model, W["test"], st)
        preds.append(p)
        per_seed.append(metrics(p, W["test"][3]))

    P = np.stack(preds)
    ens = metrics(P.mean(0), W["test"][3])
    # spread across seeds at the final lead - a first, uncalibrated uncertainty signal
    spread = P[:, :, -1].std(0)
    err = np.abs(P.mean(0)[:, -1] - W["test"][3][:, -1])
    out = {
        "features": args.features, "history": args.history, "seeds": args.seeds,
        "n_features": int(W["train"][0].shape[-1]),
        "single": {k: [float(np.mean([s[k] for s in per_seed])),
                       float(np.std([s[k] for s in per_seed]))]
                   for k in per_seed[0]},
        "ensemble": ens,
        "spread_mean": float(spread.mean()),
        "spread_error_corr": float(np.corrcoef(spread, err)[0, 1]),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    s = out["single"]
    print(f"{args.features:>4} h{args.history:<3} nfeat={out['n_features']:>2} | "
          f"single all {s['rmse_all'][0]:5.2f}±{s['rmse_all'][1]:4.2f} "
          f"intense {s['rmse_intense'][0]:6.2f}±{s['rmse_intense'][1]:5.2f} "
          f"under {s['deepest_undershoot'][0]:5.1f} | "
          f"ENS all {ens['rmse_all']:5.2f} intense {ens['rmse_intense']:6.2f} "
          f"under {ens['deepest_undershoot']:5.1f} | "
          f"spread-err corr {out['spread_error_corr']:+.2f}")


if __name__ == "__main__":
    main()
