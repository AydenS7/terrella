"""Sweep the state dimension and compare against Burton on identical windows.

Burton is the H=1 point of this curve. Where the curve flattens is the estimate of how
many degrees of freedom the magnetosphere's observable dynamics actually need.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from terrella import baselines as B
from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned

HORIZON = 24


def rmse(pred, truth, mask=None):
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def report(name, pred, truth, lead=HORIZON):
    p, t = pred[:, lead - 1], truth[:, lead - 1]
    return {"model": name, "all": rmse(p, t),
            "storm<-50": rmse(p, t, t < -50), "intense<-100": rmse(p, t, t < -100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    splits = D.build()
    W = {k: aligned(v, stride=args.stride) for k, v in splits.items()}
    Wg = {k: v[0] for k, v in W.items()}
    Wb = {k: v[1] for k, v in W.items()}
    print(f"windows  " + "  ".join(f"{k}={len(v[0]):,}" for k, v in Wg.items()))

    # Burton on the identical windows, refit here so the comparison is exact.
    p_om, _ = B.fit(*Wb["train"], tau_fn=B.tau_om)
    rows = []
    for split in ("val", "test"):
        vbs, sqp, dst = Wb[split]
        rows.append({"split": split, "params": 5,
                     **report("burton-OM (H=1)", B.rollout(p_om, vbs, sqp, dst, B.tau_om), dst[:, 1:])})
        rows.append({"split": split, "params": 0,
                     **report("persistence", B.persistence(dst), dst[:, 1:])})

    st = N.Stats(Wg["train"][0], Wg["train"][1])
    for h in args.dims:
        for seed in range(args.seeds):
            t0 = time.time()
            model, val_rmse = N.train(Wg["train"], Wg["val"], h, st,
                                      device=args.device, epochs=args.epochs, seed=seed)
            npar = sum(p.numel() for p in model.parameters())
            for split in ("val", "test"):
                pred = N.predict(model, Wg[split], st, args.device)
                rows.append({"split": split, "params": npar, "seed": seed,
                             **report(f"gru H={h}", pred, Wg[split][3])})
            print(f"  H={h:3d} seed={seed}  {npar:6,} params  "
                  f"val {val_rmse:5.2f} nT  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    agg = (df.groupby(["split", "model", "params"], as_index=False)
             [["all", "storm<-50", "intense<-100"]].mean())
    for split in ("val", "test"):
        print(f"\n=== {split.upper()} — RMSE (nT) at {HORIZON}h lead, free-running ===")
        sub = agg[agg.split == split].drop(columns="split").set_index("model")
        order = ["persistence", "burton-OM (H=1)"] + [f"gru H={h}" for h in args.dims]
        print(sub.reindex([o for o in order if o in sub.index]).round(2).to_string())

    df.to_csv("data/sweep_results.csv", index=False)
    print("\nwrote data/sweep_results.csv")


if __name__ == "__main__":
    main()
