"""Storm-weighted loss, and optionally the full 63-year record.

Unweighted MSE is dominated by the ~95% of hours that are quiet, so the model is rewarded
for hedging precisely on the rare violent events that are the only ones anyone cares about.
The diagnostic for that is not RMSE but *bias*: a hedging model predicts systematically
shallower than truth on deep storms.

Evaluation stays unweighted throughout so runs at different alpha remain comparable.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from terrella import baselines as B
from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned

torch.set_num_threads(1)


def metrics(pred, truth):
    p, t = pred[:, -1], truth[:, -1]
    out = {"rmse_all": float(np.sqrt(np.mean((p - t) ** 2)))}
    for name, thr in [("storm", -50), ("intense", -100), ("severe", -200)]:
        m = t < thr
        if m.sum() < 10:
            out[f"rmse_{name}"] = out[f"bias_{name}"] = float("nan")
            out[f"n_{name}"] = int(m.sum())
            continue
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))
        # positive bias = predicted shallower than reality = hedging
        out[f"bias_{name}"] = float(np.mean(p[m] - t[m]))
        out[f"n_{name}"] = int(m.sum())
    # how close does it get to the single deepest point in the test set
    i = int(np.argmin(t))
    out["deepest_truth"] = float(t[i])
    out["deepest_pred"] = float(p[i])
    out["deepest_undershoot"] = float(p[i] - t[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--era", default="1998-01-01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = D.build(era_start=args.era)
    W = {k: aligned(v, stride=6) for k, v in splits.items()}
    Wg = {k: v[0] for k, v in W.items()}
    st = N.Stats(Wg["train"][0], Wg["train"][1])

    model, val = N.train(Wg["train"], Wg["val"], args.dim, st, epochs=args.epochs,
                         seed=args.seed, alpha=args.alpha)
    pred = N.predict(model, Wg["test"], st)

    out = {"alpha": args.alpha, "seed": args.seed, "era": args.era, "dim": args.dim,
           "val_obj": float(val), "n_train_windows": int(Wg["train"][0].shape[0]),
           "train_years": round(sum(len(s) for s in splits["train"]) / 8766, 1),
           **metrics(pred, Wg["test"][3])}

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"a={args.alpha:<5g} s={args.seed} era={args.era[:4]}  "
          f"all {out['rmse_all']:6.2f}  intense {out['rmse_intense']:6.2f}  "
          f"bias {out['bias_intense']:+6.2f}  deepest {out['deepest_pred']:7.1f}"
          f" vs {out['deepest_truth']:.0f}")


if __name__ == "__main__":
    main()
