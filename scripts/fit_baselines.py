"""Build the modern-era dataset and fit the low-order baselines.

Burton is the N=1 case of the model we're building: one hidden state, one transition
rule, one exogenous driver. Its score is the floor the learned model has to clear, and
fitting it is also a check that the data pipeline is sane.
"""

import numpy as np
import pandas as pd

from terrella import baselines as B
from terrella import data as D

HORIZON = 24
pd.set_option("display.width", 140)


def main():
    print("=== pipeline sanity: derived VBs vs OMNI's own E field ===")
    raw = D.add_features(D.load_raw())
    chk = D.sanity_check(raw)
    print(f"  n={chk['n']:,}  corr={chk['corr']:.6f}  max|err|={chk['max_abs_err']:.4f} mV/m")
    assert chk["corr"] > 0.999, "sign or unit error in the coupling terms"

    splits = D.build()

    print("\n=== dataset composition (1998+, gap-free segments >=72h) ===")
    rows = []
    for name, segs in splits.items():
        dst = pd.concat([s["dst"] for s in segs])
        lo, hi = D.SPLITS[name]
        rows.append({
            "split": name, "span": f"{lo[:4]}-{hi[:4]}", "segments": len(segs),
            "hours": f"{sum(len(s) for s in segs):,}",
            "yrs": round(sum(len(s) for s in segs) / 8766, 1),
            "longest_d": round(max(len(s) for s in segs) / 24),
            "<-50": len(D.storm_events(dst, -50)),
            "<-100": len(D.storm_events(dst, -100)),
            "<-200": len(D.storm_events(dst, -200)),
            "min_dst": round(dst.min()),
        })
    print(pd.DataFrame(rows).set_index("split").to_string())

    print(f"\n=== fitting on train, {HORIZON}h free-running rollouts ===")
    W = {k: B.windows(v, HORIZON) for k, v in splits.items()}
    print(f"  windows: " + "  ".join(f"{k}={len(v[0]):,}" for k, v in W.items()))

    models = {}
    for label, tau_fn in [("burton", B.tau_const), ("burton-OM", B.tau_om)]:
        p, rmse = B.fit(*W["train"], tau_fn=tau_fn)
        models[label] = (p, tau_fn)
        a, tau, b, c, ec = p
        tau_s = f"tau={tau:6.2f}h" if tau_fn is B.tau_const else "tau=f(VBs)"
        print(f"  {label:10} a={a:7.3f}  {tau_s}  b={b:5.3f}  c={c:6.2f}  Ec={ec:5.3f}"
              f"   train RMSE {rmse:6.2f} nT")

    for split in ("val", "test"):
        print(f"\n=== {split.upper()} — RMSE (nT), free-running from one observation ===")
        VBS, SQP, DST = W[split]
        out = {"persistence": B.score(B.persistence(DST), DST)}
        for label, (p, tau_fn) in models.items():
            out[label] = B.score(B.rollout(p, VBS, SQP, DST, tau_fn), DST)

        tbl = pd.concat({k: v[["rmse", "storm<-50", "intense<-100"]] for k, v in out.items()},
                        axis=1).round(2)
        print(tbl.to_string())
        n = out["persistence"]
        print("  targets per lead: " +
              "  ".join(f"{L}h: {int(n.loc[L, 'n_storm<-50'])} storm / "
                        f"{int(n.loc[L, 'n_intense<-100'])} intense" for L in n.index))

    print("\n=== skill vs persistence at 24h (positive = better) ===")
    VBS, SQP, DST = W["test"]
    base = B.score(B.persistence(DST), DST).loc[HORIZON]
    for label, (p, tau_fn) in models.items():
        s = B.score(B.rollout(p, VBS, SQP, DST, tau_fn), DST).loc[HORIZON]
        print(f"  {label:10} all {1 - s['rmse']/base['rmse']:+.1%}"
              f"   storm {1 - s['storm<-50']/base['storm<-50']:+.1%}"
              f"   intense {1 - s['intense<-100']/base['intense<-100']:+.1%}")


if __name__ == "__main__":
    main()
