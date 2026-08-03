"""The control for the pre-conditioning counterfactual.

A linear dissipative system forgets its initial condition exponentially. With a decay
constant of order 10 h and a storm lasting 72 h, near-complete compensation is what any
such system does - it is not evidence that the magnetosphere in particular does it.

So run the identical experiment on Burton-OM. If Burton compensates just as completely,
the learned model has reproduced a generic property of driven dissipative systems and the
result says nothing specific. If they differ, the difference is the finding.
"""

import numpy as np

from terrella import baselines as B
from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned
from terrella.probe import quiet_baseline
from scripts.counterfactual import PRELOAD_BZ, COND_H, STORM_H, find_storms


def run(params, vbs_seq, sqp_seq, dst0):
    """Free-running Burton-OM over a driver sequence, initialized from dst0."""
    a, tau, b, c, ec = params
    s = dst0 - b * sqp_seq[0] + c
    out = []
    for h in range(len(vbs_seq)):
        d = vbs_seq[h]
        q = a * max(d - ec, 0.0)
        s = s + (q - s / B.tau_om(np.array([d]), tau)[0]) * B.DT
        out.append(s + b * sqp_seq[h + 1] - c)
    return np.array(out)


def main():
    splits = D.build()
    W = {k: aligned(v, stride=6) for k, v in splits.items()}
    p, _ = B.fit(*W["train"][1], tau_fn=B.tau_om)
    print(f"Burton-OM fitted: a={p[0]:.3f} b={p[2]:.3f} c={p[3]:.2f} Ec={p[4]:.3f}\n")

    base, base_dst = quiet_baseline(splits["train"])
    storms = find_storms([s for v in splits.values() for s in v])

    # the same storms, but carrying the two channels Burton needs
    vbs_i, sqp_i = N.DRIVERS.index("vbs"), N.DRIVERS.index("sqrt_p")
    quiet_vbs = base["v_sw"] * max(-base["bz_gsm"], 0.0) * 1e-3
    quiet_sqp = np.sqrt(base["pressure"])

    rows = {}
    for bz in PRELOAD_BZ:
        cond_vbs = base["v_sw"] * max(-bz, 0.0) * 1e-3
        base_l, min_l = [], []
        for sm in storms:
            v = np.concatenate([[quiet_vbs] * N.HISTORY, [cond_vbs] * COND_H,
                                sm["drivers"][:, vbs_i]])
            q = np.concatenate([[quiet_sqp] * N.HISTORY, [quiet_sqp] * COND_H,
                                sm["drivers"][:, sqp_i], [quiet_sqp]])
            dst = run(p, v, q, base_dst)
            base_l.append(dst[N.HISTORY + COND_H - 1])
            min_l.append(dst[N.HISTORY + COND_H:].min())
        rows[bz] = {"baseline": float(np.mean(base_l)), "minimum": float(np.mean(min_l)),
                    "incremental": float(np.mean(np.array(min_l) - np.array(base_l)))}

    ctrl = rows[0.0]
    print(f"{'preload':>8} {'baseline':>10} {'abs min':>10} {'abs shift':>11} {'storm own effect':>18}")
    print(f"{0.0:>8.0f} {ctrl['baseline']:>10.1f} {ctrl['minimum']:>10.1f} "
          f"{'— control —':>11} {ctrl['incremental']:>18.1f}")
    for bz in PRELOAD_BZ[1:]:
        r = rows[bz]
        print(f"{bz:>8.0f} {r['baseline']:>10.1f} {r['minimum']:>10.1f} "
              f"{r['minimum'] - ctrl['minimum']:>+11.1f} "
              f"{r['incremental'] - ctrl['incremental']:>+18.1f}")
    print("\npositive storm-own-effect = the storm adds LESS when the system is already loaded")


if __name__ == "__main__":
    main()
