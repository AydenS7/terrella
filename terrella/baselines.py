"""Low-order baselines: persistence, Burton (1975), O'Brien & McPherron (2000).

Everything is scored in free-running mode — the state is initialized from one observed
Dst and then integrated forward on drivers alone, with no further observations. That is
the same regime the learned model will be judged in, so the numbers are comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DT = 1.0  # hours

# O'Brien & McPherron (2000) values as the optimizer's starting point. Burton's original
# b=0.20 is in different pressure units; the modern nPa-based coefficient is ~7.26.
BURTON_X0 = np.array([-4.4, 7.7, 7.26, 11.0, 0.49])  # a, tau, b, c, Ec
BOUNDS = [(-40.0, 0.0), (1.0, 100.0), (0.0, 30.0), (-50.0, 100.0), (0.0, 5.0)]


def windows(segs: list[pd.DataFrame], horizon: int, stride: int = 6):
    """Stack fixed-length rollout windows from every segment into flat arrays."""
    vbs, sqp, dst = [], [], []
    for s in segs:
        n = len(s) - horizon
        if n <= 0:
            continue
        starts = np.arange(0, n, stride)
        idx = starts[:, None] + np.arange(horizon + 1)[None, :]
        vbs.append(s["vbs"].to_numpy()[idx][:, :horizon])
        sqp.append(s["sqrt_p"].to_numpy()[idx])
        dst.append(s["dst"].to_numpy()[idx])
    return (np.concatenate(vbs), np.concatenate(sqp), np.concatenate(dst))


def tau_const(vbs, tau):
    return np.full_like(vbs, tau)


def tau_om(vbs, _tau):
    """O'Brien & McPherron (2000): decay accelerates under strong driving."""
    return 2.40 * np.exp(9.74 / (4.69 + vbs))


def rollout(params, VBS, SQP, DST, tau_fn=tau_const):
    a, tau, b, c, ec = params
    s = DST[:, 0] - b * SQP[:, 0] + c
    out = np.empty((VBS.shape[0], VBS.shape[1]))
    for h in range(VBS.shape[1]):
        d = VBS[:, h]
        q = a * np.maximum(d - ec, 0.0)
        s = s + (q - s / tau_fn(d, tau)) * DT
        out[:, h] = s + b * SQP[:, h + 1] - c
    return out


def fit(VBS, SQP, DST, tau_fn=tau_const, x0=BURTON_X0):
    def loss(p):
        r = rollout(p, VBS, SQP, DST, tau_fn) - DST[:, 1:]
        return float(np.sqrt(np.mean(r ** 2)))

    res = minimize(loss, x0, method="L-BFGS-B", bounds=BOUNDS,
                   options={"maxiter": 400, "eps": 1e-4})
    return res.x, res.fun


def persistence(DST):
    return np.repeat(DST[:, :1], DST.shape[1] - 1, axis=1)


def score(pred, DST, leads=(1, 6, 12, 24)):
    """RMSE overall and restricted to storm-time targets, by lead hour."""
    truth = DST[:, 1:]
    rows = []
    for L in leads:
        if L > truth.shape[1]:
            continue
        p, t = pred[:, L - 1], truth[:, L - 1]
        row = {"lead_h": L, "rmse": np.sqrt(np.mean((p - t) ** 2))}
        for name, thr in [("storm<-50", -50), ("intense<-100", -100)]:
            m = t < thr
            row[name] = np.sqrt(np.mean((p[m] - t[m]) ** 2)) if m.sum() > 20 else np.nan
            row[f"n_{name}"] = int(m.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("lead_h")
