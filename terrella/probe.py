"""Tools for asking whether learned state dimensions correspond to anything physical."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from terrella import neural as N

QUIET_DST = 20.0


def driver_row(bz_gsm, by_gsm, v_sw, n_p, pressure, f107):
    """Physical values -> the model's driver feature vector, derived consistently.

    The derived terms must be recomputed from the perturbed field, not carried over, or
    the model gets a contradictory input during the impulse test.
    """
    bs = max(-bz_gsm, 0.0)
    b_t = np.hypot(by_gsm, bz_gsm)
    clock = np.arctan2(by_gsm, bz_gsm)
    feats = {
        "bz_gsm": bz_gsm, "by_gsm": by_gsm, "v_sw": v_sw, "n_p": n_p, "pressure": pressure,
        "bs": bs, "vbs": v_sw * bs * 1e-3, "sqrt_p": np.sqrt(pressure),
        "newell": (v_sw ** (4 / 3)) * (b_t ** (2 / 3)) * abs(np.sin(clock / 2)) ** (8 / 3),
        "sin_clock": np.sin(clock), "cos_clock": np.cos(clock), "f107": f107,
    }
    return np.array([feats[c] for c in N.DRIVERS], dtype=np.float32)


def quiet_baseline(segs):
    """Median driver conditions during quiet hours - derived from data, not assumed."""
    df = pd.concat(segs)
    q = df[df["dst"].abs() < QUIET_DST]
    med = {c: float(q[c].median())
           for c in ("bz_gsm", "by_gsm", "v_sw", "n_p", "pressure", "f107")}
    return med, float(q["dst"].median())


def efold(x, dt=1.0):
    """Hours for a decay curve to fall to 1/e of its peak deviation."""
    x = np.abs(x - x[-1])
    if x[0] <= 1e-9:
        return np.nan
    below = np.where(x < x[0] / np.e)[0]
    return float(below[0] * dt) if len(below) else np.nan


@torch.no_grad()
def impulse(model, st, base, base_dst, bz_step=-10.0, hold=6, tail=72):
    """Quiet -> step of southward Bz -> quiet. Returns latent trajectory and Dst."""
    quiet = driver_row(**base)
    storm = driver_row(**{**base, "bz_gsm": bz_step})
    seq = np.stack([quiet] * N.HISTORY + [storm] * hold + [quiet] * tail)

    x = torch.tensor((seq - st.xm) / st.xs).unsqueeze(0)
    y = torch.full((1, N.HISTORY), (base_dst - st.ym) / st.ys, dtype=torch.float32)

    traj = model.states(x[:, :N.HISTORY], y, x[:, N.HISTORY:])[0].numpy()
    dst = model(x[:, :N.HISTORY], y, x[:, N.HISTORY:])[0].numpy() * st.ys + st.ym
    return traj, dst


def r2(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())


def partial_r2(y, L, C):
    """Fraction of y's variance, after controls C, that the latent block L explains.

    Invariant to any invertible linear change of latent basis, so it survives the fact
    that latent dimensions carry no identity across training runs.
    """
    base = r2(y, C)
    if base >= 1.0:
        return float("nan")
    return (r2(y, np.column_stack([C, L])) - base) / (1.0 - base)


def partial_corr(x, y, z):
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
