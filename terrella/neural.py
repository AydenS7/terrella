"""Deterministic learned state-space model, swept over state dimension.

Same interface as the Burton rollout: encode a short history into a hidden state, then
run free on drivers alone with no further observations. A GRU with hidden size H is a
learned state-space model with H dimensions, so sweeping H traces the dimensionality
curve on which Burton is the H=1 point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DRIVERS = ["bz_gsm", "by_gsm", "v_sw", "n_p", "pressure", "vbs", "newell",
           "sqrt_p", "sin_clock", "cos_clock", "f107"]
HISTORY = 24


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sin_clock"] = np.sin(df["clock"])
    df["cos_clock"] = np.cos(df["clock"])
    return df


def windows(segs: list[pd.DataFrame], horizon: int, stride: int = 3):
    """(hist_drivers, hist_dst, fwd_drivers, fwd_dst) for every valid start."""
    hd, hy, fd, fy = [], [], [], []
    for s in segs:
        s = _prep(s)
        X = s[DRIVERS].to_numpy(np.float32)
        y = s["dst"].to_numpy(np.float32)
        n = len(s) - HISTORY - horizon
        if n <= 0:
            continue
        st = np.arange(0, n, stride)
        hi = st[:, None] + np.arange(HISTORY)[None, :]
        fi = st[:, None] + HISTORY + np.arange(horizon)[None, :]
        hd.append(X[hi]); hy.append(y[hi]); fd.append(X[fi]); fy.append(y[fi])
    return tuple(np.concatenate(a) for a in (hd, hy, fd, fy))


def aligned_windows(segs, horizon: int = 24, stride: int = 6):
    """GRU windows and Burton windows over identical target steps, so the two models are
    scored on exactly the same predictions. Burton starts one step earlier because it is
    initialized from a single observation rather than an encoded history."""
    acc = {k: [] for k in ("hd", "hy", "fd", "fy", "vbs", "sqp", "dst")}
    for s in segs:
        s = _prep(s)
        X = np.nan_to_num(s[DRIVERS].to_numpy(np.float32))
        y = s["dst"].to_numpy(np.float32)
        v, q, d = (s[c].to_numpy() for c in ("vbs", "sqrt_p", "dst"))
        n = len(s) - HISTORY - horizon
        if n <= 0:
            continue
        st = np.arange(0, n, stride)
        hi = st[:, None] + np.arange(HISTORY)[None, :]
        fi = st[:, None] + HISTORY + np.arange(horizon)[None, :]
        bi = st[:, None] + HISTORY - 1 + np.arange(horizon + 1)[None, :]
        for k, a in [("hd", X[hi]), ("hy", y[hi]), ("fd", X[fi]), ("fy", y[fi]),
                     ("vbs", v[bi][:, :horizon]), ("sqp", q[bi]), ("dst", d[bi])]:
            acc[k].append(a)
    c = {k: np.concatenate(v) for k, v in acc.items()}
    return ((c["hd"], c["hy"], c["fd"], c["fy"]), (c["vbs"], c["sqp"], c["dst"]))


class Stats:
    def __init__(self, hd, hy):
        flat = hd.reshape(-1, hd.shape[-1])
        self.xm, self.xs = flat.mean(0), flat.std(0) + 1e-6
        self.ym, self.ys = float(hy.mean()), float(hy.std()) + 1e-6


class StateModel(nn.Module):
    def __init__(self, n_drivers: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.GRU(n_drivers + 1, hidden, batch_first=True)
        self.cell = nn.GRUCell(n_drivers, hidden)
        self.dec = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, hist_x, hist_y, fwd_x):
        _, h = self.enc(torch.cat([hist_x, hist_y.unsqueeze(-1)], -1))
        h = h.squeeze(0)
        out = []
        for t in range(fwd_x.shape[1]):
            h = self.cell(fwd_x[:, t], h)
            out.append(self.dec(h).squeeze(-1))
        return torch.stack(out, 1)

    def states(self, hist_x, hist_y, fwd_x):
        """Latent trajectory, for probing what the dimensions actually do."""
        _, h = self.enc(torch.cat([hist_x, hist_y.unsqueeze(-1)], -1))
        h = h.squeeze(0)
        traj = []
        for t in range(fwd_x.shape[1]):
            h = self.cell(fwd_x[:, t], h)
            traj.append(h)
        return torch.stack(traj, 1)


def _tensors(W, st: Stats, device):
    hd, hy, fd, fy = W
    return (torch.tensor((hd - st.xm) / st.xs, device=device),
            torch.tensor((hy - st.ym) / st.ys, device=device),
            torch.tensor((fd - st.xm) / st.xs, device=device),
            torch.tensor((fy - st.ym) / st.ys, device=device))


def train(Wtr, Wva, hidden: int, st: Stats, device="cpu", epochs=30,
          batch=256, lr=3e-3, seed=0, verbose=False):
    torch.manual_seed(seed)
    tr, va = _tensors(Wtr, st, device), _tensors(Wva, st, device)
    model = StateModel(len(DRIVERS), hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = tr[0].shape[0]
    best, best_state = np.inf, None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(tr[0][j], tr[1][j], tr[2][j]), tr[3][j])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(model(va[0], va[1], va[2]), va[3]))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and ep % 5 == 0:
            print(f"    ep {ep:3d}  val {np.sqrt(vl) * st.ys:6.2f} nT")

    model.load_state_dict(best_state)
    return model, np.sqrt(best) * st.ys


@torch.no_grad()
def predict(model, W, st: Stats, device="cpu") -> np.ndarray:
    model.eval()
    t = _tensors(W, st, device)
    return model(t[0], t[1], t[2]).cpu().numpy() * st.ys + st.ym
