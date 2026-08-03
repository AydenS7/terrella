"""Ask whether the learned state dimensions are physics or noise.

Three tests, in order of how hard they are to fake:

  1. Impulse response - hold the model at a quiet equilibrium, inject a step of southward
     Bz, and watch each dimension charge and decay. Gives a timescale per dimension. The
     ring current decays with tau ~ 7-20 h; the tail loads and unloads much faster.
  2. Unseen observables - the model is trained on Dst alone and never sees AE or Kp. If a
     dimension tracks AE, it found substorm activity with no supervision pointing at it.
  3. Ablation - freeze a dimension at its mean and see what breaks.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned

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
    """Median driver conditions during geomagnetically quiet hours - derived, not assumed."""
    df = pd.concat(segs)
    q = df[df["dst"].abs() < QUIET_DST]
    med = {c: float(q[c].median()) for c in ("bz_gsm", "by_gsm", "v_sw", "n_p", "pressure", "f107")}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = D.build()
    W = {k: aligned(v, stride=6) for k, v in splits.items()}
    Wg = {k: v[0] for k, v in W.items()}
    st = N.Stats(Wg["train"][0], Wg["train"][1])

    print(f"training H={args.dim} ...")
    model, val = N.train(Wg["train"], Wg["val"], args.dim, st, epochs=args.epochs, seed=args.seed)
    print(f"  val {val:.2f} nT")
    torch.save({"state": model.state_dict(), "dim": args.dim}, f"checkpoints/h{args.dim}.pt")

    base, base_dst = quiet_baseline(splits["train"])
    print(f"\nquiet baseline (median, |Dst|<{QUIET_DST:.0f}): " +
          "  ".join(f"{k}={v:.2f}" for k, v in base.items()) + f"  dst={base_dst:.1f}")

    print(f"\n=== 1. impulse response: 6h of Bz = -10 nT from quiet ===")
    traj, dst = impulse(model, st, base, base_dst)
    peak_i = int(np.argmin(dst))
    print(f"  predicted Dst: quiet {dst[0]:6.1f} -> min {dst.min():6.1f} nT at +{peak_i}h")
    print(f"  {'dim':>4}  {'swing':>7}  {'peak@h':>7}  {'e-fold':>7}")
    order = np.argsort(-np.abs(traj[:, :] - traj[0]).max(0))
    for d in order:
        s = traj[:, d]
        swing = s.max() - s.min()
        pk = int(np.argmax(np.abs(s - s[0])))
        tau = efold(s[pk:])
        print(f"  {d:>4}  {swing:7.3f}  {pk:7d}  " + (f"{tau:7.1f}" if np.isfinite(tau) else "      -"))

    print("\n=== 2. correlation with observables the model never saw ===")
    seg = splits["test"]
    hd, hy, fd, fy = Wg["test"]
    x = torch.tensor((fd - st.xm) / st.xs)
    xh = torch.tensor((hd - st.xm) / st.xs)
    yh = torch.tensor((hy - st.ym) / st.ys)
    with torch.no_grad():
        S = model.states(xh, yh, x).numpy()          # (n, horizon, dim)

    # observables aligned to the same forward steps
    obs = {"ae": [], "kp": [], "dst": [], "vbs": []}
    for s in seg:
        s = N._prep(s)
        n = len(s) - N.HISTORY - fd.shape[1]
        if n <= 0:
            continue
        starts = np.arange(0, n, 6)
        fi = starts[:, None] + N.HISTORY + np.arange(fd.shape[1])[None, :]
        for k in obs:
            obs[k].append(s[k].to_numpy()[fi])
    obs = {k: np.concatenate(v).ravel() for k, v in obs.items()}
    flat = S.reshape(-1, S.shape[-1])

    print(f"  {'dim':>4} " + "".join(f"{k:>9}" for k in obs))
    for d in range(S.shape[-1]):
        row = []
        for k, v in obs.items():
            m = np.isfinite(v)
            row.append(np.corrcoef(flat[m, d], v[m])[0, 1])
        print(f"  {d:>4} " + "".join(f"{r:>9.3f}" for r in row))

    # AE/Kp/Dst are mutually correlated, so a dimension that merely tracks Dst shows an AE
    # correlation for free. Partial out Dst - what survives is not explained by Dst.
    m = np.isfinite(obs["ae"]) & np.isfinite(obs["dst"]) & np.isfinite(obs["kp"])
    r_ae_dst = np.corrcoef(obs["ae"][m], obs["dst"][m])[0, 1]
    r_kp_dst = np.corrcoef(obs["kp"][m], obs["dst"][m])[0, 1]
    print(f"\n  confound: corr(AE,Dst)={r_ae_dst:+.3f}   corr(Kp,Dst)={r_kp_dst:+.3f}")

    def partial(x, y, z):
        rxy = np.corrcoef(x, y)[0, 1]
        rxz = np.corrcoef(x, z)[0, 1]
        ryz = np.corrcoef(y, z)[0, 1]
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    print(f"  partial correlation, Dst held fixed:")
    print(f"  {'dim':>4} {'AE|Dst':>9} {'Kp|Dst':>9}")
    for d in range(S.shape[-1]):
        col = flat[m, d]
        print(f"  {d:>4} {partial(col, obs['ae'][m], obs['dst'][m]):>9.3f}"
              f" {partial(col, obs['kp'][m], obs['dst'][m]):>9.3f}")
    print("        AE and Kp are never shown to the model, in training or at inference.")

    print("\n=== 3. ablation: freeze one dimension at its mean ===")
    mu = torch.tensor(flat.mean(0), dtype=torch.float32)
    truth = fy[:, -1]

    @torch.no_grad()
    def rmse_with(frozen=None):
        _, h = model.enc(torch.cat([xh, yh.unsqueeze(-1)], -1))
        h = h.squeeze(0)
        out = None
        for t in range(x.shape[1]):
            h = model.cell(x[:, t], h)
            if frozen is not None:
                h = h.clone()
                h[:, frozen] = mu[frozen]
            out = model.dec(h).squeeze(-1)
        p = out.numpy() * st.ys + st.ym
        return float(np.sqrt(np.mean((p - truth) ** 2)))

    full = rmse_with()
    print(f"  {'dim':>4}  {'rmse':>7}  {'degradation':>12}")
    print(f"  {'none':>4}  {full:7.2f}")
    for d in range(S.shape[-1]):
        r = rmse_with(d)
        print(f"  {d:>4}  {r:7.2f}  {r/full - 1:+11.1%}")


if __name__ == "__main__":
    main()
