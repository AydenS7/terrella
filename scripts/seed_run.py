"""One seed of the stability check. Writes a JSON of basis-invariant metrics.

Latent dimensions carry no identity across seeds - the basis is arbitrary up to
permutation, sign and rotation - so per-dimension numbers cannot be compared run to run.
Everything reported here is either invariant to an invertible linear change of latent
basis (partial R^2, impulse response) or invariant to permutation (sorted spectra).

The AE claim also needs a second control. The latent is a function of recent drivers and
those drivers drive AE, so "carries AE information beyond Dst" could just mean "remembers
that VBs was recently high". Controls are graded:

    A  Dst
    B  Dst + instantaneous drivers
    C  Dst + instantaneous drivers + VBs averaged over the previous 3/6/12/24 h

Surviving C means the state integrated something the drivers do not directly say.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from terrella import data as D
from terrella import neural as N
from terrella.neural import aligned_windows as aligned
from terrella.probe import quiet_baseline, efold, impulse, partial_r2

torch.set_num_threads(1)
VBS_I = N.DRIVERS.index("vbs")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--untrained", action="store_true",
                    help="null control: random init, never trained. Its state is still a "
                         "nonlinear function of the same drivers, so whatever partial R^2 "
                         "it reaches is the floor a trained model has to clear.")
    args = ap.parse_args()

    splits = D.build()
    W = {k: aligned(v, stride=6) for k, v in splits.items()}
    Wg = {k: v[0] for k, v in W.items()}
    st = N.Stats(Wg["train"][0], Wg["train"][1])

    if args.untrained:
        torch.manual_seed(args.seed)
        model, val = N.StateModel(len(N.DRIVERS), args.dim), float("nan")
        model.eval()
    else:
        model, val = N.train(Wg["train"], Wg["val"], args.dim, st,
                             epochs=args.epochs, seed=args.seed)
    out = {"seed": args.seed, "dim": args.dim, "val_rmse": float(val),
           "untrained": args.untrained}

    # --- test skill (fully invariant) ---
    hd, hy, fd, fy = Wg["test"]
    pred = N.predict(model, Wg["test"], st)
    p, t = pred[:, -1], fy[:, -1]
    out["test_all"] = float(np.sqrt(np.mean((p - t) ** 2)))
    for name, thr in [("test_storm", -50), ("test_intense", -100)]:
        m = t < thr
        out[name] = float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))

    # --- impulse response (no basis at all) ---
    base, base_dst = quiet_baseline(splits["train"])
    traj, dst = impulse(model, st, base, base_dst)
    out["impulse_quiet_dst"] = float(dst[0])
    out["impulse_min_dst"] = float(dst.min())
    out["impulse_peak_h"] = int(np.argmin(dst))
    taus = []
    for d in range(args.dim):
        s = traj[:, d]
        pk = int(np.argmax(np.abs(s - s[0])))
        taus.append(efold(s[pk:]))
    out["efold_sorted"] = sorted([float(x) for x in taus if np.isfinite(x)])
    out["n_nondecaying"] = int(sum(1 for x in taus if not np.isfinite(x)))

    # --- latent trajectories on test ---
    xh = torch.tensor((hd - st.xm) / st.xs)
    yh = torch.tensor((hy - st.ym) / st.ys)
    xf = torch.tensor((fd - st.xm) / st.xs)
    with torch.no_grad():
        S = model.states(xh, yh, xf).numpy()
    L = S.reshape(-1, S.shape[-1])

    # observables + control blocks, aligned to the same forward steps
    H, T = N.HISTORY, fd.shape[1]
    obs = {"ae": [], "kp": [], "dst": []}
    for s in splits["test"]:
        s = N._prep(s)
        n = len(s) - H - T
        if n <= 0:
            continue
        fi = np.arange(0, n, 6)[:, None] + H + np.arange(T)[None, :]
        for k in obs:
            obs[k].append(s[k].to_numpy()[fi])
    obs = {k: np.concatenate(v).ravel() for k, v in obs.items()}

    drivers = fd.reshape(-1, fd.shape[-1])
    full_vbs = np.concatenate([hd[:, :, VBS_I], fd[:, :, VBS_I]], axis=1)
    lags = []
    for k in (3, 6, 12, 24):
        col = np.stack([full_vbs[:, H + t - k:H + t].mean(1) for t in range(T)], 1)
        lags.append(col.ravel())
    lagged = np.column_stack(lags)

    m = np.isfinite(obs["ae"]) & np.isfinite(obs["kp"]) & np.isfinite(obs["dst"])
    dst_c = obs["dst"][m][:, None]
    controls = {
        "A_dst": dst_c,
        "B_dst_drivers": np.column_stack([dst_c, drivers[m]]),
        "C_dst_drivers_lagged": np.column_stack([dst_c, drivers[m], lagged[m]]),
    }
    for target in ("ae", "kp"):
        for cname, C in controls.items():
            out[f"pr2_{target}_{cname}"] = float(partial_r2(obs[target][m], L[m], C))

    # permutation-invariant per-dimension spectra
    out["partial_corr_ae_sorted"] = sorted(
        (abs(float(np.corrcoef(L[m, d], obs["ae"][m])[0, 1])) for d in range(args.dim)),
        reverse=True)

    # --- ablation (sorted = permutation invariant) ---
    mu = torch.tensor(L.mean(0), dtype=torch.float32)
    truth = fy[:, -1]

    @torch.no_grad()
    def rmse_with(frozen=None):
        _, h = model.enc(torch.cat([xh, yh.unsqueeze(-1)], -1))
        h = h.squeeze(0)
        o = None
        for tt in range(xf.shape[1]):
            h = model.cell(xf[:, tt], h)
            if frozen is not None:
                h = h.clone()
                h[:, frozen] = mu[frozen]
            o = model.dec(h).squeeze(-1)
        q = o.numpy() * st.ys + st.ym
        return float(np.sqrt(np.mean((q - truth) ** 2)))

    full_rmse = rmse_with()
    degs = [rmse_with(d) / full_rmse - 1.0 for d in range(args.dim)]
    out["ablation_sorted"] = sorted(degs, reverse=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"seed {args.seed}: val {val:.2f}  test {out['test_intense']:.2f} (intense)  "
          f"impulse {out['impulse_min_dst']:.1f} nT  "
          f"pr2_ae_C {out['pr2_ae_C_dst_drivers_lagged']:.3f}")


if __name__ == "__main__":
    main()
