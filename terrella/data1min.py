"""1-minute OMNI: load, derive coupling terms at native cadence, aggregate to model steps.

The point is not just "more rows". Two things change:

1. **Coupling functions are computed at 1-minute and then aggregated.** VBs and the Newell
   function are nonlinear in Bz, so mean(VBs) != VBs(mean Bz). Computing them from hourly
   averages systematically under-counts short sharp southward excursions - exactly the
   events that drive the sharp features v1 could not reproduce. This is a correctness fix,
   not only a resolution one.

2. **SYM/H replaces Dst as the target.** Dst is only defined hourly, so v1 had a ceiling on
   its output sharpness no amount of driver resolution could lift. SYM/H is the 1-minute
   ring-current index and is what makes finer model steps meaningful at all.

Each model step carries within-step statistics (min/max/std), so sub-step structure
survives aggregation instead of being averaged away.
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw/1min")

COLS = ["time", "by_gsm", "bz_gsm", "v_sw", "n_p", "pressure",
        "ae", "al", "sym_h", "asy_h"]

FILL = {"by_gsm": 9999.99, "bz_gsm": 9999.99, "v_sw": 99999.9, "n_p": 999.99,
        "pressure": 99.99, "ae": 99999, "al": 99999, "sym_h": 99999, "asy_h": 99999}

# channels a model step needs before it counts as usable
REQUIRED = ["bz_gsm", "by_gsm", "v_sw", "pressure", "sym_h"]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Coupling terms at native 1-minute cadence, before any aggregation."""
    bz, by, v = df["bz_gsm"], df["by_gsm"], df["v_sw"]
    df["bs"] = (-bz).clip(lower=0.0)
    df["vbs"] = v * df["bs"] * 1e-3
    df["sqrt_p"] = np.sqrt(df["pressure"])
    b_t = np.hypot(by, bz)
    theta = np.arctan2(by, bz)
    df["clock"] = theta
    df["newell"] = (v ** (4 / 3)) * (b_t ** (2 / 3)) * np.abs(np.sin(theta / 2)) ** (8 / 3)
    return df


def load_year(path: str | pathlib.Path, step: str = "60min") -> pd.DataFrame:
    df = pd.read_csv(path, names=COLS, header=None)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    for c, f in FILL.items():
        df.loc[df[c] >= f, c] = np.nan
    df = _derive(df.set_index("time").sort_index())

    r = df.resample(step)
    out = pd.DataFrame({
        # means: the v1-equivalent view
        "bz_gsm": r["bz_gsm"].mean(), "by_gsm": r["by_gsm"].mean(),
        "v_sw": r["v_sw"].mean(), "n_p": r["n_p"].mean(),
        "pressure": r["pressure"].mean(), "sqrt_p": r["sqrt_p"].mean(),
        "vbs": r["vbs"].mean(), "newell": r["newell"].mean(),
        # within-step structure: what hourly averaging destroyed
        "bz_min": r["bz_gsm"].min(), "bz_max": r["bz_gsm"].max(),
        "bz_std": r["bz_gsm"].std(), "vbs_max": r["vbs"].max(),
        "newell_max": r["newell"].max(), "v_max": r["v_sw"].max(),
        "p_max": r["pressure"].max(),
        # observation channels
        "sym_h": r["sym_h"].mean(), "sym_h_min": r["sym_h"].min(),
        "asy_h": r["asy_h"].mean(), "ae": r["ae"].mean(), "al": r["al"].min(),
        "n_obs": r["bz_gsm"].count(),
    })
    theta = np.arctan2(out["by_gsm"], out["bz_gsm"])
    out["clock"] = theta
    out["sin_clock"] = np.sin(theta)
    out["cos_clock"] = np.cos(theta)
    return out


def load_all(step: str = "60min", cache: bool = True) -> pd.DataFrame:
    cache_path = RAW.parent / f"omni_1min_agg_{step}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    files = sorted(glob.glob(str(RAW / "omni_1min_*.csv")))
    if not files:
        raise FileNotFoundError(f"no 1-minute files in {RAW}; run scripts/fetch_omni_1min.py")
    df = pd.concat([load_year(f, step) for f in files]).sort_index()
    if cache:
        df.to_parquet(cache_path)
    return df


def segments(df: pd.DataFrame, cols=REQUIRED, max_gap: int = 3,
             min_len: int = 72, min_obs: int = 1) -> list[pd.DataFrame]:
    """Contiguous runs with all `cols` present and enough underlying 1-minute samples."""
    ok = df[cols].notna().all(axis=1) & (df["n_obs"] >= min_obs)
    gap_id = (ok != ok.shift()).cumsum()
    for _, idx in df.groupby(gap_id).groups.items():
        if not ok.loc[idx].iloc[0] and len(idx) <= max_gap:
            ok.loc[idx] = True
    out, run_id = [], (~ok).cumsum()
    for _, block in df[ok].groupby(run_id[ok]):
        if len(block) >= min_len:
            out.append(block.interpolate(limit_direction="both"))
    return out
