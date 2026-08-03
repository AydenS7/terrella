"""Load the OMNI hourly record, derive coupling functions, cut it into gap-free segments."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw/omni_hourly.csv")

COLS = ["time", "abs_b", "by_gsm", "bz_gsm", "n_p", "v_sw", "pressure",
        "e_field", "f107", "kp10", "dst", "ae"]

FILL = {"abs_b": 999.9, "by_gsm": 999.9, "bz_gsm": 999.9, "n_p": 999.9,
        "v_sw": 9999.0, "pressure": 99.99, "e_field": 999.99, "f107": 999.9,
        "kp10": 99, "dst": 99999, "ae": 9999}

# Chronological, forward-in-time. Test holds solar cycle 25 max including the 2024 Gannon
# storm (-406 nT). The val boundary sits at 2015 rather than 2018 because 2018-2020 was deep
# solar minimum — a later cut leaves val with 2 intense storms, too few to select on.
SPLITS = {"train": ("1998-01-01", "2015-01-01"),
          "val":   ("2015-01-01", "2023-01-01"),
          "test":  ("2023-01-01", "2026-07-23")}

# channels a sample needs before it counts as usable
REQUIRED = ["bz_gsm", "by_gsm", "v_sw", "pressure", "dst"]


def load_raw(path: pathlib.Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLS, header=None)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    for c, f in FILL.items():
        df.loc[df[c] >= f, c] = np.nan
    return df.set_index("time").sort_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bz, by, v = df["bz_gsm"], df["by_gsm"], df["v_sw"]

    df["bs"] = (-bz).clip(lower=0.0)                  # rectified southward field, nT
    df["vbs"] = v * df["bs"] * 1e-3                   # Burton driver, mV/m
    df["sqrt_p"] = np.sqrt(df["pressure"])            # magnetopause compression term

    b_t = np.hypot(by, bz)                            # transverse field, nT
    theta_c = np.arctan2(by, bz)                      # IMF clock angle, 0 = north
    df["clock"] = theta_c
    df["newell"] = (v ** (4 / 3)) * (b_t ** (2 / 3)) * np.abs(np.sin(theta_c / 2)) ** (8 / 3)

    df["kp"] = df["kp10"] / 10.0
    return df


def sanity_check(df: pd.DataFrame) -> dict:
    """Cross-check derived VBs against OMNI's own electric field. Catches sign/unit errors."""
    m = df[["e_field", "v_sw", "bz_gsm"]].notna().all(axis=1)
    ours = -df.loc[m, "v_sw"] * df.loc[m, "bz_gsm"] * 1e-3
    theirs = df.loc[m, "e_field"]
    return {"n": int(m.sum()),
            "corr": float(np.corrcoef(ours, theirs)[0, 1]),
            "max_abs_err": float((ours - theirs).abs().max())}


def segments(df: pd.DataFrame, cols=REQUIRED, max_gap_h: int = 3,
             min_len_h: int = 72) -> list[pd.DataFrame]:
    """Contiguous runs with all `cols` present. Gaps <= max_gap_h are interpolated."""
    ok = df[cols].notna().all(axis=1)

    # a short gap between two good stretches gets absorbed
    gap_id = (ok != ok.shift()).cumsum()
    for _, idx in df.groupby(gap_id).groups.items():
        if not ok.loc[idx].iloc[0] and len(idx) <= max_gap_h:
            ok.loc[idx] = True

    out = []
    run_id = (~ok).cumsum()
    for _, block in df[ok].groupby(run_id[ok]):
        if len(block) >= min_len_h:
            out.append(block.interpolate(limit_direction="both"))
    return out


def split_segments(segs: list[pd.DataFrame], splits=None) -> dict[str, list[pd.DataFrame]]:
    """Assign each segment to a split, cutting any that straddle a boundary."""
    splits = splits or SPLITS
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in splits}
    for name, (lo, hi) in splits.items():
        lo, hi = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
        for s in segs:
            piece = s[(s.index >= lo) & (s.index < hi)]
            if len(piece) >= 72:
                out[name].append(piece)
    return out


def storm_events(dst: pd.Series, thresh: float, gap_h: int = 48) -> list[float]:
    """Depth of each distinct excursion below `thresh`, merging dips < gap_h apart."""
    below = dst < thresh
    if not below.any():
        return []
    runs = dst[below].groupby((below != below.shift()).cumsum()[below])
    mins, last_end = [], None
    for _, r in runs:
        if last_end is not None and (r.index[0] - last_end).total_seconds() / 3600 < gap_h:
            mins[-1] = min(mins[-1], r.min())
        else:
            mins.append(r.min())
        last_end = r.index[-1]
    return mins


def build(era_start: str = "1998-01-01") -> dict[str, list[pd.DataFrame]]:
    """Val and test boundaries are fixed regardless of era, so extending the training era
    backwards leaves the test set byte-identical and the numbers comparable."""
    df = add_features(load_raw())
    df = df[df.index >= pd.Timestamp(era_start, tz="UTC")]
    splits = dict(SPLITS, train=(era_start, SPLITS["train"][1]))
    return split_segments(segments(df), splits)
