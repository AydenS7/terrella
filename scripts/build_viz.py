"""Inline the storm-replay trace and the aggregate headline into the viz template."""

import glob
import json
import pathlib

import numpy as np

TEMPLATE = pathlib.Path("viz/template.html")
OUT = pathlib.Path("viz/terrella.html")


def main():
    d = json.loads(pathlib.Path("data/cf/viz.json").read_text())

    runs = [json.load(open(f)) for f in sorted(glob.glob("data/cf/cf_s*.json"))]
    deepest = max(abs(b) for b in d["preload_bz"])
    shift = np.array([r[f"effect_abs_-{deepest:.1f}"] for r in runs])
    incr = np.array([r[f"effect_incremental_-{deepest:.1f}"] for r in runs])

    d["hero_fig"] = f"{shift.mean():+.1f} ± {shift.std():.1f} nT"
    d["hero_note"] = (
        f"Starting the storm from a magnetosphere already driven to about "
        f"{runs[0]['arms'][f'-{deepest:.1f}']['baseline_mean']:.0f} nT changes how deep it "
        f"finally gets by {shift.mean():+.1f} nT — indistinguishable from no change. The storm "
        f"does contribute {incr.mean():+.0f} nT less on its own, but that is almost exactly "
        f"cancelled by the lower starting point. Burton's 1975 equation compensates nearly as "
        f"completely, so this is mostly generic exponential forgetting, not a discovery."
    )

    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(d, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)  storm={d['storm']}  "
          f"hero={d['hero_fig']}")


if __name__ == "__main__":
    main()
