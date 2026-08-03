"""Fetch the merged hourly OMNI record from NASA CDAWeb's HAPI server.

No API key. Parameters must be requested in schema order or the server returns 1411.
"""

import pathlib
import urllib.request

DATASET = "OMNI2_H0_MRG1HR"
START = "1963-01-01T00:00:00Z"
STOP = "2026-07-23T00:00:00Z"

# schema order matters — see /hapi/info?id=OMNI2_H0_MRG1HR
PARAMS = [
    "BZ_GSM1800",      # IMF Bz, GSM frame (nT) — the dominant coupling driver
    "N1800",           # ion density (cm^-3)
    "V1800",           # flow speed (km/s)
    "Pressure1800",    # dynamic pressure (nPa)
    "E1800",           # motional electric field (mV/m)
    "F10_INDEX1800",   # F10.7 solar radio flux — solar cycle phase
    "KP1800",          # Kp*10, 3-hourly
    "DST1800",         # Dst (nT), hourly — primary response channel
    "AE1800",          # AE (nT), hourly — auroral electrojet / substorm activity
]

OUT = pathlib.Path("data/raw/omni_hourly.csv")


def main():
    url = (
        "https://cdaweb.gsfc.nasa.gov/hapi/data"
        f"?id={DATASET}&time.min={START}&time.max={STOP}"
        f"&parameters={','.join(PARAMS)}&format=csv"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {DATASET} {START[:10]} -> {STOP[:10]} ...")
    urllib.request.urlretrieve(url, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
