"""Fetch the 1-minute OMNI record (OMNI_HRO_1MIN) from CDAWeb HAPI, in yearly chunks.

The hourly product was a v0 convenience: it is the only OMNI file reaching back to 1963
and it carries Dst in the same pull. But hourly averaging destroys the sub-hourly structure
- shock arrivals, abrupt IMF rotations - that produces the sharp features the model cannot
reproduce, and Dst itself is only defined hourly.

SYM/H is the fix. It is effectively Dst at 1-minute cadence, and it lives in this product
alongside 1-minute drivers, so both sides of the problem gain resolution at once.

Chunked by year so a failure costs one year, not the whole download.
"""

import pathlib
import sys
import time
import urllib.error
import urllib.request

DATASET = "OMNI_HRO_1MIN"
START_YEAR, END_YEAR = 1997, 2026   # pre-1998 gains nothing (see README), 1 year of margin
STOP = "2026-07-08T00:00:00Z"

# schema order; HAPI returns error 1411 otherwise
PARAMS = [
    "BY_GSM",          # IMF By (nT), GSM
    "BZ_GSM",          # IMF Bz (nT), GSM — dominant coupling driver
    "flow_speed",      # km/s
    "proton_density",  # n/cc
    "Pressure",        # dynamic pressure, nPa
    "AE_INDEX",        # 1-min auroral electrojet
    "AL_INDEX",        # 1-min westward electrojet — substorm onset signature
    "SYM_H",           # 1-min ring current — the high-cadence Dst analogue, our new target
    "ASY_H",           # ring current asymmetry, absent from the hourly product entirely
]

OUT = pathlib.Path("data/raw/1min")


def fetch_year(year: int) -> pathlib.Path:
    dest = OUT / f"omni_1min_{year}.csv"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    lo = f"{year}-01-01T00:00:00Z"
    hi = STOP if year == END_YEAR else f"{year + 1}-01-01T00:00:00Z"
    url = (f"https://cdaweb.gsfc.nasa.gov/hapi/data?id={DATASET}"
           f"&time.min={lo}&time.max={hi}&parameters={','.join(PARAMS)}&format=csv")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  {year}: attempt {attempt + 1} failed ({e}); retrying", flush=True)
            time.sleep(5)
    raise RuntimeError(f"{year} failed after 3 attempts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for year in range(START_YEAR, END_YEAR + 1):
        t0 = time.time()
        p = fetch_year(year)
        mb = p.stat().st_size / 1e6
        total += mb
        print(f"  {year}  {mb:6.1f} MB  ({time.time() - t0:5.1f}s)  running total {total:.0f} MB",
              flush=True)
    print(f"done: {total:.0f} MB across {END_YEAR - START_YEAR + 1} years in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
