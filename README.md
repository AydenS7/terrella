# Terrella

**A learned simulator of Earth's magnetosphere.**

In 1600 William Gilbert magnetized a sphere of lodestone, called it a *terrella* — a
little Earth — and used it to reason about the planet's magnetic field. Three hundred
years later Kristian Birkeland put a terrella in a vacuum chamber, fired cathode rays at
it, and watched glowing rings form at its poles. That experiment is why we know the
aurora is caused by charged particles from the Sun.

This is the same instrument, built out of data instead of brass.

---

## What it does

The Sun blows charged gas at Earth. Earth's magnetic field mostly deflects it, but the
field also *stores* the energy that gets through — quietly, invisibly — and then dumps it
all at once. That dump is a geomagnetic storm: aurora, satellite drag, GPS errors, and in
the worst cases, current surges that destroy power transformers.

The awkward part: the same solar wind conditions sometimes produce a huge storm and
sometimes nothing at all. The outcome depends on how much energy the system had *already*
stored — and there is no instrument anywhere that measures that. It's a state variable
nobody can see.

Terrella learns it from 63 years of public NASA data.

It's a state-space model: a latent state evolves under the solar wind as an exogenous
driver, and must reconstruct every observed channel (Dst, AE, Kp) from that state alone.
The reconstruction constraint is the point — it forces the latent toward being a
description of the system's actual condition rather than a feature vector tuned to
predict one number.

## Why bother, when LSTMs already predict Dst

They do, and they've done it since the 1990s. Beating them on point-forecast skill is not
the goal and probably isn't achievable.

The goal is an **experimental apparatus for a system nobody can run experiments on.**
Once you have a simulator with an explicit state you can intervene on, you can ask
questions that a regression model structurally cannot:

1. **What is the stored energy right now?** No instrument measures it. If the model
   learned it, we can read it off.

2. **How many degrees of freedom does the magnetosphere actually have?** The standard
   low-order model (Burton et al. 1975) says one. In the 1990s there was a real argument
   about whether the system is low-dimensional, and it was never settled — the records
   were too short and the methods too weak. The record is now twice as long. This is a
   measurable number and nobody has a credible modern answer.

3. **Does pre-conditioning matter?** Take the March 1989 storm. Run it again against a
   magnetosphere that was already disturbed. Twice as bad? Ten times? This is the
   scenario that worries grid operators — sequential CMEs, the second arriving into an
   already-loaded system — and it cannot be studied observationally. There are a handful
   of such events, no controls, and no way to repeat them. You only get one Earth.

4. **How likely is the catastrophic case?** Sample the model's rollouts many times and
   count. That's a calibrated probability of a severe storm, not a point guess.

## Data

Everything comes from one public endpoint, no key, no auth: NASA's
[HAPI](https://hapi-server.org/) server at CDAWeb, dataset `OMNI2_H0_MRG1HR` — the merged
hourly OMNI record, 1963-01-01 to present. It carries the solar wind drivers *and* the
geomagnetic indices in the same file.

```bash
python scripts/fetch_omni.py     # ~2 min, 35 MB
python scripts/explore.py        # coverage + storm event survey
```

### What the record actually contains

557,136 hourly rows spanning 63.5 years.

| Storm class | Dst threshold | Events | Per year |
|---|---|---|---|
| Moderate | < −50 nT | 1218 | 19.2 |
| Intense | < −100 nT | 330 | 5.2 |
| Severe | < −200 nT | 56 | 0.9 |
| Great | < −250 nT | 25 | 0.4 |
| Extreme | < −300 nT | 12 | 0.2 |
| | < −400 nT | 3 | 0.05 |

Deepest on record: 1989-03-14 at −589 nT (the storm that took down the Quebec grid),
2003-11-20 at −422 nT, and 2024-05-11 at −406 nT (Gannon).

**The binding constraint:** solar wind coverage is ~100% after 1998 but only ~55–63%
before it, and during intense storms specifically, Bz is present just 68.8% of the time.
So the number of extreme events that have usable drivers attached is in the single digits.

You cannot fit a model to three examples of a −400 nT storm. The tail has to come from
the model's generative distribution rather than from memorized cases — which is why the
transition is probabilistic and why calibration (spread-skill, CRPS, rank histograms) is
a first-class requirement rather than a nice-to-have.

### Splits

Chronological and forward-in-time — always train on the past. The val boundary sits at
2015 rather than 2018 because 2018–2020 was deep solar minimum; cutting later leaves val
with 2 intense storms, too few to select on.

| split | span | years | storms <−50 / <−100 / <−200 | deepest |
|---|---|---|---|---|
| train | 1998–2015 | 16.3 | 280 / 79 / 13 | −422 nT |
| val | 2015–2023 | 7.9 | 107 / 14 / 1 | −234 nT |
| test | 2023–2026 | 3.4 | 78 / 26 / 6 | **−406 nT** (Gannon) |

## Baselines

Everything is scored **free-running**: the state is initialized from a single observed
Dst and then integrated forward on solar wind drivers alone, with no further
observations. That's the regime the learned model will be judged in, so the numbers are
directly comparable.

Fitting Burton also serves as the pipeline check. The optimizer, starting from published
values and given a 63-year record it has never seen before, recovers:

| coefficient | fitted | O'Brien & McPherron (2000) |
|---|---|---|
| a (injection) | −4.433 | −4.4 |
| b (pressure) | 7.434 | 7.26 |
| c (offset) | 9.47 | 11 |

That agreement is the evidence the data pipeline is sane. A sign error or unit slip
anywhere would have shown up here.

### Test set, 24h lead — RMSE in nT

| model | params | all | storm <−50 | intense <−100 |
|---|---|---|---|---|
| persistence | 0 | 26.70 | 84.37 | 161.96 |
| Burton (constant τ) | 5 | 13.16 | 35.34 | 69.28 |
| **Burton–OM (τ = f(VBs))** | **5** | **11.64** | **24.70** | **46.74** |

**Burton–OM is the bar.** 11.64 / 24.70 / 46.74.

## How many dimensions does it need?

A GRU run free on drivers with hidden size *H* is a learned state-space model with *H*
dimensions, so sweeping *H* traces the dimensionality curve. Burton sits at H=1. Scored on
identical windows, 2 seeds, 24h lead, test set:

| model | params | all | storm <−50 | intense <−100 |
|---|---|---|---|---|
| Burton–OM (H=1) | 5 | 11.84 | 26.72 | 50.68 |
| GRU H=1 | 184 | 11.31 | 25.18 | 44.80 |
| GRU H=2 | 315 | 10.23 | 24.51 | 40.95 |
| **GRU H=4** | **613** | **8.57** | **19.04** | **31.67** |
| GRU H=8 | 1,353 | 8.60 | 19.22 | 32.56 |
| GRU H=16 | 3,409 | 8.76 | 20.68 | 36.63 |
| GRU H=32 | 9,825 | 8.46 | 19.45 | 34.63 |
| GRU H=64 | 31,873 | 8.37 | 19.41 | 32.56 |

**The curve elbows hard at H=4 and then goes flat.** Going 1 → 2 → 4 buys 26% on intense
storms; going 4 → 64 buys nothing, despite 52× the parameters. H=4 has the best
intense-storm score in the entire sweep.

Two separate things are visible here. At fixed H=1, replacing Burton's hand-written
transition with a learned one is worth ~12% on intense storms — that's nonlinearity, not
dimensionality. The 1 → 4 gain on top of that is dimensionality.

## Are those four dimensions physics, or noise?

`scripts/probe.py`, three tests.

**1. Impulse response.** Hold the model at a quiet equilibrium (median conditions during
|Dst| < 20), inject 6 hours of Bz = −10 nT, release. Predicted Dst goes −14 → **−52 nT**,
a textbook moderate storm from a textbook moderate driver. Nobody asked for that; it falls
out of a model trained only to minimize error.

The four dimensions respond on visibly different timescales:

| dim | peak at | e-folding decay |
|---|---|---|
| 0 | +5 h | 19 h |
| 2 | +7 h | 20 h |
| 1 | +20 h | 31 h |
| 3 | +77 h | did not decay in 72 h |

Fast responders that decay in ~19–20 h, a slower one at 31 h, and a very slow integrator.
That is structure, not four copies of the same thing.

**2. Correlation with observables the model never saw.** The model is trained on Dst alone;
AE and Kp are never shown to it, in training or at inference. Raw correlations look strong,
but AE and Kp are themselves correlated with Dst (−0.551 and −0.596), so a dimension that
merely tracks Dst would inherit an AE correlation for free. Partialling Dst out:

| dim | AE ǀ Dst | Kp ǀ Dst |
|---|---|---|
| 0 | −0.326 | −0.478 |
| 1 | +0.183 | −0.048 |
| **2** | **−0.403** | **−0.482** |
| 3 | −0.196 | −0.137 |

They survive. The latent state carries information about auroral and planetary activity
that Dst alone does not explain. Dimension 2 is the interesting one: it has the *weakest*
raw Dst correlation (−0.187) and the *strongest* partial AE correlation, which is what a
substorm-like, Dst-orthogonal quantity would look like.

**3. Ablation.** Freeze one dimension at its mean during rollout:

| frozen dim | test RMSE | degradation |
|---|---|---|
| none | 8.77 | — |
| 0 | 20.55 | +134.3% |
| 3 | 12.70 | +44.8% |
| 1 | 10.74 | +22.5% |
| 2 | 10.02 | +14.3% |

All four are load-bearing. If the extra dimensions were noise, freezing them would cost
nothing.

## What is and isn't established

Established: four dimensions beat one, decisively and on held-out data from a different
solar cycle; all four carry weight; the impulse response is physically sensible; the state
knows things about AE and Kp that Dst alone doesn't explain.

Not established: that the decay constants match the ring current specifically — 19–31 h sits
at and above the usual 7–20 h band. Partial correlations of 0.3–0.5 are suggestive, not
decisive. The probe is a single seed. And the model is deterministic, so it still cannot say
anything about the probability of an extreme event, which is the point of the whole exercise.

## Status

Data pipeline, baselines, dimensionality sweep, and latent probe done. Next: storm-weighted
loss, then the probabilistic transition and calibration.

## License

MIT
