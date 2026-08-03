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
driver, and the observable channels have to be reconstructed from that state alone.

The current model reconstructs Dst only, which makes AE and Kp available as observables it
has never seen — a test of whether the state describes the system's condition or is merely
a feature vector tuned to predict one number. Reconstructing all three jointly is the
planned next step, and would trade that test away for a stronger constraint.

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
transition will have to be probabilistic, and why calibration (spread-skill, CRPS, rank
histograms) is a first-class requirement rather than a nice-to-have. The models built so
far are deterministic and therefore cannot do this yet.

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

**The curve elbows hard at H=4 and then goes flat.**

Read this table with the seed study below in hand. Each row is 2 seeds, and across 8 seeds
at H=4 the spread is ±0.35 nT on `all`, ±1.5 on `storm`, and **±4.5 on `intense`**. So:

- The 1 → 4 improvement on `all` (11.31 → 8.57) is roughly 11× the seed standard error.
  Real, and the elbow is real.
- The differences *among* H=4, 8, 32 and 64 are well inside seed noise. The apparent win of
  H=4 over H=64 on intense storms is not a finding — it is two lucky seeds.

So the honest claim is: **more than one dimension is needed, about four is enough, and past
four nothing improves.** Not "four is optimal."

Two separate effects are visible. At fixed H=1, replacing Burton's hand-written transition
with a learned one is worth ~12% on intense storms — that is nonlinearity, not
dimensionality. The 1 → 4 gain on top of that is dimensionality.

## Are those four dimensions physics, or noise?

`scripts/probe.py` runs three tests on one model. `scripts/seed_run.py` reruns them across
8 seeds against 8 **untrained** null models, and that is what the results below report,
because a single seed turned out to be badly misleading.

Two methodological points, both of which changed the answer:

- **Latent dimensions have no identity across seeds.** The basis is arbitrary up to
  permutation, sign and rotation, so "dimension 2 does X" cannot be checked by rerunning.
  Everything here is either invariant to an invertible linear change of basis (partial R²,
  impulse response) or invariant to permutation (sorted spectra).
- **An untrained model is the null, not zero.** A random GRU driven by the same solar wind
  is a nonlinear feature expansion of those drivers, so its state already carries a lot of
  information about everything the drivers cause. Beating zero means nothing. Beating the
  untrained model is the bar.

### 1. Impulse response — holds up

Hold the model at a quiet equilibrium (median conditions during |Dst| < 20), inject 6 hours
of Bz = −10 nT, release.

| | trained (8 seeds) | untrained null |
|---|---|---|
| Dst minimum | **−56.1 ± 4.1 nT** | −8 to −23 nT |
| peak at | +5.5 ± 0.5 h | — |

A textbook moderate storm from a textbook moderate driver, arriving on a sensible timescale,
from a model trained only to minimize error. The null gets nowhere near it, so this is a
trained behaviour and not an artifact of the architecture.

### 2. Decay timescales — claim withdrawn

On one seed this looked clean: dimensions decaying with e-folding times of 19 h, 20 h and
31 h, comfortably near the ring current's 7–20 h. Across 8 seeds it evaporates —
**17.0 ± 10.6 h, spanning 2 h to 38 h.** That is not a set of physical constants, that is
scatter that happened to look meaningful once. There is a spread of fast and slow modes,
which is something, but no identifiable timescale.

### 3. Information about observables the model never saw

The model is trained on Dst alone; AE and Kp are never shown to it. Partial R² is the
fraction of the target's variance, left unexplained by the controls, that the latent state
explains. Controls are graded, because the latent is a function of recent drivers and those
drivers also drive AE and Kp — "knows about AE" could just mean "remembers that VBs was
recently high."

| target | control | trained | untrained null | t |
|---|---|---|---|---|
| AE | A: Dst | 0.340 ± 0.090 | 0.265 ± 0.100 | +1.48 |
| AE | B: + instantaneous drivers | 0.139 ± 0.060 | 0.106 ± 0.040 | +1.20 |
| AE | C: + lagged VBs (3/6/12/24 h) | 0.116 ± 0.051 | 0.080 ± 0.033 | +1.55 |
| Kp | A: Dst | 0.366 ± 0.064 | 0.265 ± 0.126 | +1.89 |
| Kp | B: + instantaneous drivers | 0.137 ± 0.045 | 0.080 ± 0.030 | +2.82 |
| **Kp** | **C: + lagged VBs** | **0.109 ± 0.035** | **0.055 ± 0.023** | **+3.41** |

Controlling only for Dst, the latent looks like it explains a third of AE — which is what an
earlier single-seed run reported. Almost all of that is the drivers. Against the untrained
null, **AE does not survive** (t = +1.55, not significant). **Kp does** (t = +3.41): trained
models carry about twice the Kp information a random model does, beyond Dst, the
instantaneous drivers, and 24 hours of driver history.

That is a real but modest result. Kp is a 3-hourly planetary index; carrying information
about it that neither Dst nor recent driving explains means the state has integrated
something. It is not the "it discovered substorms" story the first run appeared to tell.

### 4. Ablation — holds up

Freeze one dimension at its mean during rollout, sorted by impact, across 8 seeds:

| rank | degradation |
|---|---|
| 1 | +128 ± 33% |
| 2 | +68 ± 17% |
| 3 | +33 ± 17% |
| 4 | +21 ± 12% |

The least important dimension in the worst seed still costs +3.9%. **No dead dimensions in
any seed** — H=4 is genuinely used, not padded.

## What is and isn't established

Holds up across seeds and against the null:

- Four dimensions beat one, decisively, on held-out data from a different solar cycle.
- All four are load-bearing; none is decorative.
- The impulse response is physically sensible and is a trained behaviour.
- The state carries Kp information beyond Dst, the drivers, and driver history.

Does not hold up:

- **Identifiable decay timescales.** Looked like ring current on one seed; 17 ± 11 h across
  eight.
- **The AE result.** Survives controlling for Dst, does not survive controlling for the
  drivers, and does not beat an untrained model.
- **H=4 being optimal.** Within seed noise of H=8 through H=64.

Also still open: honest test-set skill at H=4 is **36.4 ± 4.5 nT** on intense storms, not the
31.67 the 2-seed sweep reported. That still clears Burton–OM's 50.68 in every seed, so the
headline comparison survives — the margin is ~28%, not ~37%.

And the model remains deterministic, so it still says nothing about the probability of an
extreme event, which is the point of the whole exercise.

## Status

Data pipeline, baselines, dimensionality sweep, and latent probe done. Next: storm-weighted
loss, then the probabilistic transition and calibration.

## License

MIT
