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

## Status

Early. Data survey complete, model not yet built.

## License

MIT
