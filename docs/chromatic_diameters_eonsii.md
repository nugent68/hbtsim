# Chromatic diameters of Sirius A and Vega with EON-SII

Question: EON-SII's 400–550 nm spectrograph (arXiv:2608.17444) holds
Hδ, Hγ and Hβ at once. Can its two 4 m units measure the Balmer-core
versus continuum diameter difference, i.e. the centre-to-limb contrast
between the line- and continuum-forming layers predicted by the NewEra
models, on the two brightest A stars? This supersedes the 1 m
feasibility note (`/Users/nugent/phoenix/docs/chromatic_diameters_1m.md`),
which predated the radius convention, channel averaging and pupil
smearing, and lived outside hbtsim.

**Answer: yes, easily in photon terms.** With every channel tagged
through the 1 GHz links, the result is 5σ in 1–2 nights. With only the
17–150 channels that fit the links at full rate, or an on-detector
correlator, it is tens to hundreds of σ per night. What limits the
measurement is channel-to-channel systematics, not photons.

Code: `hbtsim/single.py` (single stars, pupil-smeared uniform-disk
inversion), `hbtsim/chromatic.py` (line masks, local continuum fits,
Asimov significance, link-subset channel selection),
`hbtsim run chromatic_sirius_vega_eonsii` (regression: `chromatic_c2pu_regression`). Logs:
`output/logs/chromatic_diameters_eonsii.txt`,
`output/logs/chromatic_diameters_c2pu_regression.txt`. Figures:
`output/chromatic_{sirius,vega}_eonsii.png` (SPAD, subset readout).

## Set-up

- **Stars.**
  - Sirius A: 9940 K, log g 4.33, θ_LD = 6.039 mas (Kervella et al. 2003).
    NewEra interpolated from the 9800/10 000 K × log g 4.0/4.5 corners,
    so r_outer = 1.0047 and the disk is drawn at 6.067 mas. The model
    AB(550) − V = +0.07.
  - Vega: 9550 K, log g 4.0, θ_LD = 3.329 mas (Aufdenberg et al. 2006).
    Interpolated between 9400 and 9600 K at log g 4.0; r_outer = 1.0044;
    AB(550) − V = −0.02.
  - Magnitudes come from the model, not anchored to V; the V check is
    printed.
- **Instrument.** EON-SII 4 m units from Teide: 5.5 h per night above
  30° for Sirius (it culminates at 45°), 8 h for Vega.
  - Backends: 1000 × 0.15 nm channels with the MCP-PMT or QUASAR SPAD
    (± polarizing beamsplitter), and the R = 7500 (2389 ch) SPAD
    variant.
  - Pupils are averaged: D/B ≈ 0.4 on Sirius. The UD inversion inverts
    the smeared model, because inverting the point model biases θ by
    > 0.5 % at D/B ≈ 0.4 (`tests/test_single.py`).
- **Baseline.** Chosen per case by a first-lobe scan (x = πθB/λ from
  1.0 to 3.6, never below the assumed 6 m minimum spacing of two 4 m
  units). It lands at |V|² ≈ 0.37: B = 10.3 m for Sirius and 18.7 m for
  Vega.
- **Statistic.** Asimov and local to each line. Take the channels
  within ±8 nm of the line, plus the continuum up to 15 nm beyond, and
  find the smallest χ² that a straight line in λ can reach through the
  model θ_UD(λ). The significance is √χ²_min, and the three lines add in
  quadrature. A single polynomial across 400–550 nm would also count
  its own failure to follow three separate bumps, and overstates the
  result.

## Model signal

| | Hδ 410.3 nm | Hγ 434.2 nm | Hβ 486.3 nm | continuum 400 → 550 nm |
|---|---|---|---|---|
| Sirius A: θ_UD(core)/continuum − 1 | +4.4 % | +4.1 % | +3.6 % | +1.8 % |
| Vega | +4.7 % | +4.4 % | +3.8 % | +1.9 % |

The 1 m study's Hβ number (+3.5 % for Sirius from the 10 000 K / 4.5
model) is reproduced (+3.51 %, regression below). The signal grows
toward the higher Balmer lines. Metal lines (Mg II 448.1 nm) show up
at the 0.5 % level (figure).

## Detectability per night

Significance per night (Sirius: 5.5 h; Vega: 8 h) and nights to 5σ:

| Backend | Readout | Sirius A | Vega |
|---|---|---|---|
| 1000 ch, MCP-PMT | link (×0.020 / ×0.072) | 2.8σ → 3.1 nights | 3.5σ → 2.0 nights |
| 1000 ch, SPAD | link (×0.010 / ×0.038) | 3.8σ → 1.7 nights | 4.7σ → 1.1 nights |
| 1000 ch, SPAD + PBS | link | 5.4σ → 0.9 nights | 6.7σ → 0.6 nights |
| R = 7500, SPAD | link | 5.8σ → 0.7 nights | 7.3σ → 0.5 nights |
| 1000 ch, MCP-PMT | subset (31 / 86 ch) | 72σ | 29σ |
| 1000 ch, SPAD | subset (17 / 59 ch) | 85σ | 57σ |
| 1000 ch, SPAD | correlator | 230σ | 108σ |
| R = 7500, SPAD | correlator | 452σ | 182σ |

**Readout is the decisive design choice.** Sirius sends about 10¹¹
photons/s per telescope into the 1000 channels, and the link carries
10⁹. Attenuating every channel costs the square of the attenuation in
information, because g² S/N is linear in the rate. Tagging only the
channels that fit at full rate keeps it:
- a quarter of the link goes to continuum reference channels next to
  each line;
- the rest goes to the core and wing channels with the most signal per
  photon.

The subset numbers depend on a handful of reference channels and are
statistical only. Read them as "photons are not the limit" rather than
as literal significances.

## Regression against the 1 m study

`--instrument c2pu --hbeta-window` uses two 1 m telescopes, 320 channels
at R = 5000 around Hβ, the SPAD Lambda correlator, 6 h and a flat
continuum. It gives Sirius Hβ +3.51 % at 4.9σ per night (the 1 m study:
+3.5 %, 5.2–5.7σ) and Vega +3.73 % at 1.4σ, 13 nights to 5σ (1.4σ,
13 nights). The remaining 6 % on Sirius comes from the interpolated
model (9940 K / 4.33, not the 10 000 K / 4.5 corner) and the model
magnitudes (not V-anchored).

## Caveats

- **Systematics dominate.** The spectrograph line-spread function
  leaks continuum light into the core channels and dilutes the signal
  in proportion to the leak over the core depth (about 0.35 at
  R = 5000–7500). The other limits are SPAD inter-pixel cross-talk and
  per-channel g² normalization drifts. None of them cancels in the
  ratio, and all need a laboratory characterization.
- **Metallicity.** The grid is [M/H] = 0. Sirius is a metallic-line
  (Am) star ([Fe/H] ≈ +0.5); Vega is mildly metal-poor ([M/H] ≈ −0.5)
  with λ Boo-type abundances. Balmer cores are insensitive, but the metal
  lines in the continuum windows are not.
- **Vega is a pole-on rapid rotator.** There is about 2000 K between
  pole and equator (Aufdenberg et al. 2006), so a single spherical 1D
  model is only indicative. The chromatic signature of gravity
  darkening is itself a target.
- **Teide airmass.** Sirius culminates at 45°, i.e. airmass 1.4–2.
  Extinction is not in the budget (it costs about 25–40 % of the rate in
  the blue); the link and subset strategies are affected alike.
- **Minimum spacing.** The 6 m centre-to-centre spacing of two 4 m units
  is an assumption. Sirius's optimum baseline (10.3 m) is above it in
  every configuration studied.
- **Sirius B** (ΔV ≈ 10) is negligible.

## What the measurement tests

θ_UD(core)/θ_UD(continuum) is set by the ratio of limb darkening in the
layers that form the Balmer cores (high, shallow temperature gradient)
to that in the continuum-forming layers. It is a direct test of the
upper-photosphere temperature structure of the spherical NewEra A-star
models: +4.4 / +4.1 / +3.6 % at Hδ / Hγ / Hβ for Sirius, and +1.8 %
continuum chromaticity across 400–550 nm, all measured simultaneously
in one EON-SII configuration.
