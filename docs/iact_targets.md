# Cherenkov-telescope intensity interferometers in hbtsim: Spica on VERITAS, γ Cas on MAGIC + LST-1

Made for the SII Workshop 2026 talk (6 October): one slide each applying hbtsim to the
targets two groups present there (M. Lisa: five years of Spica on VERITAS; I. Jiménez:
γ Cas on MAGIC + LST-1). Module `hbtsim/iact.py`; scripts `scripts/spica_veritas.py`,
`scripts/gammacas_magic.py`; logs `output/logs/{spica_veritas,gammacas_magic}.txt`;
figures `output/{spica_veritas,gammacas_magic}.png`; tests `tests/test_iact.py`.

## What hbtsim supplies, and what it doesn't

These instruments digitize photomultiplier currents at 250–500 MS/s and correlate the
streams. They are analog, nanosecond systems, not photon counters, so their sensitivity
is taken from their own published form (MAGIC, Abe et al. 2024, Eq. 4; the same expression
in Raiola et al. 2025):

    S/N = A α q n_ν |V|² √b_el σ_spec / (√2 F (1 + β)) √T

with their constants (`iact.IACTBackend`). hbtsim supplies only the visibility model: the
analytic two-disk binary on arbitrary projected baselines, limb-darkened (and now
elliptical) single disks, a Gaussian circumstellar component, and the averaging over the
dish apertures. Sites, layouts and the per-pair hour-angle loop are new (`iact.pair_track`).

## Parameters and sources

<!-- catalog:iact_parameters -->
|  | VERITAS | MAGIC | LST-1 |
|---|---|---|---|
| dishes | 4 × 12 m, 110 m² (**assumed**) | 2 × 17 m, 236 m² | 23 m, 390 m² (**assumed**) |
| site | Fred Lawrence Whipple Observatory (VERITAS) (31.675°, -110.952°, 1268 m) | Roque de los Muchachos (MAGIC, LST-1) (28.762°, -17.892°, 2200 m) | same |
| baselines | 81.6, 99.4, 99.5, 108.9, 126.4, 172.6 m (fitted to the published values) | MAGIC-I–II 86 m | 100, 100 m to LST-1; **orientation assumed** |
| filter | 416 nm, 13 nm effective | 425 nm / 26 nm | as MAGIC (assumed) |
| QE α | 0.3 | 0.295 | as MAGIC |
| optical q | **calibrated: 0.093** (file: 0.25) | 0.304 | as MAGIC |
| b_el | 125 MHz; 4 ns time resolution | 110 MHz effective; 2 ns | as MAGIC |
| F, σ_spec | absorbed into q | 1.15, 0.87 | as MAGIC |
| precision anchor | eps Ori: σ(|V|²) = 0.016 per pair in 4.25 h at AB 1.41 | — | — |
<!-- /catalog -->

Sources: Abeysekara et al. 2020, Nat. Astron. (arXiv:2007.10295); VERITAS γ Cas 2025, ApJ 995,
191 (arXiv:2506.15027); Abe et al. 2024, MNRAS 529, 4387 (arXiv:2402.04755); Raiola et al.
2025, PoS ICRC2025 957.

**VERITAS calibration.** The optical efficiency of the uncollimated-filter chain is not
published, so `iact.calibrate_q` solves for the q that reproduces their quoted precision:
σ(|g|²) = 2×10⁻⁸ on N₀ = 1.26×10⁻⁶, i.e. σ(|V|²) = 0.016 per telescope pair over the full
4.25 h ε Ori data set (B = 1.50). With A = 110 m², α = 0.30, b_el = 125 MHz and F = σ_spec = 1
this gives q = 0.093 (an effective value: it carries the ×2 loss they attribute to the
uncollimated filter, the mirror reflectivity, F and σ_spec). Their 17-minute measurement
unit is used as the block length.

**MAGIC check.** With their constants the formula gives, at |V|² = 1, S/N ≈ 48 in 10 h for a
B = 4.0 star, consistent with their statement that stars to ∼4 B mag are realistic.

## Spica on VERITAS (`scripts/spica_veritas.py`)

hbtsim's `spica` target (blackbody + Spica linear limb darkening, anchored g and i; circular
orbit, a = 1.71 mas, θ = 0.91 / 0.45 mas) at 416 nm, AB = 0.64, averaged over the 12 m
pupils, for eight nights spread over the 4.01-day orbit; FLWO sees Spica for 5.9 h above
30° (H = ±2.93 h); 21 blocks of 17 min per night.

- Separation 0.77–1.71 mas → fringe period 50–111 m, inside the 43–173 m projected baselines:
  VERITAS resolves the binary fringe itself. The 12 m pupils keep 0.87 of its contrast.
- σ(|V|²) = 0.031 per pair per 17 min, 0.016 per hour; 2.6 h to 0.01.
- Per pair over the orbit: T3–T4 (43–82 m) |V|² swings 0.05–0.61, T2–T4 0.01–0.51,
  T1–T2 0.03–0.36; the swing is 20–35 σ(1 h) on the four shorter pairs and ∼1 σ on T1–T4
  (121–173 m, past the first null of Spica A).
- On a fixed baseline at transit, the orbit modulates |V|² by up to 0.09 (91 m) with the
  4-day period (right panel of the figure): a 3σ effect per 17-min block, ∼15σ per night.
- Orbit caveat: Raiola et al. 2025 use the Herbison-Evans 1971 solution (a = 1.54 mas,
  θ = 0.90 / 0.40 mas, brightness ratio 6.4) with e = 0.133 from Tkachenko et al. 2016;
  hbtsim's a = 1.71 mas follows from the masses, period and 76.6 pc. Which orbit the VERITAS
  data prefer is itself a result.

## γ Cas on MAGIC + LST-1 (`scripts/gammacas_magic.py`)

γ Cas (B0.5 IVe, B = 2.29, V = 2.39, 25 000 K, log g 3.5, 168 pc) as `single.GAMMA_CAS`: a
blackbody with Spica's linear limb darkening (beyond the NewEra grid), anchored to B and V.
Models at 425 nm: the MAGIC circular fit (θ_LD = 0.532 mas; θ_UD 0.515) and the VERITAS 2025
ellipse (minor 0.43 mas, axis ratio 1.28, PA 116°), each with a Gaussian disk of FWHM 2.9 mas
(the literature envelope) carrying a fraction f of the 425 nm flux. One night from ORM
(10.6 h above 30°), 20-minute blocks.

- MAGIC's formula: σ(|V|²) = 0.023 per 20 min (0.013 per hour) on MAGIC-I–II, 0.018 (0.010)
  on the LST-1 pairs.
- The projected baselines are 53–92 m; |V|² runs 0.46–0.79. The rotating baselines sweep
  position angles 15–165° (I–II) and −78…+78° (LST-1 pairs), so an oblate star shows as a
  PA-dependent spread of up to ±0.1 in |V|² at the same projected length (figure, left).
- Fisher estimate with the major axis, axis ratio and PA free: one night gives
  σ(axis ratio) = 0.011, σ(PA) = 1°, σ(θ_major) = 0.003 mas, statistics only, with the assumed
  layout orientation. That is the floor behind their preliminary 0.76 ± 0.02 (VERITAS:
  ± 0.04 ± 0.02 from 160 pair-hours).
- **The disk hides in the normalization.** A 2.9 mas Gaussian is fully resolved on every
  baseline here, so it multiplies |V|² by (1 − f)² at all baselines, exactly what the free
  zero-baseline normalization N₀ in the groups' fits absorbs: the fitted diameter is
  unchanged (0.517 mas for f = 0, 0.1, 0.2) while N₀ drops to 0.80 and 0.63. A disk fraction
  is therefore degenerate with the calibration unless N₀ is fixed by a calibrator or the
  array reaches baselines short enough (≲ 40 m) to see the disk partly coherent.
- Not modeled: gravity darkening (the VERITAS Roche–von Zeipel fit gives 0.604 mas at the
  equator), v sin i = 389 km/s; f at 425 nm is unknown (the disk dominates in Hα).

## Assumptions to confirm with the groups

VERITAS mirror area and effective q; LST-1 mirror area, detector and filter; the MAGIC /
LST-1 triangle's orientation; the disk fraction at 425 nm; Spica's orbit solution.
