# Red-clump angular sizes with intensity interferometry: what hbtsim and NewEra add

Kim & Kaiser (2026, PASP 138, 044202) propose intensity interferometry as
an independent check of the VLTI/PIONIER angular diameters that calibrate
the red-clump surface-brightness–colour relation (and so the 1 % LMC
distance). Their fiducial case, HD 17652 (β For, V = 4.46, 1.835 mas)
with two 4 m telescopes, reaches a scale precision σ_s < 0.007 in a
2 h H-band exposure at ~100 m; HD 360 (V = 5.99, 0.906 mas) reaches
σ_s < 0.03. This note reproduces their numbers on hbtsim's photon
budget and adds what the package and the NewEra models bring:
`hbtsim/diameter.py`, `hbtsim/single.py` (`HD_17652`, `HD_360`),
`scripts/redclump_ii.py`; logs `output/logs/redclump_ii_{dwarf,supergiant}.txt`.

**Status of the models.** No NewEra model near 4800 K / log g 2.5 exists
yet: the cool-star box holds log g ≥ 4.0 at these temperatures, and
log g 0–2 only below 4000 K, plus one 5000 K / log g 0 supergiant.
Everything below uses labelled stand-ins that bracket the giant's
gravity: the 4800 K / log g 4.5 dwarf (right temperature, wrong
gravity) and the 5000 K / log g 0 supergiant. The request to Hamburg
(recorded in the phoenix `HANDOFF.md`) is [M/H] = 0, T_eff 4700, 4800,
4900 K × log g 2.0, 2.5, 3.0.

## 1. Their noise formula is our matched filter

Their σ(|V|²)⁻¹ = (dΓ/dν)(T/σt)^{1/2}(128π)^{-1/4} equals `snr.g2_snr`
with p₂ = ½ (unpolarized light) and a Gaussian pair kernel of width
√2 σt, to 5 × 10⁻⁸ (`tests/test_diameter.py`). Their instrument
(throughput 0.3 for the whole chain, 42.4 ps FWHM jitter, no dark
counts, no readout ceiling) is `diameter.KK_TELESCOPE` / `KK_DETECTOR`.

**Reproduction** (2 h, one top-hat filter, 4 m pupils averaged,
baseline scanned 20–300 m, dwarf stand-in profile):

| star | V | R | I | H | K | paper |
|---|---|---|---|---|---|---|
| HD 17652 | 0.043 (45 m) | 0.025 (50 m) | 0.017 (60 m) | **0.0061 (120 m)** | 0.0067 (160 m) | < 0.007 at ~100 m (H) |
| HD 360 | 0.18 (85 m) | 0.10 (100 m) | 0.067 (125 m) | **0.025 (245 m)** | 0.028 (300 m) | < 0.03 (H) |

The optimum sits at |V|² ≈ 0.33 (x ≈ 2.0). The supergiant profile
changes these by < 3 %.

## 2. The radius convention

The scale parameter s stretches a model profile normalized to a
reference diameter; which radius that reference is matters at the
per-cent level for a giant. From the binned NewEra tables:

| model | R_out/R_τ=1 | R_out/R_limb | θ_UD/θ_LD (V) | θ_UD/θ_LD (H) |
|---|---|---|---|---|
| 4800 K, log g 4.5 (dwarf) | 1.0014 | 1.0014 | 0.927 | 0.971 |
| 5000 K, log g 0 (supergiant) | 1.165 | 1.0076 | 0.924 | 0.963 |

- For the dwarf the three radii coincide to 0.14 %.
- For the supergiant the outer boundary lies 16.5 % beyond the
  Rosseland τ = 1 radius, while the apparent limb (where the continuum
  intensity halves) lies only 0.76 % inside the boundary: the τ = 1
  radius is 13.5 % *inside* the visible limb.
- The uniform-disk diameter, the quantity a single-baseline |V|²
  returns, is 3–4 % below θ_LD in H and 7–8 % below in V, and the
  profile shape (dwarf vs supergiant) moves it by 0.3 % in V and 0.8 %
  in H.

The red-clump giant lies between the rows. Until its model exists the
size of the τ = 1 / limb offset at log g 2.5 is unknown, and it is
plausibly a few per cent: comparable to the 1 % goal and larger than the
statistical errors. Whatever PIONIER's "limb-darkened diameter" is
tied to (the SATLAS reference radius), an intensity-interferometric
validation must fit the same profile with the same radius definition,
or the comparison tests conventions rather than stars.

## 3. Multiplexing: many optical channels against one H filter

(2 h, two 4 m telescopes, baseline scanned; "hours to 0.007" scales as
1/T from the 2 h result.)

| backend | HD 17652: σ_s (B) → hours to 0.007 | HD 360 |
|---|---|---|
| KK detector, one H filter | 0.0061 (120 m) → 1.5 h | 0.025 (245 m) → 26 h |
| KK detector (PDE 1, 42 ps), 1000 ch 400–950 nm | 0.0007 (60 m) → 0.02 h | 0.0029 (120 m) → 0.33 h |
| SPAD Lambda 320 ch, time-tag (10⁸ cps) | 0.10 → 430 h | 0.10 → 430 h |
| SPAD Lambda 320 ch, correlator | 0.021 → 18 h | 0.084 → 290 h |
| R = 5000 (4325 ch), correlator | 0.0056 (50 m) → 1.3 h | 0.023 (95 m) → 21 h |
| R = 5000, correlator + PBS | 0.0040 → 0.64 h | 0.016 → 11 h |
| EON-SII 1000 ch 400–550 nm, QUASAR SPAD (1 GHz link) | **0.0036 (40 m) → 0.52 h** | 0.015 (80 m) → 8.7 h |

- With an *ideal* optical detector (their throughput, unit PDE, 42 ps
  jitter) 1000 channels beat the single H filter by ×9 in σ_s, i.e. ×76
  in time: their "factor ~100" is right for that assumption.
- With *real* detectors the gain shrinks to parity or a factor of a
  few: the R = 5000 SPAD Lambda correlator backend matches the H
  filter, EON-SII's 1000-channel SPAD beats it by ×1.7 in σ_s (×3 in
  time) on HD 17652 and ×1.7 on HD 360, and the current 320-channel
  time-tag detector is hopeless (link-limited ×0.2, 120 ps jitter).
- For HD 360 the H filter needs 245 m; the optical channels do their
  work at 80–120 m, within the PIONIER-like baselines they want.

So multiplexed optical intensity interferometry is at least competitive
with their H-band case, and it carries the chromatic information of §4
for free. The H-band case itself needs an infrared photon counter with
picosecond jitter that hbtsim does not yet model (InGaAs SPAD, SNSPD).

## 4. Chromaticity at R = 5000

θ_UD/θ_LD across 400–950 nm at the optical-best baseline (dwarf
stand-in; figure `output/redclump_hd17652_dwarf_chromatic.png`):

| 400–450 | 450–500 | 500–550 | 550–650 | 650–800 | 800–950 nm |
|---|---|---|---|---|---|
| 0.891 | 0.903 | 0.917 | 0.929 | 0.944 | 0.955 |

A 6 % run across the band, with the strong lines (Ca II H and K, the
G band, Mg b, Hα, the Ca II triplet) 2–4 % above the neighbouring
continuum. Per channel σ(θ_UD)/θ is 0.45 in 2 h on HD 17652 (1.8 on
HD 360), so 50 nm bins of ~400 channels reach ~2 % per 2 h and the
continuum slope is a ~3σ measurement per 2 h, ~7σ per night. That is
the limb-darkening chromaticity of a K giant measured rather than
assumed, the test the authors say G/K atmospheres need.

## 5. Caveats

- **Stand-in profiles.** Every profile-dependent number above is for
  a dwarf or a supergiant of the right temperature. The gravity
  dependence between them is small for the UD ratios (< 1 %) and large
  for the radius convention (0.1 % → 16 %); the giant's own values are
  the point of the model request.
- **Photometry.** With θ_LD and the stand-in fluxes the model V is
  0.18 mag brighter than observed (0.34 with the supergiant's larger
  drawn disk); the giant model and its metallicity will move this.
- **Not modeled:** baseline projection over the 2 h, atmospheric
  extinction, sky (negligible for V ≈ 4–6), an H-band detector. All
  are hooks that exist in hbtsim except the last.
- **Not fitted:** the Cramér–Rao bound assumes the accidental level is
  known; the Monte Carlo of `hbtsim.montecarlo` can check the
  attainable precision including the singles-based normalization.
