# The Hyades giants on Cherenkov-telescope arrays: what MAGIC + LST-1 and CTAO-North would add to CHARA

> **Updated 2026-11-02.** The real giant models (log g 2.5) have replaced the stand-ins, and the
> profile-separation statistic quoted below ("2.2–2.9 σ per night") was a fitting artifact: the
> comparison fit did not pupil-average its model while the simulated track did, so even an identical
> profile "differed" at 2.5 σ. The fit is fixed; the corrected numbers and the comparison with the
> real models are in [the 2026-11-02 section](#2026-11-02-the-real-giant-models-and-a-corrected-profile-test)
> at the end. The October tables are kept as written, with their compact-vs-extended columns superseded.

Written 2026-10-28 for the France-Berkeley Fund proposal work (`~/proposals/fbf`),
following `handoff_hyades_limb_darkening.md`. Campaigns `iact_hyades_magic_lst1` and
`iact_hyades_ctao_alpha` (runner `iact`, `hbtsim/runners/iact.py`); targets `eps_tau`,
`gam_tau`, `del1_tau`, `tet1_tau`; tests `tests/test_runner_iact.py`. This page covers the
intensity-interferometry part of the handoff; its `scale`-runner deliverables (θ_UD/θ_LD
per band, the chromatic θ_UD(λ), the Kim & Kaiser 4 m pair) are still to do.

## The question

CHARA Classic measured all four Hyades red-clump giants in the K′ band in one night of
2007 (Boyajian et al. 2009, ApJ 691, 1243): limb-darkened diameters to 1.3–1.9 %, from
the main lobe of the visibility curve on 120–156 m baselines at 2.2 µm, with the
limb-darkening correction taken from a model (linear coefficients from Claret). What
would photon-counting intensity interferometry at 425 nm on the La Palma Cherenkov
telescopes add? Two things in principle: a diameter in the blue, where limb darkening is
three times stronger than in K′ and the UD-to-LD correction correspondingly more
model-dependent, and a direct test of the limb-darkening profile from the sidelobes,
which CHARA's main-lobe data do not reach.

| star | sp. type | V | B | d [pc], Gaia DR3 | θ_LD CHARA K′ [mas] | θ_UD CHARA | first null at 425 nm |
|---|---|---|---|---|---|---|---|
| ε Tau (Ain) | G9.5 III | 3.53 | 4.54 | 44.71 | 2.733 ± 0.031 | 2.659 | 44.2 m |
| γ Tau | K0 III | 3.65 | 4.64 | 46.16 | 2.517 ± 0.034 | 2.452 | 48.1 m |
| δ¹ Tau | K0 III | 3.76 | 4.74 | 49.24 | 2.408 ± 0.038 | 2.347 | 50.3 m |
| θ¹ Tau | K0 IIIb | 3.84 | 4.78 | 40.37 | 2.305 ± 0.043 | 2.247 | 52.4 m |

The catalog carries the CHARA diameters (ε Tau's former 2.59 mas, from a radius and the
parallax, is 5 % smaller than CHARA's; both are kept in its sources), B V H K anchors,
and an `atmosphere` pointer at the NewEra red-clump resource.

## The models

No NewEra PHOENIX model exists at the giants' gravity (log g ≈ 2.6 at 4800–5000 K). The
giants are therefore **bracketed**, as the handoff prescribes, not modeled:

| stand-in | role | drawn / τ=1 radius | second-lobe peak |V|² at 425 nm |
|---|---|---|---|
| `lte05000-4.50-0.0` | compact limit (dwarf) | 1.0017 | 0.0078 |
| `lte05000-0.00-0.0` | extended limit (supergiant) | 1.165 | 0.0057 |
| `lte05000-4.50+0.5` | metallicity lever, dwarf at [M/H] +0.5 | 1.0016 | 0.0075 |

The campaigns draw the giants with the compact profile and simulate the same nights with
the other two (`options.compare_models`): the runner fits the compact profile, with the
diameter and the zero-baseline normalization free, to each alternative's noiseless |V|²
track and reports the residual χ² per night, i.e. the significance with which a night
separates the profiles after the diameter has absorbed what it can. The uniform-disk
second lobe of a 2.7 mas star peaks at 0.0175; the compact PHOENIX profile halves that
and the extended one lowers it by a further 27 %. The flux at 425 nm is the anchored
PHOENIX spectrum: 0.43 mag fainter than a blackbody anchored to B and V, because the
B anchor at 445 nm is held constant blueward and the K-giant spectrum keeps falling.

## MAGIC + LST-1 (`iact_hyades_magic_lst1`): ten nights

Every projected baseline of the surveyed array (43–107 m) lies past the first null for
all four stars; only the short end of MAGIC-II–LST-1 (43–55 m) climbs the foot of the
main lobe for the three smaller stars (|V|² to 0.1–0.24). With MAGIC's analog
sensitivity, LST-1's area and QE, and the PHOENIX flux at AB 5.1–5.3:

| star | |V|² on the track | S/N per night | σ(θ)/θ after 10 nights | nights to 10 % | compact vs extended (superseded, see 2026-11-02) |
|---|---|---|---|---|---|
| ε Tau | 0.0003–0.10 | 1.2 | 36 % | 127 | 0.11 σ per night |
| γ Tau | 0.0006–0.19 | 2.1 | 20 % | 38 | 0.11 σ |
| δ¹ Tau | 0.0007–0.19 | 2.0 | 19 % | 37 | 0.09 σ |
| θ¹ Tau | 0.0007–0.24 | 2.5 | 15 % | 21 | 0.09 σ |

Statistics only, and already decisive: MAGIC + LST-1 does not detect the sidelobes of a
Hyades giant in a night (σ(|V|²) per 20-min block 0.011–0.023 against lobes of 0.006–
0.008), and the diameter it gets after ten nights is ten times worse than CHARA's. The
sign is instructive: a 1 % precision would need thousands of nights, and the profile test
hundreds. The analog chain has one 26 nm channel, so spectral multiplexing is not
available to it. The Hyades giants are the wrong targets for the present La Palma
trio: they are resolved past the null on its shortest baseline.

## CTAO-North alpha (`iact_hyades_ctao_alpha`): three nights

The 13-telescope array (68–560 m) samples the second to the twelfth lobe. At Stage I
(camera PMTs, 26 nm, 400 MHz) the per-night S/N is 0.5–1.0 and σ(θ)/θ 18–34 % after three
nights: no better than MAGIC + LST-1, because every pair sits where |V|² < 0.02. Stage II
(dedicated photodetectors, 2 GHz, 1 nm × 10 channels, 9.6× per pair) changes the picture:

| star | S/N per night | σ(θ)/θ after 3 nights | nights to 1 % | compact vs extended (superseded) | compact vs +0.5 dex (superseded) |
|---|---|---|---|---|---|
| ε Tau | 5.0 | 3.6 % | 38 | 2.2 σ per night | 2.7 σ |
| γ Tau | 6.8 | 2.4 % | 17 | 2.7 σ | 2.9 σ |
| δ¹ Tau | 7.4 | 2.1 % | 14 | 2.7 σ | 2.8 σ |
| θ¹ Tau | 9.5 | 1.9 % | 10 | 2.7 σ | 2.8 σ |

At Stage II a week on each star reaches CHARA's precision on the diameter at 425 nm, and
one or two nights separate the compact from the extended profile at 3 σ, which is the
measurement CHARA cannot make. The caveats are the Stage II constants themselves (the
2026 talk's targets, not an instrument), the stand-in models (a real giant model will
sit between the two and the separation between it and either bracket is smaller than
between the brackets), and the systematics of fitting sidelobes an order of magnitude
below the calibration level with the normalization free: the fitted diameter when the
wrong profile is assumed is off by 10–30 %, which says the sidelobe measurement is as
much a profile measurement as a diameter one.

## Verdict against CHARA

MAGIC + LST-1 adds nothing to Boyajian et al. 2009 on these stars; the simulation says so
at the 1–2 σ-per-night level with no room for a factor of ten. The CTAO-North alpha array
at Stage I is in the same position. Stage II would make the Hyades giants a limb-darkening
experiment in the blue, independent of CHARA's K′ model correction, at about a week per
star. If the aim is a blue diameter of a red-clump star on the present La Palma telescopes,
pick stars whose first null lies beyond 110 m at 425 nm, i.e. θ < 1 mas: the red-clump
targets of Kim & Kaiser (HD 17652, 1.8 mas; HD 360, 0.9 mas) and their northern
counterparts are the ones to simulate next.

## 2026-11-02: the real giant models and a corrected profile test

Peter Hauschildt's cool-giant box (4500–5500 K × log g 2.0–3.5 at [M/H] 0, the 4900/5000 K ×
2.5/3.0 corners at +0.5, 4800 K / 2.5 at −0.5; 49 HSR-RF models, binned into the
`newera_redclump` resource) arrived on 2026-11-01. The catalog targets now carry their own
models (ε Tau `lte04900-2.50-0.0`; γ, δ¹, θ¹ Tau `lte04800-2.50-0.0`, the grid node nearest
each star's 4810–4880 K and log g 2.6–2.7), and the campaigns `iact_hyades_magic_lst1_giants`
and `iact_hyades_ctao_alpha_giants` repeat the October runs with them. Their `compare_models`
now bracket the gravity (log g 2.0 and 3.0 at 4800 K, i.e. ±0.5 dex around the star), the
metallicity (4900 K / 2.5 at +0.5; the Hyades are at +0.15) and repeat the two October
stand-ins. The October campaigns were rerun unchanged with the fixed fit so that the two
sets of numbers are comparable.

### What the real profile looks like

At 425 nm the giant is markedly darker at the limb than the dwarf stand-in and far less
extreme than the supergiant; the gravity and metallicity neighbours are close to it:

| model | role | I(0.5)/I(1) | I(0.2)/I(1) | I(0.1)/I(1) | R_out / R_τ=1 |
|---|---|---|---|---|---|
| `lte04800-2.00-0.0` | log g −0.5 dex | 0.477 | 0.139 | 0.000 | 1.0205 |
| `lte04800-2.50-0.0` | **the giant** (γ, δ¹, θ¹ Tau) | 0.496 | 0.219 | 0.000 | 1.0134 |
| `lte04900-2.50-0.0` | **the giant** (ε Tau) | 0.500 | 0.222 | 0.000 | 1.0136 |
| `lte04800-3.00-0.0` | log g +0.5 dex | 0.513 | 0.259 | 0.009 | 1.0073 |
| `lte04900-2.50+0.5` | [M/H] +0.5 | 0.499 | 0.221 | 0.000 | 1.0145 |
| `lte05000-4.50-0.0` | October compact stand-in | 0.558 | 0.325 | 0.238 | 1.0017 |
| `lte05000-0.00-0.0` | October extended stand-in | 0.133 | 0.000 | 0.000 | 1.1650 |

The drawn disk of the giant is 1.3 % larger than its τ = 1 disk (the dwarf's was 0.2 %, the
supergiant's 16.5 %), its first null moves out by 0.3–0.4 m, and its second lobe peaks at
|V|² = 0.0073 instead of the dwarf's 0.0078. The anchored PHOENIX flux at 425 nm is 0.06 mag
fainter than the dwarf stand-in's (AB 5.13 instead of 5.07 for ε Tau).

### Diameter: 3–8 % less signal, same verdicts

| array, backend | quantity | stand-in (Oct) | real giant |
|---|---|---|---|
| MAGIC + LST-1 | S/N per night, ε / γ / δ¹ / θ¹ Tau | 1.24 / 2.14 / 2.00 / 2.49 | 1.21 / 2.02 / 1.90 / 2.35 |
| MAGIC + LST-1 | σ(θ)/θ after 10 nights | 36 / 20 / 19 / 15 % | 37 / 20 / 20 / 15 % |
| CTAO-N alpha, Stage I | σ(θ)/θ after 3 nights | 34 / 23 / 20 / 18 % | 37 / 26 / 23 / 20 % |
| CTAO-N alpha, Stage II | S/N per night | 5.0 / 6.8 / 7.4 / 9.5 | 4.6 / 6.2 / 6.9 / 8.9 |
| CTAO-N alpha, Stage II | σ(θ)/θ after 3 nights | 3.6 / 2.4 / 2.1 / 1.9 % | 3.9 / 2.7 / 2.4 / 2.1 % |
| CTAO-N alpha, Stage II | nights to 1 % | 38 / 17 / 13 / 10 | 45 / 21 / 17 / 13 |

The lower blue flux and the 6 % lower sidelobes cost 3–8 % in S/N, i.e. 15–30 % more nights
to a given precision. Nothing in the October verdicts moves: MAGIC + LST-1 and Stage I do not
reach the sidelobes in a night; Stage II reaches 1 % in two to seven weeks per star.

### The profile test, corrected

The October claim that Stage II separates the compact from the extended profile at 2.2–2.9 σ
per night does not survive. `_fit_theta` in the `iact` runner fitted a model without pupil
averaging to a track computed with it, so the residual was the pupil mismatch, not the
profile: an alternative identical to the campaign profile "differed" at 2.5 σ with a −11 %
diameter bias. With the fit pupil-averaged like the track (commit of 2026-11-02), the
identical-profile control gives 0.0 σ and +0.1 %, and the real numbers are:

| CTAO-N alpha, Stage II, real giant vs … | max Δ|V|² on the track | σ per night, ε / γ / δ¹ / θ¹ Tau | nights to 3 σ | θ bias if that profile is assumed |
|---|---|---|---|---|
| log g 2.0 (−0.5 dex) | 0.0003–0.0006 | 0.07 / 0.08 / 0.09 / 0.10 | 900–1700 | −0.5 % |
| log g 3.0 (+0.5 dex) | ≤ 0.0004 | 0.02 / 0.05 / 0.05 / 0.06 | 2800–22 000 | +0.1 to +0.3 % |
| [M/H] +0.5 | ≤ 0.0002 | 0.05 / 0.06 / 0.06 / 0.07 | 1800–3300 | −0.2 % |
| October compact stand-in (5000 K dwarf) | 0.0005–0.0020 | 0.14 / 0.22 / 0.23 / 0.26 | 130–490 | +0.8 to +1.4 % |
| October extended stand-in (5000 K supergiant) | 0.0012–0.0024 | 0.45 / 0.58 / 0.61 / 0.70 | 19–45 | −2.6 to −3.5 % |

The October bracket itself (dwarf vs supergiant, in the rerun stand-in campaign with the
fixed fit) is 0.6–1.0 σ per night at Stage II, 9–25 nights to 3 σ, not one or two. On
MAGIC + LST-1 every comparison is below 0.03 σ per night, and at Stage I below 0.05.

Two conclusions replace the October ones:

- **The blue diameter is robust to the model.** Assuming a profile 0.5 dex off in gravity
  biases θ by 0.5 %; the metallicity lever is 0.2 % at +0.5 dex (0.06 % scaled to the Hyades'
  +0.15). Using a dwarf instead of a giant would have biased it by about +1 %, a supergiant by
  −3 %. With a model at the right gravity to ±0.5 dex, Stage II's 1 % diameter in 13–45 nights
  is not limited by the limb-darkening model, which is the point of having the cool-giant box.
- **The sidelobe profile test is out of reach.** Separating log g 2.5 from 2.0 or 3.0 at 3 σ
  takes of order a thousand Stage II nights per star; the real-versus-dwarf case takes
  130–490. The sidelobes of a 2.3–2.7 mas star at 425 nm are too low (|V|² < 0.008) for the
  Cherenkov arrays to measure the shape of the profile; what they can measure is the diameter,
  and the model then supplies the shape. The "limb-darkening experiment in the blue" of the
  October verdict should be read as "a blue diameter whose model dependence is now small".

Outputs: `output/campaigns/iact_hyades_{magic_lst1,ctao_alpha}{,_giants}/` (all four rerun
2026-11-02).

## Not done from the handoff

The `scale`-runner campaigns (`hyades_ld_compact` / `_extended` / `_metal`) with the
θ_UD/θ_LD table per band, the |V|²(B) figure at 420–900 nm, the chromatic θ_UD(λ) at
R = 5000, the per-hour σ at the second lobe for the Kim & Kaiser 4 m pair and EON-SII, the
`g2_epstau_isi_line` rerun with the stand-ins, and the `newera_redclump.json` notes. The
tables load and the targets validate (`tests/test_runner_iact.py::test_iact_hyades_profiles`
checks the supergiant's 1.165 radius ratio through the runner).
