# Orbital parallaxes of β Aur and Algol with MAGIC + LST-1: a 30 h proposal

Written 2026-11-04 for the MAGIC + LST-1 cycle-22 call (deadline 2027-01-22) with
J. Biteau, T. Hassan, A. Kim, F. Izraelevitch and R. Kaiser. Campaigns
`iact_betaaur_magic_lst1`, `iact_algol_magic_lst1` and their `_free` variants (runner
`iact`); figures in `~/proposals/fbf/magic_deb/`; the simulation assumptions are listed in
`~/proposals/fbf/magic_lst1_assumptions.docx`.

## The schedule the simulation assumes

The MAGIC + LST-1 interferometry program runs in bright-moon periods, in short slots handed
out by the SII schedulers across all approved targets (the cycle-21 proposals simulate
20-minute observations; the calibration program budgets 5-minute runs; the 2024-25 log shows
35 to 1220 minutes per target per year). The campaigns therefore use **one-hour slots**
(three 20-minute blocks; `options.slot_hours`), each at a different hour angle, on nights at
listed **orbital phases** (`options.phases`: 0, 0.5, 0.1, 0.6, 0.9, 0.4, ..., with phase 0 the
quadrature and the eclipses at 0.25 and 0.75 avoided by more than 0.1 in phase). The slot's
track is computed at its own phase, so the rest of the night may be in eclipse. The request
is 18 h on β Aur and 12 h on Algol, category 1 (time-critical on days: a few hours every
second night), over the October-to-January full-moon periods.

The Moon never comes within 18° of either star (their declinations, +45° and +41°, exceed
the Moon's maximum), and the separation changes by about 12° per night inside a window, so
the ordering of nights matters more than the period: Algol on the late nights of the October
and December periods and the early nights of November, β Aur on the early nights of December
and anywhere in January (`fig3_moon.png`).

## Results (statistics only, zero-baseline normalization free)

The 40-55 m MAGIC-II–LST-1 pair sits inside the first fringe of both binaries (fringe periods
41 m and 61 m at 425 nm) at S/N 45-140 per one-hour slot; the two long pairs reach the fringe
minimum (`fig1_tracks.png`).

| target | hours | σ(a)/a, radii tied to the eclipse solution | radii free | radii free, 1 % / 2 % prior | existing distance |
|---|---|---|---|---|---|
| β Aur (A1 IV + A1 IV, a = 3.29 mas) | 18 × 1 h | 0.24 % | 0.26 % (radii to 3-5 %) | 0.24 / 0.24 % | Hipparcos 0.6 %; CHARA orbit (Jonak et al. 2026) 2.3 % away; Gaia none |
| Algol A–B (B8 V + K0 IV, a = 2.15 mas) | 12 × 1 h | 0.42 % | 5.7 % | 1.1 / 1.9 % | Hipparcos 3.9 %; CHARA/MIRC orbit; Gaia none |

With the double-lined spectroscopic a sin i (Southworth et al. 2007 for β Aur; the Algol SB2
solution), σ(d)/d equals σ(a)/a plus the spectroscopic error. β Aur's orbit is independent of
the stellar radii because the components are equal and nearly unresolved: the fringe is
100 % deep. Algol's is not: the K subgiant carries ~8 % of the blue light, so the fringe is
shallow and, over the 40-107 m baselines (less than one fringe period), the orbit's scale is
degenerate with the primary's diameter unless the eclipse radii are used, which for Algol
they can be (the archetype's radii are known to 1-2 %).

## Caveats

- Analog S/N constants for the LST-1 pairs (bandwidth, F, σ_spec) are MAGIC's; the LST-1
  optical efficiency comes from the MST-N table (see the assumptions document).
- The analytic binary visibility treats both components as limb-darkened disks; Algol B
  fills its Roche lobe. In-eclipse phases are not modelled (nor requested).
- No weather, moon-background (β = 0) or scheduling losses; the hour angles of the slots are
  spread by construction, which the schedulers would have to be asked for.
- Algol's orbital parallax rests on the eclipse radii; β Aur's does not.

## The spectroscopic a sin i (pulled 2026-11-04)

| system | K_1, K_2 [km/s] | source | a sin i → a [R_sun] | σ(a)/a from spectroscopy | best existing distance |
|---|---|---|---|---|---|
| β Aur | 108.053 ± 0.072, 110.911 ± 0.071 (formal; ×1.4 scaled) | Behr et al. 2011, AJ 142, 6 (dFTS, 20 double-lined RVs) | 17.60 R_sun with i = 76.80 ± 0.10° (Southworth et al. 2007) | 0.05 % (0.37 % with Smith 1948's K's, which Southworth 2007 used) | Jonák et al. 2026 (A&A, CHARA SPICA + MIRC-X + MYSTIC + RVs + light curves): 24.30 ± 0.05 pc, R = 2.752 / 2.622 R_sun; Hipparcos 25.18 ± 0.49 pc (1997) / 24.87 pc (2007) |
| Algol A–B | 44.1 ± 0.2, 194.2 ± 1.2 | Kolbas et al. 2015, MNRAS 451, 4150 (spectral disentangling); a = 13.65 ± 0.07 R_sun, i = 81.4 ± 0.2° (Richards 1988) | 13.66 R_sun | 0.51 % | Zavala et al. 2010 / Peterson et al. 2011 (VLBA, outer orbit): 34.7 ± 0.6 mas (1.7 %); Hipparcos 36.27 ± 1.40 mas (3.9 %); CHARA inner orbit a = 2.15 ± 0.05 mas (Baron et al. 2012) |

Radii: β Aur 2.762 ± 0.017 / 2.568 ± 0.017 R_sun (Southworth 2007; Jonák 2026 gives 2.752 ± 0.002 / 2.622 ± 0.002), Algol A 2.87 ± 0.04, B 3.43 ± 0.01 R_sun (Kolbas 2015; Baron 2012 from the angular diameters 2.73 ± 0.20 / 3.48 ± 0.28). Distance budgets: β Aur σ(d)/d = 0.24 % (SII) ⊕ 0.05 % (RV) = 0.25 %, matching the 0.2 % of the 2026 CHARA solution and testing the 2.3 % Hipparcos–CHARA offset at 8 σ; Algol σ(d)/d = 0.42 % ⊕ 0.51 % = 0.66 % with the eclipse radii, 1.1 % ⊕ 0.51 % = 1.2 % with a 1 % radius prior, against 1.7 % from the radio outer orbit.
