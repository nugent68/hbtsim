# Cherenkov-telescope intensity interferometers in hbtsim: VERITAS, MAGIC + LST-1, CTAO-North

Made for the SII Workshop 2026 talk (6 October): one slide each applying hbtsim to the
targets two groups present there (M. Lisa: five years of Spica on VERITAS; I. Jiménez:
γ Cas on MAGIC + LST-1). Module `hbtsim/iact.py`; scripts `scripts/spica_veritas.py`,
`scripts/gammacas_magic.py`; logs `output/logs/{spica_veritas,gammacas_magic}.txt`;
figures `output/{spica_veritas,gammacas_magic}.png`; tests `tests/test_iact.py`.
Extended 2026-10-28 with the CTAO-North telescopes, layouts and the two CTAO-SII
instrument stages from J. Biteau's talks (MST-N SII, 2024-09-13, with Q. Luce; Stellar II
with the CTAO, La Palma, 2026-10-08), and with the surveyed MAGIC / LST-1 positions.

## What hbtsim supplies, and what it doesn't

These instruments digitize photomultiplier currents at 250–500 MS/s and correlate the
streams. They are analog, nanosecond systems, not photon counters, so their sensitivity
is taken from their own published form (MAGIC, Abe et al. 2024, Eq. 4; the same expression
in Raiola et al. 2025):

    S/N = A α q n_ν |V|² √b_el σ_spec / (√2 F (1 + β)) √T

with their constants (`iact.IACTBackend`). A mixed array (MAGIC + LST-1, LSTs + MSTs) has a
different A, α and q per telescope; a backend with `per_station: true` takes q from each
station's telescope throughput and α from its detector PDE (unless the backend fixes α),
and the pair carries √(A_i α_i q_i A_j α_j q_j), the product of the two photon rates;
`n_channels` adds √N (the CTAO Stage II multi-channel setup). hbtsim supplies only the visibility model: the
analytic two-disk binary on arbitrary projected baselines, limb-darkened (and now
elliptical) single disks, a Gaussian circumstellar component, and the averaging over the
dish apertures. Sites, layouts and the per-pair hour-angle loop are new (`iact.pair_track`).

## Parameters and sources

<!-- catalog:iact_parameters -->
|  | VERITAS | MAGIC | CTAO LST | CTAO MST |
|---|---|---|---|---|
| dishes | 4 × 12 m, 110 m² (**assumed**) | 2 × 17 m, 236 m² | 4 × 23 m, 370 m² (with shadowing) | 9 × 11.5 m, 88 m² (with shadowing) |
| site | Fred Lawrence Whipple Observatory (VERITAS) (31.675°, -110.952°, 1268 m) | Roque de los Muchachos (CTAO-North reference point; MAGIC, LST-1) (28.7622°, -17.892°, 2147 m) | same (CTAO-North reference point) | same |
| positions | read off Abeysekara et al. 2020, Fig. 1, fitted to the published baselines | ctapipe_io_magic CTA-frame constants | CTAO array-element position file (LSTN-01..04) | CTAO position file; the nine alpha pads matched to CTAO's layout map |
| baselines | 81.6, 99.4, 99.5, 108.9, 126.4, 172.6 m | MAGIC-I–II 85.3 m; LST-1 53.7 m from MAGIC-II, 105.9 m from MAGIC-I | LST–LST 103–179 m | alpha layout 68–560 m, 78 pairs |
| optical q (telescope throughput) | 0.3 (**calibrated** in the backend) | 0.304 | 0.64 at 420 nm | 0.73 at 420 nm |
| camera PMT QE α at 425 nm | 0.3 | 0.295 | 0.4 (2024 table: 0.39) | 0.4 |
| sources | Abeysekara et al. 2020 | Abe et al. 2024, Table 4 | Biteau & Luce 2024-09-13 (CTA-N table); Biteau 2026-10-08 (Stage I) | Biteau & Luce 2024-09-13 (CTA-N table) |
<!-- /catalog -->

<!-- catalog:iact_backends -->
|  | filter | QE α | optical q | b_el | F | σ_spec | channels | per station |
|---|---|---|---|---|---|---|---|---|
| VERITAS SII | 416 nm / 13 nm | 0.3 | **calibrated: 0.093** (file: 0.25) | 125 MHz (4 ns) | absorbed into q | absorbed into q | 1 | no |
| MAGIC SII (+ LST-1) | 425 nm / 26 nm | from the station's detector | from the station's telescope | 110 MHz (2 ns) | 1.15 | 0.87 | 1 | yes |
| CTAO-SII Stage I | 425 nm / 26 nm | from the station's detector | from the station's telescope | 400 MHz (2.5 ns) | 1.15 | 0.91 | 1 | yes |
| CTAO-SII Stage II | 425 nm / 1 nm | 0.45 | from the station's telescope | 2000 MHz (0.5 ns) | 1.05 | 1 | 10 | yes |
<!-- /catalog -->

Sources: Abeysekara et al. 2020, Nat. Astron. (arXiv:2007.10295); VERITAS γ Cas 2025, ApJ 995,
191 (arXiv:2506.15027); Abe et al. 2024, MNRAS 529, 4387 (arXiv:2402.04755); Raiola et al.
2025, PoS ICRC2025 957; Abe et al. 2025 (arXiv:2506.04994: LST-1 joined the MAGIC correlator in
2024; MAGIC + LST-1 ∼3.6× and MAGIC + 4 LSTs ∼11× the MAGIC-alone sensitivity); Biteau & Luce
2024-09-13, *SII with NectarCAM on the MSTs-N* (CTA-N table: LST 370 m², q 0.64; MST 88 m²,
q 0.73; PMT QE 39 %, F 1.21, 1.5 ns transit-time spread; 650 / 600 MHz maximum bandwidth;
simulations at 500 MHz); Biteau 2026-10-08, *Stellar intensity interferometry with the CTAO*
(Stage I: QE 0.40, ∼400 MHz, F 1.15, 26 nm, 1 channel; Stage II: QE 0.45, > 2 GHz, F 1.05,
1 nm, 10 channels; credit T. Hassan). Positions: the CTAO-North array-element position file
(CTAN_ArrayElements_Positions.ecsv, reference 28.762158° N, −17.892009° E, 2147 m; LSTN-01 =
LST-1 at 28.761526° N, −17.891497° E, 2185 m) and ctapipe_io_magic's CTA-frame MAGIC constants.

**The MAGIC / LST-1 geometry.** The two talks and the papers describe LST-1 as "∼100 m from the
MAGIC telescopes" with "two further baselines of approximately 100 m". The surveyed positions
(LST-1 from CTAO's file, MAGIC-I/II from the collaboration's software constants in the same
frame, 85.3 m apart as published) put LST-1 53.7 m from MAGIC-II and 105.9 m from MAGIC-I,
with MAGIC-I to the ENE of LST-1 and MAGIC-II to the ESE. The former hbtsim layout (two 100 m
baselines, orientation assumed) is gone; the MAGIC positions remain software constants rather
than a survey document, which CTAO says may shift by less than ten metres.

**VERITAS calibration.** The optical efficiency of the uncollimated-filter chain is not
published, so `iact.calibrate_q` solves for the q that reproduces their quoted precision:
σ(|g|²) = 2×10⁻⁸ on N₀ = 1.26×10⁻⁶, i.e. σ(|V|²) = 0.016 per telescope pair over the full
4.25 h ε Ori data set (B = 1.50). With A = 110 m², α = 0.30, b_el = 125 MHz and F = σ_spec = 1
this gives q = 0.093 (an effective value: it carries the ×2 loss they attribute to the
uncollimated filter, the mirror reflectivity, F and σ_spec). Their 17-minute measurement
unit is used as the block length.

**MAGIC check.** With their constants the formula gives, at |V|² = 1, S/N ≈ 48 in 10 h for a
B = 4.0 star, consistent with their statement that stars to ∼4 B mag are realistic. Adding
the pairs in quadrature, MAGIC + LST-1 (α 0.40, q 0.64, 370 m²) reaches 151, 3.2× MAGIC alone
(Abe et al. 2025 quote ∼3.6×), and MAGIC + the four LSTs 598, 12.5× (they quote ∼11×).

## Spica on VERITAS (`scripts/spica_veritas.py`)

hbtsim's `spica` target (blackbody + Spica linear limb darkening, anchored g and i; e = 0.108
since 2026-10, circular when this script was run; a = 1.71 mas, θ = 0.91 / 0.45 mas) at 416 nm, AB = 0.64, averaged over the 12 m
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

- MAGIC's formula: σ(|V|²) = 0.023 per 20 min (0.013 per hour) on MAGIC-I–II, 0.011 (0.0061)
  on the two LST-1 pairs (370 m², q 0.64, QE 0.40 against 236 m², 0.304, 0.295).
- The projected baselines are 45–82 m (MAGIC-I–II), 67–107 m (MAGIC-I–LST-1) and 43–55 m
  (MAGIC-II–LST-1); |V|² runs 0.34–0.84. The three pairs sweep different position angles, so
  an oblate star shows as a PA-dependent spread of up to ±0.1 in |V|² at the same projected
  length (figure, left).
- Fisher estimate with the major axis, axis ratio and PA free: one night gives
  σ(axis ratio) = 0.008, σ(PA) = 1°, σ(θ_major) = 0.002 mas, statistics only, on the surveyed
  layout. That is the floor behind their preliminary 0.76 ± 0.02 (VERITAS:
  ± 0.04 ± 0.02 from 160 pair-hours).
- **The disk hides in the normalization.** A 2.9 mas Gaussian is fully resolved on every
  baseline here, so it multiplies |V|² by (1 − f)² at all baselines, exactly what the free
  zero-baseline normalization N₀ in the groups' fits absorbs: the fitted diameter is
  unchanged (0.517 mas for f = 0, 0.1, 0.2) while N₀ drops to 0.80 and 0.63. A disk fraction
  is therefore degenerate with the calibration unless N₀ is fixed by a calibrator or the
  array reaches baselines short enough (≲ 40 m) to see the disk partly coherent.
- Not modeled: gravity darkening (the VERITAS Roche–von Zeipel fit gives 0.604 mas at the
  equator), v sin i = 389 km/s; f at 425 nm is unknown (the disk dominates in Hα).

## CTAO-North (`ctao_n_lst4`, `ctao_n_2027`, `ctao_n_alpha`, `magic_lst4`)

The CTAO-North arrays in the catalog, all on the surveyed pads from CTAO's position file
(east/north offsets from the reference point; the `up_m` of each pad, 10–65 m, is kept):

| array | stations | pairs | baselines | status |
|---|---|---|---|---|
| `ctao_n_lst4` | LST-1…4 | 6 | 103–179 m | LST-2…4 under construction, CTAO: end of 2025 |
| `ctao_n_2027` | LST-1…4 + MST-03 | 10 | 103–278 m | the 2026 talk's "intermediate config. '27": four LSTs and one MST west of them; **which MST pad is assumed** (MSTN-03 by eye) |
| `ctao_n_alpha` | LST-1…4 + nine MSTs (pads 01–07, 10, 15) | 78 | 68–560 m | the alpha configuration (4 LSTs + 9 MSTs); the nine pads identified against CTAO's released layout map, the other six are reserve pads |
| `magic_lst4` | MAGIC-I, II + LST-1…4 | 15 | 54–228 m | Abe et al. 2025's "MAGIC + 4 CTAO LSTs" |

The two instrument stages are backends: `ctao_sii_stage1` (the Cherenkov cameras' PMTs read
out through robotic 425/26 nm filters, one photodetector per camera, ∼400 MHz, F 1.15, a GPU
correlator already validated on LST-1 + MAGIC) and `ctao_sii_stage2` (dedicated photodetectors,
QE 0.45, > 2 GHz, F 1.05, focal-plane corrective optics with 1 nm filters in 10 channels). The
analog S/N does not depend on the filter width (n_ν is per Hz), so Stage II gains from the
bandwidth (√5), the noise factor and the ten channels (√10): a factor 9.6 per pair. Band, σ_spec
and β of Stage II are **assumed** (425 nm, 1.0, 0).

With γ Cas (AB 2.14 at 425 nm) and these constants, σ(|V|²) per hour at |V|² = 1 is 0.0015 /
0.0028 / 0.0054 on an LST–LST / LST–MST / MST–MST pair at Stage I and 0.0002 / 0.0003 / 0.0006 at
Stage II. One night of γ Cas on the alpha array (32 blocks of 20 min, 78 pairs) covers 48–563 m,
|V|² from 0.81 down through the first null (the 0.53 mas disk nulls at 200 m), with σ(|V|²) =
0.003–0.009 per block. Summed in quadrature over the pairs at |V|² = 1 for a B = 4 star in 10 h,
S/N is 1044 (4 LSTs), 1135 (2027), 1830 (alpha) at Stage I and 10 000, 10 900, 17 500 at Stage
II, against 48 for MAGIC alone and 151 for MAGIC + LST-1. Luce et al.'s simulations (2.5 h,
uniform disk with a free normalization): ∼1 % on θ_UD for B < 3 with the 4 LSTs and B < 4 with
4 LSTs + 9 MSTs; ∼170 northern stars at 1 % with the alpha array; limb darkening to 10 % for
B < 1.2. These are the groups' numbers, not reproduced here; the MST pairs are what carries a
0.1–0.2 mas star past its first null (slide 25: θ = 0.1 mas at B = 3 to ±0.005 mas with the LSTs,
±0.000 with the MSTs added).

Not in the catalog: CTAO-South (14 MSTs + 37 SSTs, Paranal), the SSTs' optical design for SII,
the NectarCAM 1 ON + 5 OFF pixel scheme, the 0.7 ns dish-shape spread of the MSTs, and the
Stage II corrective optics' throughput.

## Assumptions to confirm with the groups

VERITAS mirror area and effective q; MAGIC's positions relative to LST-1 (software constants);
the CTAO Stage II band, σ_spec and β; which MST pad the 2027 configuration uses; the disk
fraction at 425 nm; Spica's orbit solution.
