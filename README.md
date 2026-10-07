# hbtsim — Intensity interferometry of binary stars

Simulation of the Hanbury Brown–Twiss (HBT) effect in optical intensity
interferometry for bright binaries: **Beta Aurigae (Menkalinan)**,
**Algol (β Persei) A–B**, **Spica** and **δ Velorum Aa–Ab**. The code renders each binary
as a pair of limb-darkened stellar disks on a sky grid, computes the
complex visibility V(**u**) by an exact discrete Fourier transform of the
image at the sampled baselines (JAX, GPU-batched for spectral work),
predicts the signal-to-noise of real photon-counting
observations, and produces movies:

1. **Orbit movie** (`hbtsim movie --target {betaaur,algol,spica,deltavel}`) — three
   panels over one orbital period: the stars on the sky (false-color RGB
   from renders at 700/550/440 nm, Planck-weighted and white-balanced to
   7500 K, so hot stars look blue and cool ones orange; to-scale mas
   axes), the apparent g/i lightcurves anchored to observed photometry,
   and g²(B) at 400 and 800 nm for baselines 10–150 m.
2. **g²(λ) measurement movie** (`hbtsim g2spec`) — the stars on
   the sky beside the g² *spectrum* a real telescope pair would measure
   each hour: 320 spectral channels (400–950 nm) with 1σ error bars from
   the full photon budget, simulated noisy data over the true curve.

All simulations start at greatest projected separation (quadrature) and
follow the baseline along the projected separation axis by default
(`--baseline-pa` for a fixed instrumental orientation).

## The four systems

<!-- catalog:systems -->
|  | **Beta Aurigae** (`betaaur`) | **Algol A–B** (`algol`) | **Spica** (`spica`) | **δ Velorum Aa–Ab** (`deltavel`) |
|---|---|---|---|---|
| Components | A1m IV + A1m IV near-twins | B8V dwarf + K0IV subgiant | B1 III-IV + B2 V | A2 IV + A4 V rapid rotators |
| Masses | 2.376 / 2.291 M☉ | 3.17 / 0.7 M☉ | 11.43 / 7.21 M☉ | 2.43 / 2.27 M☉ |
| Radii | 2.762 / 2.568 R☉ | 2.73 / 3.48 R☉ | 7.47 / 3.74 R☉ | 2.97 / 2.52 R☉ |
| T_eff | 9350 / 9200 K | 12550 / 4900 K | 25300 / 20900 K | 9450 / 9830 K |
| Period | 3.96004 d | 2.867328 d | 4.0145 d | 45.1503 d, e = 0.29 |
| Inclination | 76.8° | 98.7° | 63.1° | 89.0° |
| Distance | 24.87 pc | 28.82 pc | 76.6 pc | 25.13 pc (orbital parallax) |
| Angular semi-major axis | 3.29 mas | 2.15 mas | 1.71 mas | 16.54 mas |
| Angular diameters | 1.03 / 0.96 mas | 0.88 / 1.12 mas | 0.91 / 0.45 mas | 1.10 / 0.93 mas |
| Ω (ascending node) | 295.15° | 43.43° | 131.6° | 65° |
| Eclipses | partial, ~0.08 mag | deep primary, **1.48 mag in g** | none | grazing |
| Anchored photometry | g 1.80, i 2.10 | g 2.07, i 2.58 (C-corrected) | g 0.71, i 1.06 | g 1.90, i 2.25 (B-corrected) |
| NewEra tables | both stars (interpolated) | A (clamped to 12 000 K); B blackbody | blackbody | both stars (interpolated) |
| Sources | Southworth et al. 2007; Hipparcos; Jonak et al. 2026 (Ω) | Baron et al. 2012 (CHARA); Zavala et al. 2010; Kolbas et al. 2015 | Herbison-Evans et al. 1971; Tkachenko et al. 2016 | Mérand et al. 2011 |
<!-- /catalog -->

The two systems probe complementary regimes. Beta Aurigae is the clean
textbook case: equal stars, equal colors, fringes of period λ/ρ ≈ 25 m
(400 nm) modulating the disk envelopes. Algol is the extreme-contrast
case: the hot dwarf supplies nearly all the blue light while the cool
subgiant matters only in the red, so g²(B, λ) is strongly
wavelength-dependent — nearly a single-star envelope at 400 nm but with
clear binary fringes at 800 nm — and the primary eclipse (the orange
subgiant covering ~75% of the blue dwarf's area) plunges the system by
1.5 mag while transforming the g² spectrum. Algol is also why limb
darkening is **per star** (`Star.ld_table_nm`, Claret & Bloemen 2011
linear coefficients): u(400 nm) = 0.42 for the B8V primary vs 0.89 for
the K subgiant.

Algol caveats: component C (~70 mas away) is excluded — the published
V = 2.12 and 1.27 mag eclipse depth include its ~10% third light, so the
model anchors and validation use C-corrected values (undiluted depth
≈ 1.5 mag, matched by the model's 1.48). The Roche-lobe-filling
secondary is rendered as a sphere, so the ellipsoidal variation and
reflection effect of the real out-of-eclipse lightcurve are absent.

**Spica** (α Vir, V = 0.97; Herbison-Evans et al. 1971 Narrabri
intensity-interferometer orbit, Tkachenko et al. 2016 disentangling) is
the recommended bright target for three-telescope closure-phase work
(true e = 0.108 approximated as circular; β Cep pulsations and tidal
distortion not modeled; non-eclipsing). **δ Vel** (Mérand et al. 2011:
eccentric 45-day orbit, both components rotating at ~145 km/s, not
modeled as oblate) sits at 25.1 pc, so its 16.6 mas orbit puts the blue
fringe period (4.7 m) below 8 m pupils: it is a target for 1–4 m
telescopes, not for the VLT. Its NewEra photometry comes out 0.24 mag
brighter than the observed A-only V, a tension in the published
parameters recorded in `tests/test_newera.py`.

Every target, telescope, detector, spectrograph, backend, site, array and
campaign is a JSON file under `hbtsim/configs/<kind>s/` (the table above
is rendered from them by `scripts/catalog_tables.py`). Adding a binary
is one file: copy `hbtsim/configs/targets/betaaur.json` to
`my_target.json` in a directory of your own, edit the numbers and the
`sources` / `notes`, and point the tools at it:

```bash
export HBTSIM_CONFIG_PATH=/path/to/my_configs      # or --config-dir on every command
hbtsim catalog validate                            # schema + build check, names the field on error
hbtsim snr --target my_target --instrument keck_pair
```

Files on the search path override shipped ones of the same name;
`extends` derives a definition from another (`spad_lambda_ng` is
`spad_lambda` with a correlator readout). `hbtsim catalog show target
deltavel` prints a definition with its provenance; `hbtsim catalog list`
the names; `from hbtsim import load_target, load_array` the Python side.

## Physics

For a chaotic (thermal) source, the Siegert relation links the measured
intensity correlation to the first-order coherence:

    g²(B) = 1 + |V(B)|²,

where V(B) is the complex degree of coherence, the normalized Fourier
transform of the sky brightness distribution at spatial frequency
**u** = **B**/λ (van Cittert–Zernike). Numerically:

    image I(θx, θy)  →  exact K-point DFT at (u, v) = B/λ (separable
                        matrix products, float32-exact phase)
                     →  V(u, v) / V(0, 0)  →  g²(B) = 1 + |V(B)|².

Each star is a limb-darkened disk whose centre-to-limb profile is
tabulated on a μ grid: by default the linear law I(μ)/I(1) = 1 − u_λ(1 − μ)
with per-star Claret & Bloemen (2011) coefficients, weighted by the
Planck function at its effective temperature; with `--newera-dir`
(`hbtsim.sed.NewEraGrid`, `with_newera`) the angle-resolved intensities
I(μ, λ) and surface fluxes of the NewEra PHOENIX models (Hauschildt et
al. 2025; `scripts/prepare_newera.py` bins an HSR-RF file to a 0.02 nm
table; `hbtsim data fetch` pulls the binned tables from NERSC, whose grid
covers 8000–12 000 K × log g 3.0–4.5), interpolated bilinearly in T_eff and
log g per star, which then set both the flux ratio of the two stars
(hence the fringe contrast) and their limb profiles, line by line.
Three things a spherical model needs are built in: its μ = 0 is the
model's outer boundary, drawn at R_outer = (1.004–1.008) × the τ = 1
radius the catalogue quotes (`Star.radius_ref`,
`BinarySystem.drawn_radius_mas`); each star renders on its own μ nodes
so the drawn profile is exactly the one `disk_flux_factor` integrates;
and the 0.02 nm tables are averaged over each spectrograph channel
(`sed.prepare_system`), optionally Doppler-shifted by the orbital
radial velocities (`BinarySystem.doppler`) and rotationally broadened
(`Star.vsini_kms`); interstellar reddening (`BinarySystem.a_v`, CCM89)
and an air-wavelength channel grid (`Spectrograph.frame`) are hooks
with defaults off. Eclipses are handled by z-ordering the
disks on the grid, which also yields the lightcurves by direct image
summation. Lightcurves are calibrated in two steps: synthetic
monochromatic AB magnitudes from the physical flux at Earth, then a
constant per-band shift anchoring maximum light to the observed
photometry (for blackbody photospheres the zero point is a few tenths
of a magnitude off — no line blanketing or H⁻ opacity; with model SEDs
the anchors only serve as a check, and eclipse shapes and depths are
purely simulated in both cases).

The g² = 1 + |V|² normalization is the ideal fully-coherent-detection
limit; a real intensity interferometer measures a contrast reduced by
the ratio of coherence time to detector resolution (Rai, Basak & Saha
2021, eq. 6) — that physics lives in the SNR module below.

Numerical layout: 1024² source grid at 0.01 mas/pixel (disk radii
~45–55 px, limb darkening well resolved); `GridConfig.fit_orbit()` (the
renderer default) grows the grid to hold an orbit (δ Vel's 16.6 mas
needs 2048 px), `GridConfig.for_system()` also refines the scale for
stars below 50 px, and
`GridConfig(supersample=4)` renders each pixel as the mean of 16
sub-pixel soft-rim renders (with the exact Dirichlet pixel window of
the s × s average removed from the DFT), which takes the renderer's
limb bias from ~1e-4 to ~1e-5 in
|V|² and the rendered-vs-analytic closure phase from 0.15° to 0.02°.
The visibility is *not*
taken from a padded FFT map: the interferometer only ever needs V at a
few points per channel, and the K-point DFT (`hbtsim/hbt.py`) gives them
exactly — no interpolation error, no crop limit on the baseline, ~1000×
fewer operations than the 8192² FFT it replaced (which survives in
`hbtsim/fftmap.py` for 2-D maps and cross-checks). The pipeline is
validated against analytic results (Airy nulls, limb-darkened Bessel
series, the binary fringe formula, circle-overlap eclipse depths) in the
test suite; the DFT core agrees with a float64 reference to 1e-6 in
|V|² and 1e-5 rad in phase.

## Observation SNR (hbtsim.snr)

`hbtsim/snr.py` estimates the signal-to-noise of a g² measurement with a
pair of telescopes and photon-counting detectors: detected stellar rates
from the anchored source model, excess coincidences
N_sig = ½|V|²·τ_c·R₁R₂·T for unpolarized light spread over the
detectors' combined timing jitter, matched-filtered against the
accidental floor (dark and sky counts included),

    SNR = ½ |V|² τ_c R₁ R₂ √T / √(b₁ b₂ · 2√π σ_pair),

with per-pixel non-paralyzable dead time. Everything is parameterized
via `Telescope`, `Detector`, `Observation` and `Spectrograph` dataclasses,
built from the catalog (`load_telescope("keck_10m")`). The instrument
model carries the effects a real implementation cannot escape:

- **Throughput** is telescope (atmosphere + optics, 0.3) × backend
  (0.9 for a narrow-band filter, **0.5 for a cross-dispersed
  spectrograph** — every dispersed channel sees 0.15, not 0.3) × PDE.
- **Constant resolving power**: `Spectrograph.from_resolving_power(5000)`
  builds the geometric channel grid an R ≈ 5000 spectrograph actually
  has (4325 channels of 0.08–0.19 nm over 400–950 nm), not 5500
  uniform 0.1 nm channels.
- **Readout**: a bright star dispersed over thousands of channels
  delivers 10¹⁰–10¹¹ detected photons/s per 8–10 m telescope, far
  beyond any time-tag link. `Detector.readout` is `"timetag"` (the
  SPAD Lambda as delivered, USB3, `max_total_cps` = 1.4 × 10⁸, the
  manufacturer's 140 Mcps — the spectral functions scale the rates
  down to the ceiling and flag `readout_limited`) or `"correlator"`
  (the next-generation design, `spad_lambda_ng`: a real-time correlator
  with no link limit, either both beams on one sensor or tags streamed
  to a central FPGA/GPU; an idealization, not an existing system). A
  per-pixel dead-time load r·τ_dead > 1 raises a warning (the
  non-paralyzable model is unreliable there; spread the light over
  more pixels).
- **Polarization**: `polarization_mode="pbs"` (a polarizing beamsplitter
  into two detectors per telescope) gains √2 in g² SNR and ×2 in g³ at
  the same photon budget while halving the per-pixel load;
  `"single_pol"` gains nothing; the default is unpolarized (p₂ = ½,
  p₃ = ¼).
- **Coherence broadening** of the correlation kernel (σ_c = 0.376 τ_c)
  — a ~1 % loss at 0.1 nm in the red, on by default.

<!-- catalog:hardware -->
Telescopes, detectors, spectrographs, sites and arrays are JSON files in
`hbtsim/configs/` (`hbtsim catalog list`); the shipped ones:

- **Telescopes**: `c2pu_1m` (1 m, throughput 0.3); `eonsii_4m` (4 m, 9 m², throughput 0.64); `gtc_10p4m` (10.4 m, 78.54 m², throughput 0.3); `int_2p54m` (2.54 m, throughput 0.3); `keck_10m` (10 m, throughput 0.3); `kk_4m` (4 m, throughput 0.3); `lst1_23m` (23 m, 390 m², throughput 0.304); `magic_17m` (17 m, 236 m², throughput 0.304); `not_2p56m` (2.56 m, throughput 0.3); `subaru_8p2m` (8.2 m, throughput 0.3); `tng_3p58m` (3.58 m, throughput 0.3); `veritas_12m` (12 m, 110 m², throughput 0.3); `vlt_ut_8p2m` (8.2 m, throughput 0.3); `wht_4p2m` (4.2 m, 13.8 m², throughput 0.3).
- **Detectors**: `eonsii_mcp_pmt` (46 ps FWHM, 1 ns dead, 1 cps dark, timetag, link ≤ 1e+09 cps); `eonsii_spad` (20 ps FWHM, 10 ns dead, 250 cps dark, timetag, link ≤ 1e+09 cps); `eonsii_spad_correlator` (20 ps FWHM, 10 ns dead, 250 cps dark, correlator); `kk_ideal` (39 ps FWHM, 0 ns dead, 0 cps dark, correlator); `lpqi_spad64_i2cass` (500 ps FWHM, 10 ns dead, 1.68 cps dark, timetag, link ≤ 6.7e+06 cps); `lpqi_spad_nextgen` (100 ps FWHM, 10 ns dead, 1.68 cps dark, timetag, link ≤ 4e+07 cps); `magic_pmt` (8540 ps FWHM, 0 ns dead, 0 cps dark, correlator); `spad_lambda` (120 ps FWHM, 10 ns dead, 250 cps dark, timetag, link ≤ 1.4e+08 cps); `spad_lambda_ng` (120 ps FWHM, 10 ns dead, 250 cps dark, correlator); `veritas_pmt` (7515 ps FWHM, 0 ns dead, 0 cps dark, correlator).
- **Spectrographs**: `eonsii_1000ch` (400–550 nm, 1000 ch, 0.15 nm channels, throughput 0.6); `eonsii_60ch` (400–550 nm, 60 ch, 2.50 nm channels, throughput 0.6); `eonsii_r7500` (400–550 nm, 2389 ch, R = 7500, throughput 0.6); `filter_2mass_h` (1480–1780 nm, 1 ch, 300.00 nm channels, throughput 1); `filter_2mass_k` (1995–2385 nm, 1 ch, 390.00 nm channels, throughput 1); `filter_cousins_i` (732–880 nm, 1 ch, 149.00 nm channels, throughput 1); `filter_cousins_r` (589–727 nm, 1 ch, 138.00 nm channels, throughput 1); `filter_johnson_b` (398–492 nm, 1 ch, 94.00 nm channels, throughput 1); `filter_johnson_v` (507–595 nm, 1 ch, 88.00 nm channels, throughput 1); `filter_lpqi_500_1nm` (500–500 nm, 1 ch, 1.00 nm channels, throughput 0.9); `filter_lpqi_550_1nm` (550–550 nm, 1 ch, 1.00 nm channels, throughput 0.9); `filter_lpqi_halpha_1nm` (656–657 nm, 1 ch, 1.00 nm channels, throughput 0.9); `filter_lpqi_hbeta_1nm` (486–487 nm, 1 ch, 1.00 nm channels, throughput 0.9); `hbeta_window_r5000` (471–502 nm, 320 ch, R = 5000, throughput 0.5); `kk_1000ch_400_950` (400–950 nm, 1000 ch, 0.55 nm channels, throughput 1); `r5000_400_950` (400–950 nm, 4325 ch, R = 5000, throughput 0.5); `spad_lambda_320` (400–950 nm, 320 ch, 1.72 nm channels, throughput 0.5).
- **Arrays**: `c2pu_pair` (2 stations, 15 m at Calern (C2PU)) (generator: `load_array(name, side_m=…)`); `eonsii_pair_teide` (2 stations, 100 m at Teide (Izana)) (generator: `load_array(name, side_m=…)`); `eonsii_triangle_paranal` (3 stations, 20–20 m at Paranal) (generator: `load_array(name, side_m=…)`); `keck_pair` (2 stations, 85 m at Maunakea); `lpqi_orm` (5 stations, 447–1547 m at Roque de los Muchachos, Nordic Optical Telescope (LPQI reference point)); `lpqi_pathfinder` (2 stations, 550 m at Roque de los Muchachos, Nordic Optical Telescope (LPQI reference point)) (generator: `load_array(name, side_m=…)`); `magic_lst1` (3 stations, 86–100 m at Roque de los Muchachos (MAGIC, LST-1)); `maunakea_subaru_keck` (3 stations, 85–226 m at Maunakea); `veritas` (4 stations, 82–173 m at Fred Lawrence Whipple Observatory (VERITAS)); `vlt_ut` (4 stations, 47–130 m at Paranal).
- **Sites**: `calern` (Calern (C2PU)); `flwo` (Fred Lawrence Whipple Observatory (VERITAS)); `maunakea` (Maunakea); `orm` (Roque de los Muchachos (MAGIC, LST-1)); `orm_not` (Roque de los Muchachos, Nordic Optical Telescope (LPQI reference point)); `paranal` (Paranal); `teide` (Teide (Izana)).
<!-- /catalog -->

Two observing modes (`hbtsim snr --target spica --instrument keck_pair`, or
`--telescope keck_10m --detector spad_lambda_ng --baseline 85`):

- **Spectral (default)** — the light is dispersed along the SPAD
  Lambda's 320-pixel array: each pixel pair is an independent ~1.7 nm
  channel measuring its own g², and channel SNRs add in quadrature
  (~√320 ≈ 18× multiplexing gain). Per-channel |V|²(B, λ) comes from the
  batched render + DFT pipeline (valid through eclipses), averaged over
  the two apertures; `--vis2-method analytic` is the instant
  out-of-eclipse alternative (agrees to 1e-3). `--resolving-power R`
  switches to a constant-R grid, `--readout correlator` lifts the
  time-tag ceiling, `--polarization pbs` adds the beamsplitter,
  `--n-pixels` spreads each channel over several pixels.
- **Narrowband** — single filters (`--mode narrowband --wavelengths 400
  800 --filter-width 10`).

Headline numbers for Beta Aurigae at quadrature, 1 h
(`hbtsim run g2_c2pu`, docs §5f): a single 10 nm filter on
C2PU gives SNR ≈ 0.1–0.9; the 320-channel SPAD Lambda with a correlator
readout lifts that to **≈ 15/√h at B = 35 m** (8 at 50 m) — the
time-tag link caps it at ≈ 7 — and an R = 5000 backend to 60 (54 with
the NewEra tables); the Keck pair at 85 m reaches 240/√h with 320
channels (σ_|V|² ≈ 0.02 per channel-hour) and the EON-SII 4 m pair at
30 m ≈ 570/√h with 1000 channels, both link-limited. The EON-SII
design study quotes 10–25× shorter times; `hbtsim run mc_sirius_b_eonsii`
reruns its Sirius B Monte Carlo and shows its hours need a coincidence
window of 2–6 ps, well below the detectors' 12–28 ps jitter (docs §5e.1).
The matched filter stays the default. On single stars
(`hbtsim run chromatic_sirius_vega_eonsii`, `docs/chromatic_diameters_eonsii.md`)
EON-SII measures Sirius's and Vega's Balmer-core diameters (+3.6 to
+4.7 % over the continuum at Hβ–Hδ) at 5σ in 1–2 nights through its
links, or in minutes with channel-subset tagging. Keck caveats: a
single SPAD pixel saturates at Keck count rates (`--n-pixels`), and a
10 m aperture on an 85 m baseline averages |V|² over B ± 10 m
(modeled: `hbtsim.aperture`).

The **LPQI-Pathfinder** (La Palma Quantum Interferometer: NOT 2.56 m +
TNG 3.58 m at 550 m, 64×64 SPAD cameras behind 1 nm filters, one filter per
night; later GTC, WHT and INT to 1.5 km) is in the catalog as
`arrays/lpqi_pathfinder`, `arrays/lpqi_orm`, the `lpqi_*` detectors,
filters and backends, with every unpublished number flagged;
[docs/lpqi_pathfinder.md](docs/lpqi_pathfinder.md) shows that at 550 m
all of the catalog's stars are resolved out and what the 1 nm / 500 ps
budget means for them (`hbtsim run g2_binaries_lpqi_pathfinder`).

## g²(λ) movies with error bars

`hbtsim g2spec --target algol --instrument keck_pair`
computes one frame per hour over a full orbit (69 epochs for Algol, 96
for Beta Aur): per-channel simulated measurements with 1σ error bars
(inverse-variance 8-channel bins highlighted) over the true model curve,
beside the temperature-colored sky view. During Algol's primary eclipse
the blue channels dim by 1.5 mag and the error bars visibly inflate
while the fringe spectrum collapses. Compute (`--compute-only`, GPU) and
rendering (`--render-only`, needs ffmpeg) are separable via the npz file.

## Three telescopes: bispectrum and closure phase

Two-point HBT gives only |V|² — no Fourier phase. With three telescopes
the triple correlation g³ = 1 + Σ|γᵢⱼ|² + 2|γ₁₂γ₂₃γ₃₁|cos φc carries the
**closure phase** (the bispectrum phase, immune to per-telescope phase
and source translation), the entry point to image reconstruction.
`hbtsim/bispectrum.py` samples complex visibilities from the rendered
image by the same exact DFT (validated against the analytic binary to
<0.1° in closure phase, renderer-limited), defines
telescope triangles (catalog arrays: `maunakea_subaru_keck` — Subaru +
Keck I + Keck II at 152/85/226 m, from site coordinates; `vlt_ut`), projects
them onto the (u, v) plane for a real hour angle and declination
(`Triangle.projected`, `hbtsim.geometry`), and computes closure phases
through eclipses via a GPU-batched spectral path.

Two effects a real correlator cannot avoid are modeled by default in
the SNR functions: **aperture smearing** — a 10 m pupil on an 85 m
baseline averages |V|² over B ± 10 m, which keeps only
A(πD₁ρ/λ)·A(πD₂ρ/λ) of a binary fringe's contrast (0.66 for Keck on
β Aur at 400 nm, 0.45 for the VLT UTs on δ Vel at maximum
separation); the triple product is averaged exactly over all three
pupils (`hbtsim.aperture`) — and **sky orientation**: every system
carries the position angle of its ascending node, so closure phases
on a fixed ground triangle are computed at the true orientation, and
`snr3.track_g3_snr` integrates a night block by block along the uv
track, flagging blocks over which the fringe drifts by more than 1/8
cycle.
`hbtsim/snr3.py` extends the photon budget to triple coincidences
(pol₃ = ¼, 2D lag-plane matched filter; SNR₃ ∝ 1/σ_jitter and
∝ Δλ^(−1/2) — narrowband multiplexing is the lever).

**Feasibility verdict** ([docs/three_telescope_feasibility.md](docs/three_telescope_feasibility.md),
`hbtsim run g3_{spica,deltavel}_vlt`, `g3_{betaaur,algol}_maunakea`): the Subaru arms resolve out the ~1 mas
disks (triple amplitude ≲ 0.04) and with the stock 320-channel SPAD
Lambda the closure phase of Algol needs **centuries**; an R ≈ 5000
backend with a correlator readout brings the template detection of
Algol's closure phase (Δcos φc ≤ 0.1) to **~200 nights** along the uv
track. Geometry beats aperture — and target selection rescues the real
triangle: **Spica** (`--system spica`; V = 0.97, hot small-disk B-star
pair whose primary diameter was itself measured by the Narrabri
intensity interferometer) reaches Δcos φc ≤ 0.1 in **~11 nights** on
Subaru + Keck I + Keck II with the R ≈ 5000 backend (2 with a
polarizing beamsplitter).

The best configuration studied is the **VLT 4×UT array**
(`bispectrum.VLT_UT`: 8.2 m × 4, baselines 46.6–130.2 m — all inside
Spica's first null, four simultaneous triangles + six |V|² baselines):
combined bispectrum sensitivity ≈ 13/√h (29 with a beamsplitter) on
Spica with the R ≈ 5000 backend → **Δcos φc ≤ 0.1 in ~36 minutes**
(7 with the beamsplitter), R = 100 closure-phase *curves* in 8.6 nights
(14 h) per epoch around the 4-day orbit, and a one-night limiting
magnitude of g ≈ 2.2 (southern targets only; Paranal cannot see
Algol/β Aur). δ Vel, at its correct distance of 25.1 pc, has a 4.7 m
blue fringe period at maximum separation that 8 m pupils smear out (it
returns near conjunction for ~21 % of the orbit). It is a 1–4 m-class g²
target (C2PU, EON-SII). Three EON-SII units on a 12 m triangle at
Paranal (`arrays/eonsii_triangle_paranal`, `snr3.campaign_g3_snr`,
`hbtsim run g3_deltavel_eonsii_paranal`) detect its closure-phase template in ~20
nights over an orbit. The asymmetric part of its closure phase
(φc ≠ 0, π) needs ~10⁴ nights on any array considered, because g³ sees
only cos φc (docs §5g).

## Batched spectral pipeline (CPU/GPU)

`hbtsim.spectral.spectral_vis(pos, bvecs, wavelengths, system, grid)`
(and the `spectral_vis2` wrapper for scalar baselines along a position
angle) renders the binary and evaluates its DFT at the baseline vectors
for every channel inside one jitted JAX computation; channels run in
chunks via `jax.lax.map(batch_size=...)` (renders and matrix products
batched per chunk, ~30 MB of temporaries per channel; default chunk 64
on GPU, 8 on CPU). Measured on an Intel iMac Pro (20 cores, CPU JAX):
**1.2 ms/channel** (320 channels in 0.37 s steady state), ~380× faster
than the padded-FFT version of this pipeline (460 ms/channel). On a
*shared* GPU set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (XLA's 75%
preallocation fights other users).

```bash
python scripts/bench_spectral.py                    # benchmark, default device
JAX_PLATFORMS=cpu python scripts/bench_spectral.py  # force CPU
```

## Campaigns (`hbtsim run`)

Every study behind the docs and the paper is a campaign file in
`hbtsim/configs/campaigns/`: the target(s), array or telescope, backends,
observing window and runner-specific options, resolved through the same
catalog as everything else. `hbtsim run <name>` (or a path to your own
`.json`) writes `output/campaigns/<name>/{results.json, table.md,
table.tex, log.txt, figures}`; `--set key=value` edits any field for one
run (`--set night.block_minutes=30`, `--set atmosphere.use=false`),
`--no-figures`, `--no-track`, `--newera-dir` / `--no-newera` as on the
other tools. `results.json` records the resolved campaign definition, its
content hash and the git revision, so a number in a table can always be
traced to its inputs. `suite_phase6` runs everything (`--jobs 4`).

<!-- catalog:campaigns -->
| Campaign | Runner | What it computes |
|---|---|---|
| `chromatic_c2pu_regression` | chromatic | Chromatic-diameter regression: Sirius A and Vega, 2 x 1 m, 320 ch at R = 5000 around H-beta, 6 h |
| `chromatic_sirius_vega_eonsii` | chromatic | Chromatic (wavelength-dependent) diameters of Sirius A and Vega with the EON-SII pair from Teide |
| `g2_binaries_lpqi_pathfinder` | g2 | The four binaries on the LPQI-Pathfinder (NOT + TNG, 550 m, 1 nm filters, one filter per night) |
| `g2_c2pu` | g2 | Two-telescope g2 sensitivity of the four binaries on the C2PU 1 m pair |
| `g2_eonsii` | g2 | Two-telescope g2 sensitivity of the four binaries on the EON-SII pair |
| `g2_keck` | g2 | Two-telescope g2 sensitivity of the four binaries on the Keck pair |
| `g3_algol_maunakea` | g3 | Algol closure phases with Subaru + Keck I + Keck II |
| `g3_betaaur_maunakea` | g3 | beta Aur closure phases with Subaru + Keck I + Keck II |
| `g3_deltavel_eonsii_paranal` | g3_campaign | delta Vel closure-phase campaign with three EON-SII units at Paranal |
| `g3_deltavel_vlt` | g3 | delta Vel closure phases with the four VLT UTs |
| `g3_spica_vlt` | g3 | Spica closure phases with the four VLT UTs (docs/three_telescope_feasibility.md) |
| `mc_sirius_b_eonsii` | montecarlo | Sirius B diameter Monte Carlo with the EON-SII pair (paper Table 2 cross-check) |
| `mc_sirius_b_lpqi_pathfinder` | montecarlo | Sirius B diameter Monte Carlo on the LPQI-Pathfinder (NOT + TNG, 550 m, one 1 nm filter) |
| `redclump_ii_dwarf` | scale | Red-clump scale precision (Kim & Kaiser comparison) with the 4800 K / log g 4.5 dwarf stand-in |
| `redclump_ii_supergiant` | scale | Red-clump scale precision (Kim & Kaiser comparison) with the 5000 K / log g 0 supergiant stand-in |
| `suite_phase6` | suite | Everything behind docs/three_telescope_feasibility.md and the paper tables (output/logs/run_*.sh) |
<!-- /catalog -->

## Installing

```bash
pip install hbtsim                 # CPU JAX; add "hbtsim[movie]" for the movies (matplotlib + ffmpeg)
hbtsim catalog validate            # the shipped JSON catalog
hbtsim snr --target spica --instrument keck_pair --vis2-method analytic
```

Python 3.11+; `pip install "hbtsim[gpu]"` for CUDA JAX, `"hbtsim[sed]"`
for h5py (only `scripts/prepare_newera.py` needs it). From a checkout:
`uv venv .venv && uv pip install -p .venv/bin/python -e ".[test,movie]"`.

### Model atmospheres (NewEra tables)

The NewEra PHOENIX tables that replace the blackbody + Claret model are not
in the package. They are hosted at NERSC
(`https://portal.nersc.gov/project/newera/binned/`, the `remote` block of
`hbtsim/configs/resources/*.json`) and cached on your machine:

```bash
hbtsim data path                         # where tables are looked for
hbtsim data fetch --target betaaur       # the four grid corners beta Aur needs (~48 MB)
hbtsim data fetch --target algol --allow-extrapolation
hbtsim data fetch --all                  # every 380-1000 nm table (19 files, 210 MB)
hbtsim data fetch --resource newera_redclump --model lte04800-4.50-0.0
hbtsim data list                         # what the cache holds
```

Every tool takes `--fetch` to pull what its target needs first
(`HBTSIM_AUTO_FETCH=1` does it always); without tables the stars fall back
to blackbody + linear limb darkening and the report says so. Environment
variables: `HBTSIM_DATA_DIR` (cache root; default `~/Library/Caches/hbtsim`
on macOS, `~/.cache/hbtsim` on Linux), `HBTSIM_DATA_URL` (mirror base URL,
`file://` works), `HBTSIM_NEWERA_DIR` / `HBTSIM_NEWERA_REDCLUMP_DIR` (an
explicit directory per resource), `HBTSIM_CONFIG_PATH` (your own catalog
files). A checkout's `data/newera/` is used when present. Hosting a mirror
is one command: `hbtsim data manifest <dir> --write` next to the tables.

## Running from a checkout

```bash
# one-time setup (macOS)
brew install ffmpeg
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[test,movie]"   # + [sed] for h5py, [gpu] for CUDA

# what is in the catalog
.venv/bin/python -m hbtsim catalog list
.venv/bin/python -m hbtsim catalog show target spica

# orbit movies (240 frames, ~1 min each on CPU); also installed as hbtsim-movie
.venv/bin/python -m hbtsim movie --target betaaur
.venv/bin/python -m hbtsim movie --target algol

# SNR tables (hbtsim-snr)
.venv/bin/python -m hbtsim snr --target algol --instrument keck_pair --baseline 50 85

# tests (analytic validation suite, ~65 s)
.venv/bin/python -m pytest

# feasibility campaigns with the NewEra tables (hbtsim data fetch --all first, or --fetch)
.venv/bin/python -m hbtsim run g3_betaaur_maunakea
.venv/bin/python -m hbtsim run g2_eonsii
.venv/bin/python -m hbtsim run suite_phase6 --jobs 4     # everything behind the docs
```

## Running on NERSC Perlmutter

```bash
git clone https://github.com/nugent68/hbtsim.git ~/hbtsim
bash ~/hbtsim/scripts/perlmutter/setup_env.sh   # conda env jax-gpu-env
```

The env uses pip `jax[cuda12]` wheels, which bundle CUDA/cuDNN — do
**not** `module load cudatoolkit cudnn nccl` (their LD_LIBRARY_PATH
entries shadow the bundled libraries and break JAX).

Login nodes carry a *shared* A100: fine for short runs with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `--chunk 4`, but contention
can OOM you. For real work use a dedicated GPU:

```bash
# interactive (account: your _g allocation)
salloc -N 1 -C gpu -G 1 -c 32 -q interactive -t 30 -A m2218_g
module load python && conda activate jax-gpu-env
cd ~/hbtsim && srun -n 1 python -m hbtsim g2spec --target algol \
    --compute-only --diameter 10 --baseline 85 \
    --npz $SCRATCH/hbt/g2spec_algol_keck.npz
# (note: standalone srun needs an explicit -n 1)

# or batch
sbatch ~/hbtsim/scripts/perlmutter/bench.sbatch
```

Keep the repo and env in `$HOME` ($SCRATCH is purged after ~8 weeks);
write job outputs to `$SCRATCH/hbt/` (keep them out of the scratch root —
`mkdir -p $SCRATCH/hbt` once), then copy the npz back and render the
movie locally with `--render-only`.

## Package layout

- `hbtsim/configs/` — the catalog: one JSON file per band table,
  limb-darkening table, target, telescope, detector, spectrograph,
  backend, site, array, data resource and campaign (single source of
  truth; `sources` / `notes` / `provenance` / `assumptions` carry the
  literature trail)
- `hbtsim/catalog/` — loads and validates the catalog (`Catalog`,
  `load_target`, `load_array`, …; `extends`, search path, `hbtsim catalog`
  CLI); `hbtsim/serialize.py` — canonical JSON / content hashes of the
  built objects (the campaign cache key)
- `hbtsim/params.py` — constants, `Star`/`BinarySystem`/`DiskTarget`/
  `GridConfig`/`MovieConfig`
- `hbtsim/cli_common.py`, `hbtsim/__main__.py` — the shared `--target` /
  `--instrument` / `--config-dir` / `--fetch` options and the `hbtsim
  <command>` dispatcher
- `hbtsim/data.py` — the NewEra table cache and fetcher (`hbtsim data
  fetch|list|path|manifest`; NERSC portal manifest, sha256-verified)
- `hbtsim/orbit.py` — Keplerian sky geometry, oriented on the sky by Ω
- `hbtsim/aperture.py` — finite-aperture (pupil) averaging of |V|² and
  of the three-pupil bispectrum
- `hbtsim/geometry.py` — `Site`, hour angle, uv projection, fringe drift
- `hbtsim/limbdark.py` — linear LD law, analytic and numeric (tabulated
  profile) disk visibilities
- `hbtsim/sed.py`, `scripts/prepare_newera.py` — model-atmosphere flux
  and I(μ, λ) tables (NewEra PHOENIX HSR-RF reader), the `NewEraGrid`
  (T_eff, log g) interpolation, channel averaging, Doppler / rotational
  broadening / extinction hooks
- `hbtsim/render.py` — JAX rendering of the occulted limb-darkened disks
- `hbtsim/hbt.py` — exact K-point DFT sampling of V(u, v), |V|², g²(B);
  analytic binary visibility
- `hbtsim/fftmap.py` — padded-FFT 2-D maps (plots/cross-checks only)
- `hbtsim/spectral.py` — batched multi-wavelength V(B, λ) (GPU)
- `hbtsim/photometry.py` — band fluxes, AB magnitudes, anchoring
- `hbtsim/snr.py`, `hbtsim/snr_cli.py` — photon-budget SNR (telescopes,
  detectors, spectrograph multiplexing)
- `hbtsim/snr3.py` — triple-correlation (closure-phase) SNR, uv tracks,
  shared-geometry backends and multi-night campaigns
  (`campaign_g3_snr`)
- `hbtsim/estimators.py`, `hbtsim/montecarlo.py` — the EON-SII design
  study's photon-level and classic-HBT estimators beside the matched
  filter, and a Poisson coincidence-histogram Monte Carlo of a
  multiplexed diameter measurement
- `hbtsim/single.py`, `hbtsim/chromatic.py` — single stars (Sirius A,
  Vega) with NewEra profiles, smeared uniform-disk inversion, and
  Balmer-core vs continuum (chromatic) diameters
- `hbtsim/movie.py`, `hbtsim/cli.py` — the 3-panel orbit movie
- `hbtsim/g2spec.py` — the g²(λ)-with-error-bars movie
- `scripts/` — `catalog_tables.py` (renders the parameter tables of this
  README and the docs from the catalog), `feasibility_g3.py` (every number in the docs),
  `sed_compare.py` (blackbody vs NewEra), `eonsii_crosscheck.py` and
  `eonsii_montecarlo.py` (the EON-SII sensitivity gap),
  `deltavel_eonsii.py` (δ Vel closure phases with three EON-SII units),
  `chromatic_diameters.py` (Sirius/Vega Balmer-core diameters),
  benchmark + Perlmutter setup/sbatch
- `tests/` — analytic validation tests (`pytest tests/`, ~330 tests,
  ~75 s; the NewEra-data tests skip without `data/newera/`); the objects
  come from the catalog through the fixtures of `tests/conftest.py`

## References

- Hanbury Brown, R. & Twiss, R. Q. 1956, Nature 178, 1046
- Rai, K. N., Basak, S. & Saha, P. 2021, MNRAS (arXiv:2105.09532) —
  binary-star intensity interferometry formalism
- Guerin, W. et al. 2025, Comptes Rendus Physique (arXiv:2503.22446) —
  SII in the photon-counting regime
- Southworth, J., Bruntt, H. & Buzasi, D. L. 2007, A&A 467, 1215 —
  Beta Aurigae absolute dimensions
- Hummel, C. A. et al. 1995, AJ 110, 376 — Beta Aur interferometric orbit
- Baron, F. et al. 2012, ApJ 752, 20 — Algol CHARA/MIRC imaging
- Zavala, R. T. et al. 2010, ApJ 715, L44 — Algol orbit and parallax
- Kolbas, V. et al. 2015, MNRAS 451, 4150 — Algol spectral disentangling
- Claret, A. & Bloemen, S. 2011, A&A 529, A75 — limb-darkening coefficients
- Jester, S. et al. 2005, AJ 130, 873 — Johnson→SDSS transformations
