# hbtsim — Intensity interferometry of binary stars

Simulation of the Hanbury Brown–Twiss (HBT) effect in optical intensity
interferometry for bright binaries: **Beta Aurigae (Menkalinan)**,
**Algol (β Persei) A–B**, **Spica** and **δ Velorum Aa–Ab**. The code renders each binary
as a pair of limb-darkened stellar disks on a sky grid, computes the
complex visibility V(**u**) by an exact discrete Fourier transform of the
image at the sampled baselines (JAX, GPU-batched for spectral work),
predicts the signal-to-noise of real photon-counting
observations, and produces movies:

1. **Orbit movie** (`python -m hbtsim --system {betaaur,algol}`) — three
   panels over one orbital period: the stars on the sky (false-color RGB
   from renders at 700/550/440 nm, Planck-weighted and white-balanced to
   7500 K, so hot stars look blue and cool ones orange; to-scale mas
   axes), the apparent g/i lightcurves anchored to observed photometry,
   and g²(B) at 400 and 800 nm for baselines 10–150 m.
2. **g²(λ) measurement movie** (`python -m hbtsim.g2spec`) — the stars on
   the sky beside the g² *spectrum* a real telescope pair would measure
   each hour: 320 spectral channels (400–950 nm) with 1σ error bars from
   the full photon budget, simulated noisy data over the true curve.

All simulations start at greatest projected separation (quadrature) and
follow the baseline along the projected separation axis by default
(`--baseline-pa` for a fixed instrumental orientation).

## The four systems

|  | **Beta Aurigae** (`betaaur`, default) | **Algol A–B** (`algol`) | **Spica** (`spica`) | **δ Velorum Aa–Ab** (`deltavel`) |
|---|---|---|---|---|
| Components | A1m IV + A1m IV near-twins | B8V dwarf + K0IV subgiant | B1 III-IV + B2 V | A2 IV + A4 V rapid rotators |
| Masses | 2.376 / 2.291 M☉ | 3.17 / 0.70 M☉ | 11.43 / 7.21 M☉ | 2.43 / 2.27 M☉ |
| Radii | 2.762 / 2.568 R☉ | 2.73 / 3.48 R☉ | 7.47 / 3.74 R☉ | 2.97 / 2.52 R☉ |
| T_eff | 9350 / 9200 K | 12550 / 4900 K | 25300 / 20900 K | 9450 / 9830 K |
| Period | 3.96004 d | 2.867328 d | 4.0145 d | 45.1503 d, e = 0.29 |
| Inclination | 76.8° | 98.70° | 63.1° | 89.0° |
| Distance | 24.87 pc | 28.82 pc | 76.6 pc | 25.13 pc (orbital parallax) |
| Angular semi-major axis | 3.303 mas | 2.151 mas | 1.71 mas | 16.56 mas |
| Angular diameters | 1.033 / 0.960 mas | 0.881 / 1.123 mas | 0.91 / 0.45 mas | 1.10 / 0.93 mas |
| Eclipses | partial, ~0.08 mag | deep primary, **1.48 mag in g** | none | grazing |
| Anchored photometry | g 1.80, i 2.10 | g 2.07, i 2.58 (C-corrected) | g 0.71, i 1.06 | g 1.90, i 2.25 (B-corrected) |
| NewEra tables | both stars (interpolated) | A (clamped to 12 000 K); B blackbody | blackbody | both stars (interpolated) |
| Sources | Southworth et al. 2007; Hipparcos | Baron et al. 2012 (CHARA); Zavala et al. 2010; Kolbas et al. 2015 | Herbison-Evans et al. 1971; Tkachenko et al. 2016 | Mérand et al. 2011 |

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

Adding another binary is one `BinarySystem` instance in
`hbtsim/params.py` (registered in `SYSTEMS`).

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
table, the grid at NERSC `/global/cfs/projectdirs/newera` covers
8000–12 000 K × log g 3.0–4.5), interpolated bilinearly in T_eff and
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
via `Telescope`, `Backend`, `Detector`, `Observation` and `Spectrograph`
dataclasses. The instrument model carries the effects a real
implementation cannot escape:

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
  (the next-generation design, `SPAD_LAMBDA_NG`: a real-time correlator
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

Built-in hardware: the **C2PU pair** (Calern, 2 × 1 m, 15 m apart), the
**Keck pair** (2 × 10 m, ~85 m), the **EON-SII pair** (arXiv:2608.17444:
two transportable 4 m telescopes of 9 m², a 400–550 nm spectrograph with
1000 effective channels, Photonis MCP-PMT or QUASAR SPAD detectors,
1 GHz time-tag links; `EON_SII_TELESCOPE`, `EONSII_*`,
`bispectrum.eonsii_pair`, `--instrument eonsii`), and the Pi Imaging
**SPAD Lambda** detector (320×1 pixels, PDE 22%/14% at 400/800 nm,
120 ps FWHM jitter, 10 ns dead time, 250 cps dark).

Two observing modes (`python -m hbtsim.snr_cli --system {betaaur,algol}`):

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
(`scripts/feasibility_g3.py --g2`, docs §5f): a single 10 nm filter on
C2PU gives SNR ≈ 0.1–0.9; the 320-channel SPAD Lambda with a correlator
readout lifts that to **≈ 15/√h at B = 35 m** (8 at 50 m) — the
time-tag link caps it at ≈ 7 — and an R = 5000 backend to 60 (54 with
the NewEra tables); the Keck pair at 85 m reaches 240/√h with 320
channels (σ_|V|² ≈ 0.02 per channel-hour) and the EON-SII 4 m pair at
30 m ≈ 570/√h with 1000 channels, both link-limited. The EON-SII
design study quotes 10–25× shorter times; `scripts/eonsii_montecarlo.py`
reruns its Sirius B Monte Carlo and shows its hours need a coincidence
window of 2–6 ps, well below the detectors' 12–28 ps jitter (docs §5e.1).
The matched filter stays the default. On single stars
(`scripts/chromatic_diameters.py`, `docs/chromatic_diameters_eonsii.md`)
EON-SII measures Sirius's and Vega's Balmer-core diameters (+3.6 to
+4.7 % over the continuum at Hβ–Hδ) at 5σ in 1–2 nights through its
links, or in minutes with channel-subset tagging. Keck caveats: a
single SPAD pixel saturates at Keck count rates (`--n-pixels`), and a
10 m aperture on an 85 m baseline averages |V|² over B ± 10 m
(modeled: `hbtsim.aperture`).

## g²(λ) movies with error bars

`python -m hbtsim.g2spec --system algol --diameter 10 --baseline 85`
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
telescope triangles (built in: `MAUNAKEA_SUBARU_KECK` — Subaru + Keck I
+ Keck II at 152/85/226 m, from site coordinates; `VLT_UT`), projects
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
`scripts/feasibility_g3.py`): the Subaru arms resolve out the ~1 mas
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
Paranal (`bispectrum.eonsii_triangle`, `snr3.campaign_g3_snr`,
`scripts/deltavel_eonsii.py`) detect its closure-phase template in ~20
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

## Running locally

```bash
# one-time setup (macOS)
brew install ffmpeg
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[test,movie]"   # + [sed] for h5py, [gpu] for CUDA

# orbit movies (240 frames, ~1 min each on CPU); also installed as hbtsim-movie
.venv/bin/python -m hbtsim --system betaaur
.venv/bin/python -m hbtsim --system algol

# SNR tables (hbtsim-snr)
.venv/bin/python -m hbtsim.snr_cli --system algol --baseline 50 85

# tests (analytic validation suite, ~65 s)
.venv/bin/python -m pytest

# feasibility tables with the NewEra tables (rsync data/newera/ from NERSC first)
.venv/bin/python scripts/feasibility_g3.py --system betaaur --array maunakea --newera-dir data/newera
.venv/bin/python scripts/feasibility_g3.py --g2 --instrument eonsii --newera-dir data/newera --allow-extrapolation
```

## Running on NERSC Perlmutter

```bash
git clone https://github.com/nugent68/binary.git ~/binary
bash ~/binary/scripts/perlmutter/setup_env.sh   # conda env jax-gpu-env
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
cd ~/binary && srun -n 1 python -m hbtsim.g2spec --system algol \
    --compute-only --diameter 10 --baseline 85 \
    --npz $SCRATCH/hbt/g2spec_algol_keck.npz
# (note: standalone srun needs an explicit -n 1)

# or batch
sbatch ~/binary/scripts/perlmutter/bench.sbatch
```

Keep the repo and env in `$HOME` ($SCRATCH is purged after ~8 weeks);
write job outputs to `$SCRATCH/hbt/` (keep them out of the scratch root —
`mkdir -p $SCRATCH/hbt` once), then copy the npz back and render the
movie locally with `--render-only`.

## Package layout

- `hbtsim/params.py` — constants, `Star`/`BinarySystem`/`GridConfig`/
  `MovieConfig`, per-star LD tables, the `BETA_AUR`, `ALGOL`, `SPICA`
  and `DELTA_VEL` instances and the `SYSTEMS` registry (single source
  of truth)
- `hbtsim/orbit.py` — Keplerian sky geometry, oriented on the sky by Ω
- `hbtsim/aperture.py` — finite-aperture (pupil) averaging of |V|² and
  of the three-pupil bispectrum
- `hbtsim/geometry.py` — sites, hour angle, uv projection, fringe drift
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
- `scripts/` — `feasibility_g3.py` (every number in the docs),
  `sed_compare.py` (blackbody vs NewEra), `eonsii_crosscheck.py` and
  `eonsii_montecarlo.py` (the EON-SII sensitivity gap),
  `deltavel_eonsii.py` (δ Vel closure phases with three EON-SII units),
  `chromatic_diameters.py` (Sirius/Vega Balmer-core diameters),
  benchmark + Perlmutter setup/sbatch
- `tests/` — analytic validation tests (`pytest tests/`, ~255 tests,
  ~90 s; the NewEra-data tests skip without `data/newera/`)

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
