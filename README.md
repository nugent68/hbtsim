# hbtsim — Intensity interferometry of binary stars

Simulation of the Hanbury Brown–Twiss (HBT) effect in optical intensity
interferometry for the eclipsing binary **Beta Aurigae (Menkalinan)**.
The code renders the binary as a pair of limb-darkened stellar disks on a
sky grid, computes the squared visibility |V|² via a zero-padded 2D FFT
(JAX), and produces a three-panel MP4 movie over one orbital period:

1. **Sky image** — the two stars orbiting each other (false-color RGB
   from renders at 700/550/440 nm, Planck-weighted and white-balanced to
   7500 K so hot stars look blue and cool ones orange; to scale,
   milliarcsecond axes), including the partial eclipses.
2. **Lightcurve** — apparent SDSS g- and i-band magnitudes with a moving
   phase cursor, anchored to the observed system brightness
   (g ≈ 1.80, i ≈ 2.10).
3. **g²(B)** — the second-order correlation versus telescope baseline at
   400 nm and 800 nm, sampled every 10 m from 10 m to 150 m, with the
   smooth underlying curve.

The simulation starts at greatest projected separation (quadrature).

## Physics

For a chaotic (thermal) source, the Siegert relation links the measured
intensity correlation to the first-order coherence:

    g²(B) = 1 + |V(B)|²,

where V(B) is the complex degree of coherence, the normalized Fourier
transform of the sky brightness distribution evaluated at spatial frequency
**u** = **B**/λ (van Cittert–Zernike). Here this is computed numerically:

    image I(θx, θy)  →  zero-padded 2D real FFT  →  V = Ṽ/Ṽ(0,0)
                     →  |V|²(u, v)  →  bilinear sample along the baseline
                     →  g²(B) = 1 + |V(B)|².

Each star is a linearly limb-darkened disk, I(μ)/I(1) = 1 − u_λ(1 − μ),
with Claret & Bloemen (2011) coefficients for ~9300 K stars, weighted by
the Planck function at each star's effective temperature. Eclipses are
handled by z-ordering the disks on the grid, which also yields the g/i
lightcurves by direct image summation. The lightcurve is calibrated to
apparent magnitudes in two steps: synthetic monochromatic AB magnitudes
are computed from the physical flux at Earth, f_λ = B_λ(T) × Ω_star
(blackbody photospheres), and each band is then shifted by a constant so
its maximum light matches the observed photometry, g ≈ 1.80, i ≈ 2.10
(from V = 1.90, B−V = 0.03 via the Jester et al. 2005 transformations).
The blackbody approximation alone is ~0.46 mag too faint in g and
~0.18 mag in i (no Balmer/Paschen line blanketing or H⁻ opacity); the
anchoring removes that zero-point error while the eclipse shapes and
depths remain purely simulated. The g²(B) panel orients the baseline
along the instantaneous projected separation axis by default
(`--baseline-pa` accepts a fixed position angle in degrees instead), so the
binary fringes — period λ/ρ ≈ 25 m at 400 nm at quadrature — are always in
view and visibly stretch as the projected separation shrinks toward eclipse,
collapsing to the single-disk envelope (first null at 1.22λ/θ ≈ 98 m at
400 nm) at conjunction.

The g² = 1 + |V|² normalization is the ideal fully-coherent-detection
limit; a real intensity interferometer measures a contrast reduced by the
ratio of the coherence time to the detector resolution time (see Rai,
Basak & Saha 2021, eq. 6).

Numerical layout: 1024² source grid at 0.01 mas/pixel (disk radii ≈ 50 px,
limb darkening well resolved), FFT zero-padded to 8192² giving a baseline
sampling of ~1.0 m at 400 nm over the 10–150 m range of interest.

## Beta Aurigae parameters

| Quantity | Value | Source |
|---|---|---|
| Period, eccentricity | 3.96004 d, e = 0 | Southworth, Bruntt & Buzasi (2007) |
| Inclination | 76.8° | Southworth et al. (2007) |
| Masses | 2.376, 2.291 M☉ | Southworth et al. (2007) |
| Radii | 2.762, 2.568 R☉ | Southworth et al. (2007) |
| T_eff | 9350, 9200 K (A1m IV) | Southworth et al. (2007) |
| Distance | 24.87 pc (π = 40.21 mas) | Hipparcos (van Leeuwen 2007) |
| Angular diameters | 1.033, 0.960 mas | derived |
| Angular semi-major axis | 3.303 mas | derived |

## Running

```bash
# one-time setup (macOS)
brew install ffmpeg
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python jax numpy matplotlib pytest scipy

# full movie (240 frames over one orbit, ~10 min on CPU)
.venv/bin/python -m hbtsim --frames 240 --out output/betaaur_hbt.mp4

# quick draft
.venv/bin/python -m hbtsim --frames 24 --out output/draft.mp4

# fixed instrumental baseline orientation instead of tracking the binary axis
.venv/bin/python -m hbtsim --baseline-pa 0
```

## Systems

Two binaries are built in (`--system` on every CLI; add more as
`BinarySystem` instances in `hbtsim/params.py`):

- **`betaaur`** — Beta Aurigae (default): near-twin A1m IV pair, shallow
  0.08 mag eclipses (parameters above).
- **`algol`** — Algol (β Persei) A–B: a B8V dwarf (3.17 M☉, 2.73 R☉,
  12,550 K) eclipsed by a K0IV subgiant (0.70 M☉, 3.48 R☉, 4,900 K);
  P = 2.867328 d, i = 98.7°, a = 2.151 mas, d = 28.82 pc (Baron et al.
  2012 CHARA imaging; Zavala et al. 2010 parallax; Kolbas et al. 2015
  Teffs). Angular diameters 0.881 / 1.123 mas. The model's primary
  eclipse depth is 1.48 mag in g (the published 1.27 mag V depth is
  diluted by Algol C's ~10% third light; C-corrected it is ~1.5 mag).
  Caveats: component C (~70 mas away) is excluded from the image and
  removed from the photometric anchors (g 2.07, i 2.58); the
  Roche-lobe-filling secondary is rendered as a sphere, so the
  ellipsoidal variation and reflection effect of the real out-of-eclipse
  lightcurve are absent. Limb darkening is **per star** (`Star.ld_table_nm`,
  Claret & Bloemen 2011): u(400 nm) = 0.42 for the B8V primary vs 0.89
  for the K subgiant — Algol is why the LD coefficient lives on `Star`
  rather than on the system.

## Observation SNR (hbtsim.snr)

`hbtsim/snr.py` estimates the signal-to-noise of a g²(B) measurement with
a pair of telescopes and photon-counting detectors, using the standard
photon-counting intensity-interferometry budget: detected stellar rates
R_i from the source AB magnitude (anchored blackbody model), excess
coincidences N_sig = ½|V|²·τ_c·R₁R₂·T for unpolarized light spread over
the detectors' combined timing jitter, and a Gaussian matched filter
against the accidental-coincidence floor (including dark and sky counts),

    SNR = ½ |V|² τ_c R₁ R₂ √T / √(b₁ b₂ · 2√π σ_pair).

Per-pixel non-paralyzable dead time is included; SNR is nearly independent
of the filter width in the unsaturated limit. Everything is parameterized
(`Telescope`, `Detector`, `Observation` dataclasses): telescope diameter
and throughput, detector PDE curve / jitter / dead time / dark rate /
pixel count, filter width, integration time, polarization.

Built-in example: the C2PU pair (Centre Pédagogique Planète Univers,
Calern plateau) — two 1 m telescopes on a 15 m baseline — with Pi Imaging
SPAD Lambda detectors (PDE 22% at 400 nm / 14% at 800 nm, 120 ps FWHM
jitter, 10 ns dead time, 250 cps dark; `background/SPADlambdadatasheet.pdf`).

Two observing modes:

- **Spectral (default)** — the light is dispersed along the SPAD Lambda's
  320×1 linear array, so each pixel pair is an independent ~1.7 nm
  spectral channel (400–950 nm) measuring its own g² with per-pixel dead
  time and dark counts. Channel SNRs add in quadrature — a ~√320 ≈ 18×
  multiplexing gain over a single filter. Per-channel |V|²(B, λ) comes
  from the batched FFT pipeline (`hbtsim.spectral.spectral_vis2`, valid
  at all phases including eclipses; see "Batched spectral FFT" below).
  `--vis2-method analytic` switches to the analytic binary visibility
  (`hbt.binary_vis2_analytic`) — instant and agreeing with the FFT to
  <0.5%, but valid only out of eclipse. At fixed baseline the fringe
  phase sweeps with wavelength, so individual channels sit on fringe
  maxima and nulls — the SNR-weighted channel spectrum traces the binary
  fringes.
- **Narrowband** — a single filter per wavelength (`--mode narrowband`),
  with |V|²(B) from the FFT pipeline.

```bash
# C2PU defaults: spectral mode, B = 50 m, 2 x 1 m, 320 channels, 1 h
.venv/bin/python -m hbtsim.snr_cli

# several baselines / different hardware
.venv/bin/python -m hbtsim.snr_cli --baseline 15 50 100 --diameter 1.5

# the original two-filter setup
.venv/bin/python -m hbtsim.snr_cli --mode narrowband --baseline 15 \
    --wavelengths 400 800 --filter-width 10
```

For Beta Aurigae at quadrature with C2PU-class hardware, per 1 h:
spectral mode gives **total SNR ≈ 16 at B = 50 m** (≈ 6 at 15 m, ≈ 5 at
100 m; best single channels reach SNR ≈ 1.5 near 700 nm), whereas a
single 10 nm filter gives only ≈ 0.1–0.4 per wavelength — the spectral
multiplexing is what makes a 1 m-class measurement practical.

## g²(λ) movie with hourly error bars

`python -m hbtsim.g2spec` produces a two-panel movie — the stars
projected on the sky next to the measurable g²(λ) spectrum: one frame
per hour over the full 3.96-day orbit, each showing a simulated one-hour
measurement (per-channel points, inverse-variance 8-channel bins with 1σ
error bars from the photon budget — jitter, dead time, dark counts,
unpolarized factor) over the true model curve. The per-channel
|V|²(B, λ) comes from the batched FFT pipeline, so the eclipse frames
(overlapping disks) are exact. Telescope diameter, throughput and
baseline are options; `snr.KECK` models the two 10 m Keck telescopes at
B ≈ 85 m (per-hour per-channel SNR ~7 vs ~0.5 for C2PU — caveats: a
single SPAD pixel saturates at Keck count rates, and a 10 m aperture on
an 85 m baseline averages |V|² over B ± 10 m, neither modeled). The
compute step is GPU-friendly and separable from the rendering:

```bash
# on a Perlmutter GPU (~4-10 min; on shared/login GPUs keep
# XLA_PYTHON_CLIENT_PREALLOCATE=false and --chunk 4):
python -m hbtsim.g2spec --compute-only --diameter 10 --baseline 85 \
    --chunk 4 --npz $SCRATCH/g2spec_keck.npz
# locally (needs ffmpeg):
python -m hbtsim.g2spec --render-only --npz output/g2spec_keck.npz \
    --out output/g2spec_keck.mp4
```

## Batched spectral FFT on GPU

`hbtsim.spectral.spectral_vis2(pos, baselines, wavelengths, system, grid)`
computes |V|² for every (wavelength, baseline) pair in one jitted JAX
computation: per channel it renders the limb-darkened binary (the
limb-darkening coefficient and Planck weights are wavelength-traceable),
takes the zero-padded 2D real FFT, and samples along the baseline PA.
Channels are processed in chunks via `jax.lax.map(..., batch_size=chunk)`
— each chunk is one batched (cu)FFT, and buffers are reused between
chunks, so peak memory is ~0.85 GiB × chunk (default chunk 16 on GPU
≈ 14 GiB, fits a 40 GB A100; chunk 4 on CPU).

Measured (320 channels, 15 baselines, pad 8192²): **3.2 ms/channel on a
Perlmutter A100** (1.02 s steady-state after a one-off ~12 s compile;
end-to-end 320-channel spectral SNR in 1.3 s, agreeing with the analytic
visibility to 0.11%) vs ~460 ms/channel on an Intel iMac Pro CPU — a
~140× speedup. On a *shared* GPU (e.g. a login node) set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and reduce `--chunk`, otherwise
XLA's default 75% preallocation collides with other users and starves
the cuFFT workspace.

```bash
python scripts/bench_spectral.py                    # benchmark, default device
JAX_PLATFORMS=cpu python scripts/bench_spectral.py  # force CPU
```

## Running on NERSC Perlmutter

One-time setup (login node; **login nodes have no GPUs** — `jax.devices()`
showing CPU there is expected):

```bash
git clone https://github.com/nugent68/binary.git ~/binary
bash ~/binary/scripts/perlmutter/setup_env.sh   # creates conda env jax-gpu-env
```

The env uses the pip `jax[cuda12]` wheels, which bundle CUDA/cuDNN — do
**not** `module load cudatoolkit cudnn nccl` (their LD_LIBRARY_PATH
entries shadow the bundled libraries and can break JAX).

Interactive GPU session (pick your `_g` account;
`sacctmgr show assoc user=$USER format=account%20 -n -P | sort -u`):

```bash
salloc -N 1 -C gpu -G 1 -c 32 -q interactive -t 30 -A m2218_g
module load python && conda activate jax-gpu-env
cd ~/binary && python scripts/bench_spectral.py --channels 320
```

Batch job (edit the account in the header if needed):

```bash
sbatch ~/binary/scripts/perlmutter/bench.sbatch
```

Keep the repo and env in `$HOME` (not `$SCRATCH`, which is purged after
~8 weeks); write large job outputs to `$SCRATCH`.

## Tests

Analytic sanity checks (Airy pattern and first null, limb-darkened disk
visibility, binary fringe formula, eclipse circle-overlap area, g²(0) = 2,
FFT-vs-direct-DFT interpolation accuracy):

```bash
.venv/bin/python -m pytest tests/ -v
```

## Package layout

- `hbtsim/params.py` — physical constants, `BinarySystem`/`GridConfig`/
  `MovieConfig` dataclasses, the `BETA_AUR` instance (single source of
  truth; add new systems here)
- `hbtsim/orbit.py` — circular-orbit sky geometry
- `hbtsim/limbdark.py` — linear limb-darkening law + analytic visibility
- `hbtsim/render.py` — JAX rendering of the occulted limb-darkened disks
- `hbtsim/hbt.py` — FFT → |V|² → g²(B)
- `hbtsim/photometry.py` — band fluxes and magnitudes
- `hbtsim/movie.py`, `hbtsim/cli.py` — frame precomputation and the
  3-panel animation

## References

- Hanbury Brown, R. & Twiss, R. Q. 1956, Nature 178, 1046
- Rai, K. N., Basak, S. & Saha, P. 2021, MNRAS (arXiv:2105.09532) —
  binary-star intensity interferometry formalism
- Guerin, W. et al. 2025, Comptes Rendus Physique (arXiv:2503.22446) —
  SII in the photon-counting regime
- Southworth, J., Bruntt, H. & Buzasi, D. L. 2007, A&A 467, 1215 —
  Beta Aurigae absolute dimensions
- Hummel, C. A. et al. 1995, AJ 110, 376 — interferometric orbit
- Claret, A. & Bloemen, S. 2011, A&A 529, A75 — limb-darkening coefficients
