# hbtsim — Intensity interferometry of binary stars

Simulation of the Hanbury Brown–Twiss (HBT) effect in optical intensity
interferometry for bright eclipsing binaries, currently **Beta Aurigae
(Menkalinan)** and **Algol (β Persei) A–B**. The code renders each binary
as a pair of limb-darkened stellar disks on a sky grid, computes the
squared visibility |V|² via zero-padded 2D FFTs (JAX, GPU-batched for
spectral work), predicts the signal-to-noise of real photon-counting
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

## The two systems

|  | **Beta Aurigae** (`betaaur`, default) | **Algol A–B** (`algol`) |
|---|---|---|
| Components | A1m IV + A1m IV near-twins | B8V dwarf + K0IV subgiant |
| Masses | 2.376 / 2.291 M☉ | 3.17 / 0.70 M☉ |
| Radii | 2.762 / 2.568 R☉ | 2.73 / 3.48 R☉ |
| T_eff | 9350 / 9200 K | 12550 / 4900 K |
| Period | 3.96004 d | 2.867328 d |
| Inclination | 76.8° | 98.70° |
| Distance | 24.87 pc | 28.82 pc |
| Angular semi-major axis | 3.303 mas | 2.151 mas |
| Angular diameters | 1.033 / 0.960 mas | 0.881 / 1.123 mas |
| Eclipses | partial, ~0.08 mag | deep primary, **1.48 mag in g** |
| Anchored photometry | g 1.80, i 2.10 | g 2.07, i 2.58 (C-corrected) |
| Sources | Southworth et al. 2007; Hipparcos | Baron et al. 2012 (CHARA); Zavala et al. 2010; Kolbas et al. 2015 |

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

Adding another binary is one `BinarySystem` instance in
`hbtsim/params.py` (registered in `SYSTEMS`).

## Physics

For a chaotic (thermal) source, the Siegert relation links the measured
intensity correlation to the first-order coherence:

    g²(B) = 1 + |V(B)|²,

where V(B) is the complex degree of coherence, the normalized Fourier
transform of the sky brightness distribution at spatial frequency
**u** = **B**/λ (van Cittert–Zernike). Numerically:

    image I(θx, θy)  →  zero-padded 2D real FFT  →  V = Ṽ/Ṽ(0,0)
                     →  |V|²(u, v)  →  bilinear sample along the baseline
                     →  g²(B) = 1 + |V(B)|².

Each star is a linearly limb-darkened disk, I(μ)/I(1) = 1 − u_λ(1 − μ),
with per-star Claret & Bloemen (2011) coefficients, weighted by the
Planck function at its effective temperature. Eclipses are handled by
z-ordering the disks on the grid, which also yields the lightcurves by
direct image summation. Lightcurves are calibrated in two steps:
synthetic monochromatic AB magnitudes from the physical flux at Earth
(blackbody photospheres), then a constant per-band shift anchoring
maximum light to the observed photometry (the blackbody zero point is a
few tenths of a magnitude off — no line blanketing or H⁻ opacity — but
eclipse shapes and depths remain purely simulated).

The g² = 1 + |V|² normalization is the ideal fully-coherent-detection
limit; a real intensity interferometer measures a contrast reduced by
the ratio of coherence time to detector resolution (Rai, Basak & Saha
2021, eq. 6) — that physics lives in the SNR module below.

Numerical layout: 1024² source grid at 0.01 mas/pixel (disk radii
~45–55 px, limb darkening well resolved), FFT zero-padded to 8192²
giving ~1 m baseline sampling at 400 nm. The FFT pipeline is validated
against analytic results (Airy nulls, limb-darkened Bessel series, the
binary fringe formula, circle-overlap eclipse depths) in the test suite.

## Observation SNR (hbtsim.snr)

`hbtsim/snr.py` estimates the signal-to-noise of a g² measurement with a
pair of telescopes and photon-counting detectors: detected stellar rates
from the anchored source model, excess coincidences
N_sig = ½|V|²·τ_c·R₁R₂·T for unpolarized light spread over the
detectors' combined timing jitter, matched-filtered against the
accidental floor (dark and sky counts included),

    SNR = ½ |V|² τ_c R₁ R₂ √T / √(b₁ b₂ · 2√π σ_pair),

with per-pixel non-paralyzable dead time. Everything is parameterized
via `Telescope`, `Detector`, `Observation` dataclasses. Built-in
hardware: the **C2PU pair** (Calern, 2 × 1 m, 15 m apart), the **Keck
pair** (2 × 10 m, ~85 m), and the Pi Imaging **SPAD Lambda** detector
(320×1 pixels, PDE 22%/14% at 400/800 nm, 120 ps FWHM jitter, 10 ns
dead time, 250 cps dark; datasheet in `background/`).

Two observing modes (`python -m hbtsim.snr_cli --system {betaaur,algol}`):

- **Spectral (default)** — the light is dispersed along the SPAD
  Lambda's 320-pixel array: each pixel pair is an independent ~1.7 nm
  channel measuring its own g², and channel SNRs add in quadrature
  (~√320 ≈ 18× multiplexing gain). Per-channel |V|²(B, λ) comes from the
  batched FFT pipeline (valid through eclipses); `--vis2-method
  analytic` is the instant out-of-eclipse alternative (agrees to <0.5%).
- **Narrowband** — single filters (`--mode narrowband --wavelengths 400
  800 --filter-width 10`).

Headline numbers for Beta Aurigae at quadrature, 1 h: a single 10 nm
filter on C2PU gives SNR ≈ 0.1–0.4; spectral multiplexing lifts that to
**≈ 16 at B = 50 m**; the Keck pair reaches per-channel SNR ≈ 7
(σ_|V|² ≈ 0.01–0.03). Keck caveats (documented, not modeled): a single
SPAD pixel saturates at Keck count rates, and a 10 m aperture on an
85 m baseline averages |V|² over B ± 10 m.

## g²(λ) movies with error bars

`python -m hbtsim.g2spec --system algol --diameter 10 --baseline 85`
computes one frame per hour over a full orbit (69 epochs for Algol, 96
for Beta Aur): per-channel simulated measurements with 1σ error bars
(inverse-variance 8-channel bins highlighted) over the true model curve,
beside the temperature-colored sky view. During Algol's primary eclipse
the blue channels dim by 1.5 mag and the error bars visibly inflate
while the fringe spectrum collapses. Compute (`--compute-only`, GPU) and
rendering (`--render-only`, needs ffmpeg) are separable via the npz file.

## Batched spectral FFT on GPU

`hbtsim.spectral.spectral_vis2(pos, baselines, wavelengths, system, grid)`
computes |V|² for every (wavelength, baseline) pair in one jitted JAX
computation; channels run in chunks via `jax.lax.map(batch_size=...)`
(one batched cuFFT per chunk, buffers reused, peak ≈ 0.85 GiB × chunk).
Measured: **3.2 ms/channel on a Perlmutter A100** (320 channels in
1.02 s steady-state) vs ~460 ms/channel on an Intel iMac Pro — ~140×.
On a *shared* GPU set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and reduce
`--chunk`, or XLA's 75% preallocation fights other users and starves
the cuFFT workspace.

```bash
python scripts/bench_spectral.py                    # benchmark, default device
JAX_PLATFORMS=cpu python scripts/bench_spectral.py  # force CPU
```

## Running locally

```bash
# one-time setup (macOS)
brew install ffmpeg
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[test]"

# orbit movies (240 frames, ~10 min each on CPU)
.venv/bin/python -m hbtsim --system betaaur
.venv/bin/python -m hbtsim --system algol

# SNR tables
.venv/bin/python -m hbtsim.snr_cli --system algol --baseline 50 85

# tests (analytic validation suite)
.venv/bin/python -m pytest tests/ -v
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
    --compute-only --diameter 10 --baseline 85 --npz $SCRATCH/algol_keck.npz
# (note: standalone srun needs an explicit -n 1)

# or batch
sbatch ~/binary/scripts/perlmutter/bench.sbatch
```

Keep the repo and env in `$HOME` ($SCRATCH is purged after ~8 weeks);
write job outputs to `$SCRATCH`, then copy the npz back and render the
movie locally with `--render-only`.

## Package layout

- `hbtsim/params.py` — constants, `Star`/`BinarySystem`/`GridConfig`/
  `MovieConfig`, per-star LD tables, the `BETA_AUR` and `ALGOL`
  instances and the `SYSTEMS` registry (single source of truth)
- `hbtsim/orbit.py` — circular-orbit sky geometry
- `hbtsim/limbdark.py` — linear LD law + analytic disk visibility
- `hbtsim/render.py` — JAX rendering of the occulted limb-darkened disks
- `hbtsim/hbt.py` — FFT → |V|² → g²(B); analytic binary visibility
- `hbtsim/spectral.py` — batched multi-wavelength |V|²(B, λ) (GPU)
- `hbtsim/photometry.py` — band fluxes, AB magnitudes, anchoring
- `hbtsim/snr.py`, `hbtsim/snr_cli.py` — photon-budget SNR (telescopes,
  detectors, spectrograph multiplexing)
- `hbtsim/movie.py`, `hbtsim/cli.py` — the 3-panel orbit movie
- `hbtsim/g2spec.py` — the g²(λ)-with-error-bars movie
- `scripts/` — benchmark + Perlmutter setup/sbatch
- `tests/` — 34 analytic validation tests

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
