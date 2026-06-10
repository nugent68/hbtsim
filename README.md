# hbtsim — Intensity interferometry of binary stars

Simulation of the Hanbury Brown–Twiss (HBT) effect in optical intensity
interferometry for the eclipsing binary **Beta Aurigae (Menkalinan)**.
The code renders the binary as a pair of limb-darkened stellar disks on a
sky grid, computes the squared visibility |V|² via a zero-padded 2D FFT
(JAX), and produces a three-panel MP4 movie over one orbital period:

1. **Sky image** — the two stars orbiting each other (g band, to scale,
   milliarcsecond axes), including the partial eclipses.
2. **Lightcurve** — apparent SDSS g- and i-band magnitudes (synthetic
   monochromatic AB magnitudes at the band effective wavelengths) with a
   moving phase cursor.
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
apparent AB magnitudes from the physical flux at Earth,
f_λ = B_λ(T) × Ω_star (blackbody photospheres), giving g ≈ 2.26 out of
eclipse — within the few-tenths-of-a-magnitude accuracy expected of the
blackbody approximation compared with the observed V ≈ 1.90. The g²(B) panel orients the baseline
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
