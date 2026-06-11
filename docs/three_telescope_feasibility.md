# Three-telescope intensity interferometry: closure phases from Maunakea

**Question.** Two-point HBT measurements give only |V|²(B, λ) — the power
spectrum of the source, with no Fourier phase. Can a third telescope,
through the triple intensity correlation, recover the bispectrum (closure
phase) of our binaries, using Subaru + Keck I + Keck II?

**Answer in one line.** The geometry and the photon statistics make it
infeasible with the current SPAD Lambda configuration (centuries), but
R ≈ 5000 spectroscopy brings Maunakea to a heroic-but-conceivable
~1 month for Δcos φc ≤ 0.3 — and a *compact* (≲ 85 m) triangle of
10 m-class apertures would make closure phases of bright ~mas binaries
genuinely measurable in nights. Geometry beats aperture.

## 1. The triple correlation of thermal light

For Gaussian (chaotic) light, Wick's theorem gives the three-point
intensity correlation (Malvimat, Wucknitz & Saha 2013; Nuñez & Domiciano
de Souza 2015; Zmija et al. 2025):

    g³₁₂₃ = 1 + |γ₁₂|² + |γ₂₃|² + |γ₃₁|² + 2 Re[γ₁₂ γ₂₃ γ₃₁]

The last term is the **bispectrum**: amplitude 2|γ₁₂γ₂₃γ₃₁| and phase

    φc = arg(γ₁₂ γ₂₃ γ₃₁)   (the closure phase).

Because intensities are real, intensity interferometry measures only
**cos φc**. The closure phase is immune to per-telescope phase errors and
to source translation (the vector baselines close: B₁₂+B₂₃+B₃₁ = 0, so
linear phase ramps cancel) — it is the phase observable that two-point
HBT fundamentally lacks, and the entry point to image reconstruction.

## 2. Photon budget (extending the hbtsim pair formalism)

**Polarization.** Unpolarized light is two independent Gaussian modes
each carrying I/2. The pair interference terms scale as Σ(I/2)² → ½|γ|²
(the familiar pol factor in `snr.py`); the triple term scales as
2·Σ(I/2)³ → **pol₃ = ¼**.

**Signal.** Time-tag triples live on the 2D lag plane (τ₁, τ₂). For a
rectangular passband of width Δν, the lag-integrated triple excess is
exactly τ_c² = 1/Δν² (Parseval on the cubed unit-area spectrum — the 2D
analogue of the pair case's τ_c), so over integration time T:

    N_sig = pol₃ · 2|γ₁₂γ₂₃γ₃₁| cos φc · τ_c² · R₁R₂R₃ · T

with R_i the detected (dead-time-capped) rates per channel.

**Noise.** Accidental triples arrive at density b₁b₂b₃ per unit lag²
(b_i = R_i + dark + sky). The detector jitters (σ_i each) smear the
signal into a *correlated* 2D Gaussian — telescope 2's jitter enters
both lags with opposite signs — with covariance

    Σ = [[σ₁²+σ₂², −σ₂²], [−σ₂², σ₂²+σ₃²]],
    det Σ = σ₁²σ₂² + σ₂²σ₃² + σ₃²σ₁².

The matched-filter effective area is A₂D = (∫K)²/∫K² = **4π√(det Σ)**
(equal jitters: 4π√3 σ² = 5.65×10⁻²⁰ s² for three SPAD Lambdas at
σ = 51 ps). Hence

    SNR₃ = N_sig / √(b₁b₂b₃ · T · A₂D)

**Scalings** (verified in `tests/test_snr3.py`): SNR₃ ∝ √(A₁A₂A₃), ∝ √T,
∝ 1/σ_jitter (the pair SNR scales only as 1/√σ — timing is more
valuable here), and ∝ Δλ^(−1/2) at fixed source (the pair SNR is
bandwidth-independent) — so narrow channels with heavy spectral
multiplexing (× √N_channels) are the levers. These reduce to Nuñez &
Domiciano de Souza (2015) eq. 8 and Zmija et al. (2025) eq. 12.

**Criterion.** Estimating cos φc to precision Δcos requires
SNR₃(cos φc = 1) ≥ 1/Δcos; we quote times for Δcos φc ≤ 0.3 and ≤ 0.1.

## 3. Geometry

Site coordinates (Subaru 19°49′32″ N 155°28′34″ W; Keck I 19.8259465 N
155.474719 W; Keck II 19.8265606 N 155.474234 W) give local ENU
positions, relative to Subaru: Keck I (145.8 E, 43.3 N) m, Keck II
(196.6 E, 111.3 N) m, i.e. pairwise

| Baseline | Length |
|---|---|
| Subaru – Keck I | 152.1 m |
| Keck I – Keck II | 84.9 m |
| Keck II – Subaru | 225.9 m |

confirming the nominal 150 / 85 / 225 m. Summit elevations agree to
~20 m, so we use a flat layout with the source at zenith (a documented
simplification; hour-angle projection shortens the effective baselines
and would only help). `bispectrum.MAUNAKEA_SUBARU_KECK` carries this
triangle; apertures are Subaru 8.2 m and 2 × Keck 10 m.

## 4. The triple amplitude is geometry-crushed

Figures: `output/g3_gammas_{algol,betaaur}.png` (per-pair |γ|(λ) and the
triple product at quadrature), `output/g3_cosphi_{algol,betaaur}.png`
(cos φc over wavelength × orbital phase, FFT path, valid through
eclipses).

The Keck I–Keck II 85 m pair retains healthy coherence (|γ| up to 0.66
for Algol around 700 nm), but the two Subaru arms (152 m, 226 m) sit at
or beyond the disks' first nulls over most of the band (Beta Aur's
primary null is at 98 m at 400 nm, 207 m at 850 nm; Algol A's at 114 m
at 400 nm). Since the bispectrum needs all three, the triple amplitude
|γ₁₂γ₂₃γ₃₁| peaks at only **0.041 (Algol, 950 nm)** and **0.013
(Beta Aur, 776 nm)** — one to two orders of magnitude below what a
compact triangle would see.

The closure-phase maps are nonetheless scientifically rich: cos φc
flips sign at wavelengths set by the disk nulls and the binary fringe,
the flip pattern sweeps with orbital phase, and it reorganizes sharply
through the eclipses — exactly the phase information that two-telescope
HBT can never see, and the dataset an imaging reconstruction would
consume.

## 5. Feasibility

Nights (8 h) of integration for the spectrally-multiplexed bispectrum to
reach Δcos φc ≤ 0.3 / ≤ 0.1, at quadrature, computed by
`scripts/feasibility_g3.py` with the photon budget of section 2
(throughput 0.3, SPAD Lambda PDE/jitter/dead time, anchored magnitudes):

| System | Triangle | Channels | SNR₃ (mux) per h | Nights, Δcos ≤ 0.3 | Nights, Δcos ≤ 0.1 |
|---|---|---|---|---|---|
| Algol | Subaru+Keck+Keck | 320 × 1.72 nm | 4.3×10⁻³ | 77,000 (**~260 yr**) | 690,000 (**~2,400 yr**) |
| Algol | Subaru+Keck+Keck | 5500 × 0.10 nm | 0.18 | **42** | 382 |
| Algol | compact 85 m (3×10 m) | 320 × 1.72 nm | 0.11 | 116 | 1,050 |
| Algol | compact 85 m (3×10 m) | 5500 × 0.10 nm | 5.5 | **0.05 (≈ 22 min)** | **0.41 (≈ 3.3 h)** |
| Beta Aur | Subaru+Keck+Keck | 320 × 1.72 nm | 3.2×10⁻³ | 137,000 | 1.2×10⁶ |
| Beta Aur | Subaru+Keck+Keck | 5500 × 0.10 nm | 0.11 | 109 | 982 |
| Beta Aur | compact 85 m (3×10 m) | 5500 × 0.10 nm | 6.8 | 0.03 | 0.27 |
| **Spica** | Subaru+Keck+Keck | 5500 × 0.10 nm | **1.42** | **0.7** | **6.2** |
| Spica | Subaru+Keck+Keck | 320 × 1.72 nm | 1.4×10⁻² | 6,700 | 60,200 |

Per-channel detected rates are 1.2–7.5×10⁷ cps (dead-time-saturated
blueward of ~650 nm at 1.72 nm channels); the photon occupancy is
R·τ_c ≈ 3×10⁻⁵ per coherence time — the n^{3/2} penalty relative to the
pair correlation's n is the fundamental difficulty of g³ on thermal
starlight.

Context (Zmija et al. 2025, Table 2): H.E.S.S. (3×100 m², 5 ns, 10 nm,
one channel) needs ~1100–2400 yr for the same criterion on m_B ≈ 2
stars; their CTA-LST projection (4×400 m², 0.1 ns, 0.1 nm, 1000
channels) reaches ~2–5 months. Our Maunakea-with-R≈5000 numbers are
consistent with that once the smaller collecting areas and the
geometry-suppressed |γ₁₂γ₂₃γ₃₁| are accounted for.

## 5b. Target selection: Spica makes the real triangle work

What the long arms punish is the stellar *disk* size, not the binary
separation (the separation only sets the fringe period — that is
signal). The figure of merit at fixed apparent flux is **surface
brightness**: hotter photospheres pack the same flux into a smaller
disk, keeping |γ| alive at 150–226 m. The optimal class is therefore
bright early-B close binaries with disks ≲ 0.5–0.9 mas and separations
~0.5–2 mas.

**Spica (α Vir)** is the textbook case, now in the package as `SPICA`
(`--system spica`): V = 0.97, B1 III-IV + B2 V (25,300/20,900 K,
θ = 0.91/0.45 mas — the primary's diameter was itself measured by the
Narrabri *intensity interferometer*, Herbison-Evans et al. 1971),
ρ = 1.71 mas, P = 4.01 d. Its triple amplitude on the Maunakea triangle
is no better than Algol's (0.036 — bright means near, and the B giant
still subtends 0.9 mas), but the 3.5× higher photon flux enters as
R^{3/2} in the unsaturated narrow-channel regime:

**With the R ≈ 5000 backend, Subaru + Keck I + Keck II reaches
Δcos φc ≤ 0.3 on Spica in ~0.7 night and ≤ 0.1 in ~6 nights** — the
real triangle becomes genuinely feasible with the right target, no
compact array required. (The stock 320-channel SPAD Lambda still needs
~6,700 nights: the spectroscopic backend remains non-negotiable.)

Spica caveats: the true orbit has e = 0.108 with apsidal motion
(approximated circular here); the primary is a β Cep pulsator and
tidally distorted (rendered as a static sphere); it is non-eclipsing
at i = 63°.

## 5c. Four telescopes: the VLT Unit Telescopes

Putting SPAD Lambdas with R ≈ 5000 backends on the four 8.2 m VLT UTs
(`bispectrum.VLT_UT`; published station coordinates reproduce the
pairwise separations 46.6 / 56.5 / 62.4 / 89.3 / 102.4 / 130.2 m to
≤ 0.2 m) changes the problem qualitatively:

- **All six baselines are short.** 46.6–130.2 m sits inside Spica's
  first null across the band, so every pair keeps |γ| ≈ 0.4–0.75 and the
  four triangles reach triple amplitudes 0.19–0.39 — an order of
  magnitude above the Maunakea triangle.
- **Four triangles at once.** Of the four closure phases, three are
  independent ((N−1)(N−2)/2), but all four triple-coincidence streams
  carry independent accidental noise and add in quadrature; six |V|²
  baselines come along simultaneously for free, giving genuine snapshot
  (u, v) coverage — the minimal configuration for model-independent
  imaging rather than model fitting.
- **Result for Spica** (transits at 77° at Paranal): combined bispectrum
  sensitivity ≈ 28/√h per unit cos φc — **Δcos φc ≤ 0.3 in ~1 minute,
  ≤ 0.1 in ~8 minutes**. The orbit (P = 4.01 d) can be tiled with
  closure-phase measurements every few minutes over a night: a
  closure-phase *curve*, not a single number.
- **Limiting magnitude.** Time scales as flux⁻³; for similar geometry a
  one-night Δcos φc ≤ 0.1 measurement works down to **g ≈ 2.2** —
  several dozen hot southern binaries and rapid rotators qualify
  (α Cen's neighborhood of bright B stars: β Cen, α Lup, λ Sco,
  β Cru ...). With the stock 320-channel SPAD Lambda instead of R ≈ 5000
  the same measurement needs ~90 nights — the spectroscopic backend
  remains the enabling hardware.
- **Declination caveat**: Paranal (−24.6°) cannot usefully observe Algol
  or Beta Aurigae (culminating below ~25°); the VLT numbers are for
  southern targets, with Spica the natural first light.

Like Maunakea, the UTs already host amplitude interferometry (VLTI);
the II niches are the same as section 5b — absolute |V|² calibration,
the blue, no beam combination or delay lines (each UT independently
time-tags photons), and validation of the technique toward km-baseline
arrays.

## 6. Caveats

- **Aperture smearing**: 8–10 m apertures on 85–226 m baselines average
  the complex visibility over B ± D/2 — fringe periods λ/ρ are 25–60 m,
  so this suppresses (and slightly biases) the triple product; not
  modeled (would reduce feasibility further).
- **Flat-layout / zenith** baselines; real hour-angle tracks shorten and
  rotate the projected triangle (generally helpful for these
  over-resolved disks).
- **Pair-term ridges**: the |γ_ij|² terms form ridges crossing the
  (τ₁, τ₂) bump; they bias the triple estimator and must be subtracted
  using the simultaneously-measured g²'s (the H.E.S.S. analysis does
  exactly this with 2D Gaussian-tube fits).
- At 0.1 nm channels τ_c (~10–20 ps) is no longer ≪ σ (51 ps): the
  matched-filter kernel should be broadened by the coherence envelope
  (an O(1) correction in the optimistic direction of our quoted times).
- **Algol C**: the ~10% incoherent third light dilutes every γ by ~0.9
  and the triple product by ~0.73 unless C is excluded optically.
- Detector saturation: per-channel rates at 1.72 nm are dead-time-capped
  blueward of ~650 nm; quoted numbers include the non-paralyzable model.

## 7. Conclusions

1. **Subaru + Keck I + Keck II with the stock SPAD Lambda cannot measure
   closure phases of these binaries** — the required integrations are
   measured in centuries. Two independent suppressions stack: photon
   occupancy (R·τ_c ~ 3×10⁻⁵ per coherence time, and SNR₃ ∝ occupancy^{3/2})
   and geometry (the 152/226 m Subaru arms resolve the ~1 mas disks past
   their first nulls, crushing |γ₁₂γ₂₃γ₃₁| to ~10⁻²).
2. **Spectroscopy is the biggest practical lever**: an R ≈ 5000
   dispersing backend (0.1 nm channels) shortens the time by ~Δλ
   (saturated regime) to ~10² — bringing Δcos φc ≤ 0.3 within ~a month
   of dedicated time. This is the same conclusion CTA-LST studies reach.
3. **Geometry beats aperture** for resolved ~1 mas disks: a compact
   ≲ 85 m triangle (e.g. Keck I + Keck II + a third 10 m-class aperture)
   raises the triple amplitude by 1–2 orders of magnitude and makes
   bright-binary closure phases measurable in nights with fine spectral
   channels. If the goal is imaging ~mas-scale bright stars through the
   bispectrum, the array to build is compact and many-channeled, not
   long-armed.
3b. **Target selection rescues the real triangle**: high-surface-
   brightness early-B binaries keep |γ| alive on the long arms while
   delivering R^{3/2} photons — Spica reaches Δcos φc ≤ 0.3 in under a
   night and ≤ 0.1 in ~6 nights on Subaru + Keck I + Keck II with the
   R ≈ 5000 backend (section 5b).
4. For image reconstruction proper, one triangle gives one closure phase
   per (λ, t); the λ-dependence across 320–5500 channels plus the
   orbital phase dependence is the dataset — Nuñez & Domiciano de Souza
   (2015) found useful reconstructions need bispectrum SNR ≳ 30 with
   ~10³ channels, consistent with the times quoted here.

## References

- Malvimat, V., Wucknitz, O. & Saha, P. 2013, MNRAS (arXiv:1304.3391)
- Nuñez, P. D. & Domiciano de Souza, A. 2015, MNRAS (arXiv:1507.07635)
- Zmija, A. et al. 2025, MNRAS (arXiv:2512.13485) — first astrophysical
  g³ measurement (H.E.S.S.), sensitivity analysis, CTA projection
- Rai, Basak & Saha 2021 (arXiv:2105.09532); Guerin et al. 2025
  (arXiv:2503.22446) — pair formalism this work extends
