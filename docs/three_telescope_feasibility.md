# Three-telescope intensity interferometry: closure phases

**Question.** Two-point HBT measurements give only |V|²(B, λ) — the power
spectrum of the source, with no Fourier phase. Can a third telescope,
through the triple intensity correlation, recover the bispectrum (closure
phase) of our binaries — with Subaru + Keck I + Keck II on Maunakea, or
with the four VLT Unit Telescopes?

**Answer in one line.** With the SPAD Lambda as delivered, no: its
time-tag link cannot carry the 10¹⁰–10¹¹ photons/s a bright star sends to
an 8–10 m telescope, and even with a correlator readout the 320 channels
need ~150 nights on the best case. A next-generation, R = 5000,
correlator-readout SPAD array on the four VLT UTs detects Spica's
closure-phase *signal* (a template amplitude over 4325 channels) in
36 minutes (7 with a polarizing beamsplitter) and δ Velorum's in
1.7–5 hours; spectrally resolved closure-phase *curves* at R = 100 take
nights per epoch, and single 0.1 nm channels are out of reach. On
Maunakea the same backend needs ~11 nights for Spica, and only because
the rotating long arms sweep through favourable geometry over the night.
The critical systematic is the calibration of the pair-correlation
kernel to ~10⁻³.

All numbers below come from `scripts/feasibility_g3.py` (instrument model
of `hbtsim.snr`: throughput 0.3 × 0.5 × PDE for dispersed channels,
aperture averaging, orbits at their true sky orientation, the source at
transit, unpolarized light unless stated).

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
(p₂ = ½ in `snr.py`); the triple term scales as 2·Σ(I/2)³ → **p₃ = ¼**.
A polarizing beamsplitter feeding two detectors per telescope makes each
stream fully coherent (p₂ = p₃ = 1 on half the rate); the streams add in
quadrature for ×√2 in SNR₂ and **×2 in SNR₃**, while halving the
per-pixel load (`Observation.polarization_mode="pbs"`).

**Signal.** Time-tag triples live on the 2D lag plane (τ₁, τ₂). For a
rectangular passband of width Δν, the lag-integrated triple excess is
exactly τ_c² = 1/Δν² (Parseval on the cubed unit-area spectrum — the 2D
analogue of the pair case's τ_c), so over integration time T:

    N_sig = p₃ · 2|⟨γ₁₂γ₂₃γ₃₁⟩| cos φc · τ_c² · R₁R₂R₃ · T

with R_i the detected (dead-time-capped) rates per channel and the
triple product averaged exactly over the three pupils
(`hbtsim.aperture.TripleQuadrature`; smearing each γ with its pair
kernel would apply every pupil twice).

**Noise.** Accidental triples arrive at density b₁b₂b₃ per unit lag²
(b_i = R_i + dark + sky). The detector jitters (σ_i each) smear the
signal into a *correlated* 2D Gaussian — telescope 2's jitter enters
both lags with opposite signs — with covariance

    Σ = [[σ₁²+σ₂², −σ₂²], [−σ₂², σ₂²+σ₃²]] + σ_c² [[1, −½], [−½, 1]],

the second term being the coherence broadening (σ_c = 0.376 τ_c, ≲1% at
0.1 nm channels in the red). The matched-filter effective area is
A₂D = (∫K)²/∫K² = **4π√(det Σ)** (equal jitters, no broadening:
4π√3 σ² = 5.65×10⁻²⁰ s² for three SPAD Lambdas at σ = 51 ps). Hence

    SNR₃ = N_sig / √(b₁b₂b₃ · T · A₂D)

**Scalings** (verified in `tests/test_snr3.py`, including a hand-computed
first-principles normalization): SNR₃ ∝ √(A₁A₂A₃), ∝ √T, ∝ 1/σ_jitter
(the pair SNR scales only as 1/√σ — timing is more valuable here), and
∝ Δλ^(−1/2) at fixed source (the pair SNR is bandwidth-independent) — so
narrow channels with heavy spectral multiplexing (× √N_channels) are the
levers. These reduce to Nuñez & Domiciano de Souza (2015) eq. 8 and
Zmija et al. (2025) eq. 12.

**Pair ridges.** The same lag-plane histogram carries the pair
correlations as ridges (|γ₁₂|² along τ₁ = 0 for every τ₂, …) whose
excess inside the triple window exceeds the triple term by

    ridge ratio = Σ_pairs (p₂/2p₃) |γᵢⱼ|² A₂D / (2√π σᵢⱼ τ_c |γ₁₂γ₂₃γ₃₁|)

— 50–800 at R = 5000 (10³–10⁴ at 320 channels). Subtracting them with the
simultaneously measured g²'s is statistically cheap, but a fractional
error ε in the modeled kernel shape biases cos φc by ε × ridge ratio:
the kernel must be calibrated to ~10⁻³ for Δcos φc = 0.1
(`SpectralSNR3Result.required_kernel_accuracy`).

**Statistics.** With one cos φc per channel, "measuring the closure
phase" must be defined. `snr3.time_to_precision` inverts three
statistics: **amplitude** — SNR_amp = √Σ(SNR_ch · cos φc,ch)², one
global amplitude on the model's per-channel cos φc template (the
detection statistic; it honours the model's own sign changes across the
band); **binned** — closure phases binned to R = 100 (median bin);
**channel** — one 0.1 nm channel (median). They differ by 10²–10⁴ in
time. Estimating cos φc to precision Δ needs the statistic's SNR at
cos φc = 1 to reach 1/Δ; we quote Δcos φc ≤ 0.1.

**Hour angle.** The projected baselines rotate; a fringe drifts by
(B(t₂)−B(t₁))·ρ/λ cycles between epochs — 1/8 cycle in 17 min for Spica
on the 130 m UT1–UT4 arm at 400 nm, 4 min for δ Vel's 15 m fringes.
`snr3.track_g3_snr` integrates a night in blocks of that length at the
projected (u, v) with the orbital phase advanced, applies the sinc loss
of the residual drift, and combines the blocks as a template fit (never
coherently).

## 3. Geometry

Maunakea site coordinates (Subaru 19°49′32″ N 155°28′34″ W; Keck I
19.8259465 N 155.474719 W; Keck II 19.8265606 N 155.474234 W) give local
ENU positions relative to Subaru: Keck I (145.8 E, 43.3 N) m, Keck II
(196.6 E, 111.3 N) m, i.e. pairwise

| Baseline | Length |
|---|---|
| Subaru – Keck I | 152.1 m |
| Keck I – Keck II | 84.9 m |
| Keck II – Subaru | 225.9 m |

confirming the nominal 150 / 85 / 225 m. The VLT UTs (published station
coordinates) span 46.6–130.2 m. `Triangle.projected(H, dec)` and
`Array.projected` rotate the stations into the (u, v) plane for a real
hour angle (`hbtsim.geometry`; UT1–UT4 is 129.4 m at Spica's transit),
and every system carries the position angle of its ascending node
(β Aur 295.15°, Algol 43.43°, Spica 131.6°, δ Vel 65.0°) so that closure
phases on a fixed ground triangle are computed at the true orientation.

## 4. Aperture averaging and the triple amplitude

Figures: `output/g3_gammas_{spica,deltavel}_vlt.png` (per-pair |γ|(λ),
point and pupil-averaged, and the three-pupil-averaged triple product of
each triangle at quadrature), `output/g3_cosphi_{spica,deltavel}_vlt.png`
(cos φc over wavelength × orbital phase on the UT1–UT2–UT4 triangle).

A binary fringe of period P = λ/ρ sampled by pupils D₁, D₂ keeps
A(πD₁/P)·A(πD₂/P) of its contrast, A(x) = 2J₁(x)/x: for 8.2 m pupils at
400 nm that is 0.93 for Spica (ρ = 1.7 mas), 0.89 for Algol, 0.76 for
β Aur (3.3 mas) and **0.46 for δ Vel at maximum separation** (5.5 mas,
P = 15 m). On the VLT every Spica baseline lies inside the disks' first
null at the true orientation, |γ| = 0.1–0.95, and the three-pupil-averaged
triple amplitudes reach 0.3–0.6 in the red; δ Vel's unresolved disks
give 0.68–0.86 around the orbit. On Maunakea the two Subaru arms
(152, 226 m) sit at or beyond the disks' first nulls over most of the
band, and the triple amplitude at transit peaks at only **0.04 (Spica,
Algol)** and **0.01 (β Aur)**.

## 5. Feasibility

Integration for Δcos φc ≤ 0.1 (8-hour nights), snapshot at quadrature
(δ Vel: range over orbital phases 0.25–0.9):

| System | Array | Backend | SNR_amp / √h | amplitude | R = 100 bins | one channel |
|---|---|---|---|---|---|---|
| Spica | VLT 4×UT | current SPAD Lambda, 320 ch, time-tag link | 9×10⁻⁵ | — | — | — |
| Spica | VLT 4×UT | current SPAD Lambda, 320 ch, correlator | 0.29 | 151 nights | 2.6×10⁴ nights | 8×10⁵ nights |
| **Spica** | **VLT 4×UT** | **next-gen R = 5000, correlator** | **12.9** | **36 min** | **13.6 nights** | 1.2×10⁴ nights |
| Spica | VLT 4×UT | next-gen R = 5000 + PBS | 28.7 | 7.3 min | 2.8 nights | 2.5×10³ nights |
| δ Vel | VLT 4×UT | current, 320 ch, correlator | 0.12–0.23 | 230–810 nights | ≥1.7×10⁵ nights | ≥5×10⁶ nights |
| **δ Vel** | **VLT 4×UT** | **next-gen R = 5000, correlator** | **4.4–7.7** | **1.7–5.1 h** | **280–800 nights** | ≥9×10⁴ nights |
| δ Vel | VLT 4×UT | next-gen R = 5000 + PBS | 9.1–16.1 | 23–73 min | 66–190 nights | ≥2×10⁴ nights |
| Spica | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.17 | 430 nights | 1.5×10⁴ nights | 8×10⁵ nights |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 + PBS | 0.38 | 88 nights | 3×10³ nights | 1.5×10⁵ nights |
| Algol | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.045 | 6.3×10³ nights | 1.9×10⁶ nights | 9×10⁷ nights |
| β Aur | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.007 | 2.8×10⁵ nights | 1.4×10⁸ nights | 6×10⁹ nights |

The current SPAD Lambda's time-tag link (≈10⁸ events/s per detector, an
estimate from its two USB3 links) is 200–800× below what these stars
deliver to 8–10 m telescopes; with the rates attenuated to the link the
time-tag rows are 10⁸–10¹⁵ nights on every target, i.e. no statistic is
reachable with the detector as delivered. The 320-channel array also
drives single pixels to dead-time loads R·τ_dead = 1.3–5.5, outside the
non-paralyzable model; the R = 5000 backend brings them to 0.1–0.3.

Integrated along the uv track (one night, blocks short enough that the
fringe drifts ≤ 1/8 cycle; blocks combined as a template fit):

| System | Array | Backend | nights to Δcos φc ≤ 0.1 (amplitude) | (R = 100 bins) |
|---|---|---|---|---|
| Spica | VLT | next-gen R = 5000 | 0.10 (≈ 47 min of the 8.3 h window) | 11.7 |
| Spica | VLT | next-gen R = 5000 + PBS | 0.02 | 2.5 |
| δ Vel | VLT | next-gen R = 5000 (4-min blocks) | 0.58 | 195 |
| δ Vel | VLT | next-gen R = 5000 + PBS (4-min blocks) | 0.13 | 46 |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 | **10.8** (snapshot: 430) | 1.3×10³ |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 + PBS | 2.2 | 260 |
| Algol | Subaru+Keck+Keck | next-gen R = 5000 | 200 | 3.5×10⁴ |

On Maunakea the track is the measurement: the transit snapshot sits at a
near-null orientation of the long arms, and the rotating (u, v) points
sweep through geometry 40× more favourable over the night.

Context (Zmija et al. 2025, Table 2): H.E.S.S. (3×100 m², 5 ns, 10 nm,
one channel) needs ~1100–2400 yr for the same criterion on m_B ≈ 2 stars
(reproduced to within an order of magnitude by `tests/test_snr3.py` under
their stated assumptions); their CTA-LST projection (4×400 m², 0.1 ns,
0.1 nm, 1000 channels) reaches ~2–5 months.

## 5b. Target selection: surface brightness

What the long arms punish is the stellar *disk* size, not the binary
separation (the separation only sets the fringe period — that is
signal, until the fringe period approaches the pupil diameter, as for
δ Vel). The figure of merit at fixed apparent flux is **surface
brightness**: hotter photospheres pack the same flux into a smaller
disk, keeping |γ| alive at 50–130 m and even at 150–226 m. Spica
(V = 0.97, B1 III-IV + B2 V, θ = 0.91/0.45 mas, ρ = 1.71 mas, P = 4.01 d)
is the textbook case; its 3.5× higher photon flux than Algol enters as
R^{3/2}. δ Vel (V = 1.95, A2 IV + A4 V, 0.34/0.29 mas) is the other
extreme: unresolved disks and the largest triple amplitudes, paid for by
the flux⁻³ scaling and the pupil averaging of its 15 m blue fringes.

## 5c. Four telescopes: the VLT Unit Telescopes

Putting next-generation SPAD arrays with R = 5000 backends and
on-detector correlators on the four 8.2 m VLT UTs changes the problem
qualitatively:

- **All six baselines are short.** 46.6–130.2 m sits inside Spica's
  first null across the band.
- **Four triangles at once.** Of the four closure phases, three are
  independent ((N−1)(N−2)/2), but all four triple-coincidence streams
  carry independent accidental noise and add in quadrature; six |V|²
  baselines come along simultaneously — genuine snapshot (u, v)
  coverage, the minimal configuration for model-independent imaging.
- **Result for Spica** (transits at 77° at Paranal): the template
  detection in 36 min (7 min with a beamsplitter); R = 100 closure-phase
  curves in 13.6 (2.8) nights per epoch.
- **Limiting magnitude.** Time scales as flux⁻³; a one-night template
  detection at Δcos φc ≤ 0.1 works down to **g ≈ 1.7 unpolarized,
  g ≈ 2.2 with the beamsplitter** — a dozen to several dozen hot
  southern binaries and rapid rotators (β Cen, λ Sco, β Cru, α Eri …).
- **Declination caveat**: Paranal (−24.6°) never sees Algol or β Aur
  above 30°; the VLT numbers are for southern targets.

## 6. Caveats

- **Kernel calibration** (the critical systematic): ridge ratios of ~100
  at R = 5000 require the pair-correlation kernel shape to 10⁻³.
- **Readout**: the on-detector correlator assumed for the next-generation
  device is a development item; the 10⁸ cps time-tag ceiling of the
  current SPAD Lambda is an estimate to be confirmed with Pi Imaging.
- **SEDs**: blackbody surface fluxes with observed anchors set the flux
  ratio of the two stars (hence the fringe contrast) only to tens of
  per cent in the blue for Algol; the model-atmosphere hooks
  (`hbtsim.sed`, NewEra PHOENIX) remove this once angle-resolved spectra
  for these parameters are available.
- **Algol C**: the ~10% incoherent third light dilutes every γ by ~0.9
  and the triple product by ~0.73 unless C is excluded optically.
- Orbital physics not modeled: Spica's e = 0.108 and apsidal motion, its
  β Cep pulsations and tidal distortion; δ Vel's rotational oblateness
  (the signal a campaign would target); the δ Vel ω convention should be
  checked against the observed eclipse timing before it is used for
  timing work.

## 7. Conclusions

1. **The SPAD Lambda as delivered cannot measure closure phases of these
   binaries** on any large telescope: its time-tag link caps the photon
   rate 200–800× below what the stars deliver, and even with a
   correlator readout its 320 channels need ~150 nights on the best
   case (Spica, VLT).
2. **Spectral resolution plus on-detector correlation is the enabling
   hardware**: an R = 5000 backend brings the Spica template detection on
   the VLT to 36 minutes (7 with a polarizing beamsplitter), a factor of
   ~2000 in time over 320 channels.
3. **Detection and imaging are different measurements.** The template
   amplitude comes in minutes; R = 100 closure-phase curves take nights
   per epoch; single 0.1 nm channels are out of reach. Image
   reconstruction (Nuñez & Domiciano de Souza 2015: bispectrum SNR ≳ 30
   in ~10³ channels) needs the multi-night regime.
4. **Geometry and tracking**: on Maunakea the long Subaru arms resolve
   ~1 mas disks past their first nulls (triple amplitude ≲ 0.04 at
   transit), yet the uv track recovers a 40× better night than the
   snapshot — 11 nights for Spica's template with the next-generation
   backend. Compact ≲ 130 m arrays of 8–10 m apertures remain the right
   geometry for milliarcsecond bright stars.
5. **The systematics budget is set by the pair ridges**: a 10⁻³ kernel
   calibration, not photon statistics, is the hard requirement for a
   closure-phase measurement at Δcos φc = 0.1.

## References

- Malvimat, V., Wucknitz, O. & Saha, P. 2013, MNRAS (arXiv:1304.3391)
- Nuñez, P. D. & Domiciano de Souza, A. 2015, MNRAS (arXiv:1507.07635)
- Zmija, A. et al. 2025, MNRAS (arXiv:2512.13485) — first astrophysical
  g³ measurement (H.E.S.S.), sensitivity analysis, CTA projection
- Rai, Basak & Saha 2021 (arXiv:2105.09532); Guerin et al. 2025
  (arXiv:2503.22446) — pair formalism this work extends
