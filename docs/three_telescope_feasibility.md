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
36 minutes (7 with a polarizing beamsplitter); spectrally resolved
closure-phase *curves* at R = 100 take nights per epoch, and single
0.1 nm channels are out of reach. δ Velorum, at its corrected 25.1 pc,
takes 20–27 nights on the VLT at maximum separation and ~20 nights over
an orbit with three EON-SII 4 m units at Paranal; on either array only
the symmetric part of its closure-phase signal is within reach (§5g). On
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
β Aur (3.3 mas) and **0.02 for δ Vel at maximum separation** (17.5 mas
at its correct distance of 25.1 pc, P = 4.7 m, D/P = 1.74 — an earlier
revision carried δ Vel at 80.6 pc, three times too far, which gave
5.5 mas and 0.46). The pupil quadrature scales its node count with D/P
beyond the validated 0.8 and refuses above 3 (`hbtsim.aperture`). On the
VLT every Spica baseline lies inside the disks' first null at the true
orientation, |γ| = 0.1–0.95, and the three-pupil-averaged triple
amplitudes reach 0.3–0.6 in the red; δ Vel's 1.10/0.93 mas disks are
resolved on the longer arms and at maximum separation its fringe is
gone. Near conjunction (ρ ≲ 5 mas) the fringe returns: 8.2 m pupils keep
> 50 % of the 400 nm contrast for 21 % of the orbit (uniform in phase),
4 m pupils for 46 % (`scripts/deltavel_eonsii.py`, §5g). On Maunakea the two Subaru arms
(152, 226 m) sit at or beyond the disks' first nulls over most of the
band, and the triple amplitude at transit peaks at only **0.04 (Spica,
Algol)** and **0.01 (β Aur)**.

## 5. Feasibility

Integration for Δcos φc ≤ 0.1 (8-hour nights), snapshot at quadrature
(δ Vel: at maximum separation, 25.1 pc; rows marked NewEra use the model
atmospheres of §5d, the others blackbody + Claret; the multi-triangle
R = 100 bins now combine the four VLT triangles in quadrature, which the
previous revision under-counted by up to √N):

| System | Array | Backend | SNR_amp / √h | amplitude | R = 100 bins | one channel |
|---|---|---|---|---|---|---|
| Spica | VLT 4×UT | current SPAD Lambda, 320 ch, time-tag link | 1.5×10⁻⁴ | — | — | — |
| Spica | VLT 4×UT | current SPAD Lambda, 320 ch, correlator | 0.29 | 151 nights | 1.9×10⁴ nights | 8×10⁵ nights |
| **Spica** | **VLT 4×UT** | **next-gen R = 5000, correlator** | **12.9** | **36 min** | **8.6 nights** | 1.2×10⁴ nights |
| Spica | VLT 4×UT | next-gen R = 5000 + PBS | 28.7 | 7.3 min | 14 h | 2.5×10³ nights |
| δ Vel | VLT 4×UT | current, 320 ch, correlator | 0.029 | 1.5×10⁴ nights | 2.2×10⁶ nights | 5×10⁷ nights |
| δ Vel | VLT 4×UT | next-gen R = 5000, correlator | 0.68 | 27 nights | 3.7×10³ nights | 2.3×10⁶ nights |
| δ Vel | VLT 4×UT | next-gen R = 5000 + PBS | 1.41 | 6.3 nights | 900 nights | 5×10⁵ nights |
| δ Vel, NewEra | VLT 4×UT | next-gen R = 5000, correlator | 0.78 | 20 nights | 2.6×10³ nights | 1.6×10⁶ nights |
| δ Vel, NewEra | VLT 4×UT | next-gen R = 5000 + PBS | 1.63 | 4.7 nights | 610 nights | 4×10⁵ nights |
| Spica | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.17 | 430 nights | 1.5×10⁴ nights | 8×10⁵ nights |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 + PBS | 0.38 | 88 nights | 3×10³ nights | 1.5×10⁵ nights |
| Algol | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.045 | 6.3×10³ nights | 1.9×10⁶ nights | 9×10⁷ nights |
| Algol, NewEra (A) | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.041 | 7.5×10³ nights | 1.8×10⁶ nights | 9×10⁷ nights |
| β Aur | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.007 | 2.8×10⁵ nights | 1.4×10⁸ nights | 6×10⁹ nights |
| β Aur, NewEra | Subaru+Keck+Keck | next-gen R = 5000, correlator | 0.0054 | 4.3×10⁵ nights | 2.1×10⁸ nights | 9×10⁹ nights |

The current SPAD Lambda's time-tag link (1.4 × 10⁸ events/s per detector,
the manufacturer's 140 Mcps) is 150–600× below what these stars
deliver to 8–10 m telescopes; with the rates attenuated to the link the
time-tag rows are 6 × 10⁸–4 × 10¹⁴ nights on every target, i.e. no statistic is
reachable with the detector as delivered. The 320-channel array also
drives single pixels to dead-time loads R·τ_dead = 1.3–5.5, outside the
non-paralyzable model; the R = 5000 backend brings them to 0.1–0.3.

Integrated along the uv track (one night, blocks short enough that the
fringe drifts ≤ 1/8 cycle; blocks combined as a template fit):

| System | Array | Backend | nights to Δcos φc ≤ 0.1 (amplitude) | (R = 100 bins) |
|---|---|---|---|---|
| Spica | VLT | next-gen R = 5000 | 0.10 (≈ 47 min of the 8.3 h window) | 9.1 |
| Spica | VLT | next-gen R = 5000 + PBS | 0.02 | 1.9 |
| δ Vel | VLT | — | not tracked (fringe smeared at maximum separation, §4); three EON-SII units: §5g | — |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 | **10.8** (snapshot: 430) | 1.3×10³ |
| Spica | Subaru+Keck+Keck | next-gen R = 5000 + PBS | 2.2 | 260 |
| Algol | Subaru+Keck+Keck | next-gen R = 5000 | 200 | 3.5×10⁴ |
| Algol, NewEra (A) | Subaru+Keck+Keck | next-gen R = 5000 | 241 | 4.0×10⁴ |
| β Aur, NewEra | Subaru+Keck+Keck | next-gen R = 5000 | 3.2×10³ | 3.7×10⁶ |

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
R^{3/2}. δ Vel (V = 1.95, A2 IV + A4 V, 1.10/0.93 mas at 25.1 pc) is the
other extreme: a 16.6 mas orbit whose blue fringe period (4.7 m) is
below the 8.2 m UT pupils, so at maximum separation the VLT sees only
the two disk envelopes (the fringe returns near conjunction, for ~21 % of
the orbit). It is a better match for 1–4 m apertures (§5e): on the C2PU
pair or the EON-SII 4 m pair the fringe survives (D/P = 0.2–0.85) and the
two-telescope g² tables below include it; three EON-SII units are
forecast in §5g.

## 5c. Four telescopes: the VLT Unit Telescopes

Putting next-generation SPAD arrays with R = 5000 backends and
real-time correlators (no link limit) on the four 8.2 m VLT UTs changes the problem
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
  curves in 8.6 nights (14 h) per epoch.
- **Limiting magnitude.** Time scales as flux⁻³; a one-night template
  detection at Δcos φc ≤ 0.1 works down to **g ≈ 1.7 unpolarized,
  g ≈ 2.2 with the beamsplitter** — a dozen to several dozen hot
  southern binaries and rapid rotators (β Cen, λ Sco, β Cru, α Eri …).
- **Declination caveat**: Paranal (−24.6°) never sees Algol or β Aur
  above 30°; the VLT numbers are for southern targets.
- **δ Vel is a poor VLT closure-phase target at maximum separation**
  (the phase of its rows above): the corrected distance puts its blue
  fringe period at 4.7 m against 8.2 m pupils (2 % of the 400 nm
  contrast left). Near conjunction the fringe returns (> 50 % contrast
  for 21 % of the orbit), but there the closure phase is nearly
  symmetric (§5g). A transportable EON-SII triangle is the preferred
  instrument.

## 5d. Effect of the NewEra model atmospheres

Every number above the line was blackbody + Claret & Bloemen linear
limb darkening; the rows marked NewEra use the angle-resolved PHOENIX
NewEra tables (Hauschildt et al. 2025; A-star box 8000–12 000 K ×
log g 3.0–4.5, [M/H] = 0, binned to 0.02 nm and mirrored at NERSC
`/global/cfs/projectdirs/newera`), interpolated bilinearly in T_eff and
log g (`hbtsim.sed.NewEraGrid`), averaged over each spectrograph channel
(`sed.prepare_system`), with the spherical models' outer boundary drawn
at R_outer = (1.004–1.008) × R_τ=1 (`Star.radius_ref`,
`BinarySystem.drawn_radius_mas`; the τ = 1 tangent ray and the
half-intensity drop agree to 10⁻⁴ in μ). β Aur (both stars bracketed)
and δ Vel (both) use interpolated tables; Algol A (12 550 K) is clamped
to the 12 000 K edge (flagged); Algol B (4900 K, log g 3.2) and Spica
have no models and stay on blackbody + Claret. `scripts/sed_compare.py`
(output in `output/logs/sed_compare.txt`):

| System | F(NewEra)/πB at 400 / 500 / 800 nm | f₁/f₂ 400 nm BB → NewEra | fringe 2f₁f₂/(f₁+f₂)² 400 nm BB → NewEra | Hβ core | u(400 nm) NewEra vs C11 | g, i: BB / NewEra / anchor |
|---|---|---|---|---|---|---|
| β Aur | 1.47 / 1.17 / 0.86 | 1.234 → 1.236 | 0.4945 → 0.4944 | fringe 0.495 (twins) | 0.68 vs 0.52 | 2.07, 2.16 / **1.81, 2.29** / 1.80, 2.10 |
| δ Vel | 1.47 / 1.16 / 0.85 | 1.193 → 1.222 | 0.4961 → 0.4950 | fringe 0.496 | 0.68 vs 0.52 | 1.89, 2.03 / **1.66, 2.17** / 1.90, 2.25 |
| Algol (A only) | 1.05 / 0.89 / 0.73 | 57.9 → 60.5 | 0.0334 → 0.0320 | **0.068 → 0.230** | 0.62 vs 0.42 | 2.07, 2.36 / 2.14, 2.64 / 2.07, 2.58 |

What the atmospheres change: (i) the *absolute* flux — the Balmer jump
and line blanketing make A stars 45 % brighter than πB at 400 nm and
15 % fainter at 800 nm, so β Aur's g magnitude lands on its anchor to
0.01 mag without any offset (the blackbody needed 0.26 mag), while
**δ Vel comes out 0.24 mag brighter than its A-only V ≈ 2.0** with the
Mérand et al. 2011 radii, T_eff and distance — a tension in the
published parameters, not in the code (a 12 % smaller radius or 700 K
cooler stars would close it); (ii) the fringe contrast of the twin
systems changes by < 0.3 % (equal stars, equal spectra), Algol's by
−4 % in the blue continuum, +10–25 % in the red and **×3.4 in the Hβ
core**, where the B8 primary's line removes most of its light and the K
subgiant's share rises; (iii) the blue limb darkening is 30 % stronger
than the Claret tables (u = 0.68 vs 0.52 at 400 nm; 0.20 in the Hβ
core), and the disks are 0.4–0.8 % larger than the catalogue radii
(R_outer). The SNR rows therefore move by the flux change (SNR ∝ rate
∝ 10^{−0.4Δm}, up to 25 % for δ Vel) rather than by the fringe; the Hβ
and Hγ channels of Algol are the one place the fringe itself changes.

## 5e. A 4 m transportable pair: EON-SII

EON-SII (arXiv:2608.17444) is two road-transportable 4 m telescopes
(≈ 9 m² each, 80 % reflectivity) with a fibre-free 400–550 nm
spectrograph (R ≈ 7000–8000, ~1000 effective channels, > 60 %
throughput), picosecond time tags (CERN picoTDC) to a central
correlator at up to ~1 GHz per telescope, and reconfigurable
1.5–3 km baselines for compact stars. `hbtsim.snr` carries it as
`EON_SII_TELESCOPE`, `EONSII_MCP_PMT` (Photonis MCP-PMT: measured HBT
pair width σ = 27.4 ps; bialkali QE assumed), `EONSII_SPAD` (QUASAR
32×32 SPAD array: 20 ps and the SPAD Lambda PDE assumed) and
`EONSII_SPECTROGRAPH` (1000 × 0.15 nm; `_R7500` for the optical
resolution); `bispectrum.eonsii_pair` places the pair on Teide;
`scripts/feasibility_g3.py --g2 --instrument eonsii` gives the
two-telescope table (§5f) with the baseline free (tens to hundreds of
metres for milliarcsecond binaries). The assumptions flagged in the
code (atmosphere 0.80, QE curves, SPAD jitter) are to be replaced by the
instrument team's numbers.

**Cross-check against the paper** (`scripts/eonsii_crosscheck.py`,
its white dwarfs as u = 0.3 disks at |V|² ≈ 0.6): the photon rates agree
— 5.5 kHz per channel and 5.3 MHz per telescope on Sirius B against the
paper's ~5–10 kHz and ~7 MHz — but the hours to a given precision come
out 5–25× longer than the paper's (Sirius B to 10 % diameter: MCP-PMT
36 h unpolarized / 18 h with a beamsplitter vs 1.5 h; SPAD 3.3 / 1.7 h
vs 0.33 h). §5e.1 traces the gap.

### 5e.1 The sensitivity gap, resolved

`hbtsim/estimators.py` implements the paper's photon-level S/N (its
Eq. 5, ηR₁R₂τ_c|V|²T / √(ηR₁R₂τ_c|V|²T + 2R₁R₂Δt_res T)) and classic
HBT form (Eq. 3) beside our matched filter. Analytically, Eq. 5 **is**
the matched filter when η = p₂ and Δt_res = √π σ_pair, and an honest
coincidence box (one that counts only the erf fraction of the peak it
captures) is best at a half-width of 1.40 σ_pair, where it reaches 0.943
of the matched filter. `scripts/eonsii_crosscheck.py` confirms the
mapping numerically on the paper's targets (SNR/h 1.41 vs 1.42 on
Sirius B, MCP-PMT; the small residuals at low rates are the dark counts,
which Eq. 5 omits).

`hbtsim/montecarlo.py` reruns the paper's Sirius B Monte Carlo on our
photon budget: 10 h at zenith 45 / 52.5 / 60° from Teide (Sirius B
culminates at 45.0°), E–W 1750 m, 1000 channels, uniform-disk truth
29.5 µas. It draws Poisson coincidence-lag histograms (3.125 ps bins,
Gaussian pair kernel of the detector's jitter), estimates |V|² per
channel and block, and fits a uniform disk
(`scripts/eonsii_montecarlo.py`, `output/logs/eonsii_montecarlo.txt`,
100 realizations):

| Estimator | MCP-PMT: bias, scatter | SPAD: bias, scatter |
|---|---|---|
| matched filter (analytic σ: 18.6 % / 6.2 %) | +1.7 %, 19.7 % | −0.6 %, 6.6 % |
| honest box, ±1.40 σ_pair | +0.1 %, 20.5 % | −1.5 %, 7.0 % |
| raw box ±σ_pair (no capture correction) | **+30 %**, 16.9 % | **+34 %**, 5.5 % |
| raw box ±3.125 ps (Eq. 5 with Δt_res = TDC bin) | **+130 %**, 52 % | **+78 %**, 5.3 % |
| matched filter, sideband accidentals | +2.2 %, 22.6 % | −0.4 %, 7.0 % |

The matched filter is unbiased with its analytic scatter (pull 0.96 /
1.03). A narrow box without the capture correction looks more precise,
but that is only because it loses most of the peak: θ comes out 30–130 %
too large. Two implementation lessons were learned on the way. The
accidental level should come from the singles rates (R₁R₂ΔT, known to
~10⁻⁴), as a correlator normalizes it; estimating it from the
histogram's sidebands adds 7–15 % scatter. And the weights must come
from the model: weighting each channel by its own realization's
background biased θ by −3 % at SPAD S/N and destabilized the MCP-PMT
fit. The paper's MCP-PMT Monte Carlo gives 0.02955 ± 0.0014 mas
(4.7 %); ours gives 19.7 %, 4.2× wider.

**Gap ladder** (hours to 10 % on θ; paper 1.5 h MCP-PMT / 0.33 h SPAD;
each step analytic unless noted):

| Step | MCP-PMT | SPAD |
|---|---|---|
| our budget, matched filter, unpolarized | 34.5 h (×23) | 3.9 h (×12) |
| + polarizing beamsplitter (p₂ → 1 per stream) | 17.2 h (×11.5) | 2.0 h (×6.2) |
| + diameter-optimal baseline (2300 / 2500 m, not 1750 m) | 13.0 h (×8.7) | 1.40 h (×4.2) |
| + QE × 1.3 | 7.7 h (×5.1) | 0.81 h (×2.5) |
| (instead of QE) + Izaña extinction beyond the zenith budget | 16.4 h | 1.71 h |
| (instead of QE) + 6 % adjacent-channel correlation (Gaussian MC) | 16.7 h | 1.68 h |
| *not honest:* Eq. 5 with η = 1 on unpolarized light | 8.6 h | 0.97 h |
| *not honest:* … and Δt_res = σ_pair, no capture correction | 4.9 h | 0.55 h |
| *not honest:* … and Δt_res = 3.125 ps, no capture correction | 0.55 h | 0.14 h |

Every honest step together (polarizing beamsplitter, optimal
baseline, 30 % more QE) still leaves us ×5 (MCP-PMT) and ×2.5 (SPAD)
slower than the paper. Channel correlation at the measured 5–7 % costs
nothing measurable. The paper's hours are reproduced by Eq. 5 only with
Δt_res ≈ 5–6 ps at the beamsplitter/optimal-baseline budget (and
2–4 ps across all three white dwarfs in the cross-check). That is the
TDC-bin scale, below the 5 ps coherence time, and 5–10× below the
detectors' pair jitter (σ = 27.5 ps MCP-PMT, 12.2 ps SPAD). Our Monte
Carlo shows that applying such a window to histograms that carry the
jitter biases θ by +80–130 %. Since the paper's Monte Carlo recovers
the true diameter, our inference is that the simulated correlation
peak there is not broadened by the detector jitter (effective
Δt_res ≈ TDC bin). This inference cannot be checked from the paper,
because its estimator constants (b_el, F, η_vis, Δt_res) are not
published.

**Decision**: the matched filter, with the measured pair jitter and
p₂ = ½ for unpolarized light, stays the default. hbtsim's EON-SII
forecasts are therefore ×2.5–5 (honest optimum) to ×10–25 (as
configured) more conservative than the design study. If the instrument
team confirms a jitter-limited Δt_res ≲ 5 ps, `snr.Detector` jitter is
the single knob to change.

## 5f. Two-telescope g² on the four systems

`scripts/feasibility_g3.py --g2 --instrument {c2pu,keck,eonsii}` scans
the baseline along the separation axis at quadrature and reports, at
the baseline maximizing the first backend's total SNR, the SNR per hour
of every backend (the tables are regenerated in §5f-tables below from
`output/logs/g2_*.txt`).

SNR₂ per √hour at quadrature (blackbody → **NewEra**; Spica has no models;
Algol's NewEra column tables only its primary), from `output/logs/g2_*.txt`
(`scripts/g2_logs_to_md.py` for the full 98-row table):

| Instrument | Backend | β Aur (35/30 m) | Algol (10 m) | δ Vel (10–15 m) | Spica (10 m) |
|---|---|---|---|---|---|
| C2PU 2 × 1 m | 320 ch, time-tag | 7.2 → **7.0** (link-limited) | 9.6 → **9.5** (link-limited) | 6.4 → **6.2** (link-limited) | 9.2 (link-limited) |
| C2PU 2 × 1 m | 320 ch, correlator | 15 → **14** | 15 → **15** | 12 → **14** | 52 |
| C2PU 2 × 1 m | R = 5000, correlator | 60 → **54** | 60 → **58** | 47 → **54** | 208 |
| C2PU 2 × 1 m | R = 5000, correlator + PBS | 85 → **76** | 85 → **82** | 67 → **76** | 294 |
| Keck 2 × 10 m | 320 ch, correlator | 608 → **569** | 780 → **761** | 484 → **511** | 1431 |
| Keck 2 × 10 m | R = 5000, correlator | 5082 → **4583** | 5616 → **5433** | 3660 → **4162** | 16576 |
| EON-SII 2 × 4 m | 1000 ch, MCP-PMT | 344 → **343** (link-limited) | 434 → **434** (link-limited) | 234 → **238** (link-limited) | 391 (link-limited) |
| EON-SII 2 × 4 m | 1000 ch, QUASAR SPAD | 575 → **568** (link-limited) | 694 → **658** (link-limited) | 350 → **351** (link-limited) | 638 (link-limited) |
| EON-SII 2 × 4 m | 1000 ch, QUASAR SPAD + PBS | 817 → **808** (link-limited) | 986 → **935** (link-limited) | 497 → **499** (link-limited) | 907 (link-limited) |
| EON-SII 2 × 4 m | R = 7500 (2388 ch), QUASAR SPAD | 848 → **841** (link-limited) | 1031 → **994** (link-limited) | 524 → **528** (link-limited) | 945 (link-limited) |

Reading the table: (i) every time-tag row is pinned at ≈ 5–10 by the 1.4 × 10⁸
cps link whatever the telescope, and the four bright binaries saturate
even EON-SII's 10⁹ cps links (rates scaled ×0.1–0.5), so real-time
correlation is the enabling item for g² as much as for g³; (ii) the
NewEra tables move the g² sensitivities by −10 % (β Aur: fainter in the
red where most channels are, stronger limb darkening) to +14 % (δ Vel:
the model is brighter than the blackbody anchored to its V) — a
photon-budget effect, since the twin systems' fringe contrast is
unchanged; (iii) δ Vel's best two-telescope baseline is 10–15 m (its
4.7 m blue fringe is already smeared by 4 m pupils, D/P = 0.85, but
survives 1 m ones), which is where a transportable pair earns its
keep; (iv) EON-SII at 30 m on β Aur reaches SNR₂ ≈ 570/√h with the
SPAD array (link-limited), i.e. |V|² per 0.15 nm channel to ~5 % per
hour on a V = 1.9 star.

## 5g. δ Vel with three EON-SII units at Paranal

`scripts/deltavel_eonsii.py` (`output/logs/deltavel_eonsii.txt`,
figures `output/g3_deltavel_eonsii_per_night.png`,
`output/g3_cosphi_deltavel_eonsii.png`) places the EON-SII pair plus a
third identical 4 m unit on an equilateral triangle
(`bispectrum.eonsii_triangle`) at Paranal / CTAO-South. Teide never sees
δ Vel (dec −54.7°); from Paranal it is above 30° for H = ±4.82 h.
NewEra tables are used for both components.

**Geometry.**
- **Orbit.** ρ = 0.21–17.5 mas, with conjunctions at phase 0.406
  (ρ = 0.36 mas) and 0.971 (0.21 mas).
- **Eclipse guard.** The disks overlap or nearly do over 0.3935–0.4192
  and 0.9633–0.9786, 4.1 % of the orbit. Blocks there are rendered on
  per-epoch grids (`GridConfig.fit_epoch`, 256–512 px) and the rest use
  the analytic model (`vis_method="auto"`).
- **Pupil smearing.** At 400 nm, 4 m pupils keep > 50 % of the fringe
  contrast for 46 % of the orbit (11 % at maximum separation); 8.2 m
  pupils do so for 21 % (1.6 % at maximum separation).
- **Triangle side.** A scan over 8–120 m (one-hour transit snapshots,
  18 phases, SPAD + PBS) peaks at **12 m** for the phase statistic; 8–20 m
  are within 7 % of each other. Longer sides resolve the 1 mas disks:
  the phase statistic falls ×0.3 at 45 m and ×0.04 at 90 m (the
  template ×0.6 and ×0.06).
- **Blocks.** They are cut automatically so the fringe drifts ≤ 1/8
  cycle: 13 min at maximum separation, 60 min near conjunction.

**Campaign.** 32 nights at uniform phases plus 5 nights spanning the
two eclipse windows (these fall in different orbital cycles), with four
backends on shared geometry (`campaign_g3_snr`, resumable per-night
cache). Totals add in quadrature over nights; "nights" repeats this
phase sampling until the target is reached:

| Backend (1 GHz time-tag links) | template SNR | nights to Δcos φc ≤ 0.1 | phase SNR | nights to 0.1 rad |
|---|---|---|---|---|
| 1000 ch, MCP-PMT | 2.3 | 708 | 0.076 | 6×10⁵ |
| 1000 ch, QUASAR SPAD | 6.3 | 93 | 0.22 | 8×10⁴ |
| 1000 ch, QUASAR SPAD + PBS | 12.7 | **23** | 0.44 | 2×10⁴ |
| R = 7500 (2388 ch), QUASAR SPAD | 13.5 | **20** | 0.46 | 2×10⁴ |

The five eclipse-window nights carry as much template signal as the 32
uniform ones (8.5 against 9.5 for SPAD + PBS). Near conjunction the
fringe period is long compared with the pupils, and the triple
amplitude is at its largest.

**Why the phase statistic is hopeless.** g³ measures |T| cos φc. The
template ("amplitude") statistic detects that pattern, which fixes the
sign of the closure phase (0 or π per channel) — information g² cannot
give. The image asymmetry lives in the phase's departure from 0 or π,
whose Fisher information ∝ (snr sin φc)² ("phase" statistic,
`hbtsim.snr3`):
- **Near conjunction** a 12 m triangle barely resolves the 1 mas disks,
  so φc ≈ 0 and sin φc ≈ 0.
- **At wide separation** the 4 m pupils average the binary fringe down
  to 11 %, and the smeared bispectrum is again nearly real.
- **In between** (ρ ≈ 5–7 mas, phases 0.31–0.34 and 0.47–0.50) it
  peaks at ~0.14 per night, which is 10³× short.

The phase statistic is thus a property of the source and the pupils,
not of the photon budget. The VLT fares no better: its 8.2 m pupils
remove even more of the wide-separation fringe.

**Verdict.** Three EON-SII units with the SPAD + PBS or R = 7500 SPAD
backends detect δ Vel's closure-phase template in about three weeks of
nights spread over an orbit. That is comparable to the VLT's 20–27
nights at maximum separation, from an array that can be built for the
purpose. What they measure is the symmetric (sign) part of the
bispectrum, not the asymmetry. The NewEra tables change the snapshot at
maximum separation by −3 % in the template and +5 % in the phase
statistic.
The conjunction phases depend on the adopted ω convention; the campaign
samples both conjunctions, so its totals do not.

## 6. Caveats

- **Kernel calibration** (the critical systematic): ridge ratios of ~100
  at R = 5000 require the pair-correlation kernel shape to 10⁻³.
- **Readout**: the "correlator" readout assumed for the next-generation
  device is an idealization with no link limit, i.e. a real-time
  correlator. A correlator needs both telescopes' photons: either both
  beams are fibred to one sensor whose FPGA counts coincidences (short
  baselines), or each detector streams time-tags over fast links to a
  central FPGA/GPU correlator. This exists for analog photomultiplier
  streams (MAGIC, real-time GPU correlator) but not yet for
  multi-channel photon counting at ≥ 10⁹ cps; it is a development item.
  The 1.4 × 10⁸ cps time-tag ceiling of the current SPAD Lambda is the
  manufacturer's quoted maximum throughput (140 Mcps over two USB3
  links, 6 Gbps).
- **SEDs**: the NewEra tables now set the flux ratio and the limb
  profiles of β Aur, δ Vel and Algol A (§5d); Algol B (K0 IV) and
  Spica (25 300 / 20 900 K) still use blackbodies with observed anchors
  until the K-subgiant box and an NLTE hot-star grid arrive from
  Hamburg. Algol A is a 550 K extrapolation (clamped to 12 000 K).
- **δ Vel photometry**: with the Mérand et al. 2011 radii, T_eff and
  25.1 pc the model is 0.24 mag brighter than the observed A-only
  V ≈ 2.0; the SNR rows with tables inherit that brightness. Rotation
  (v sin i ≈ 145 km/s; oblate, gravity-darkened) is still not modeled
  and is the obvious suspect for both the photometry and the limb
  profiles.
- **EON-SII**: atmosphere, QE curves and SPAD jitter are assumptions.
  The design paper's hours are 10–25× shorter than ours, and 2.5–5×
  shorter even after a beamsplitter, the optimal baseline and 30 % more
  QE. They are reproduced only with Δt_res ≈ 2–6 ps, well below the
  detectors' 12–28 ps pair jitter (§5e.1, Monte Carlo). We keep the
  jitter-limited matched filter.
- **Algol C**: the ~10% incoherent third light dilutes every γ by ~0.9
  and the triple product by ~0.73 unless C is excluded optically.
- Orbital physics not modeled: Spica's e = 0.108 and apsidal motion, its
  β Cep pulsations and tidal distortion; δ Vel's rotational oblateness
  (the signal a campaign would target). The code's ω is the primary's
  spectroscopic argument of periastron (dz > 0 = secondary in front,
  ascending node along −p), so entering the published 109.7° is the
  consistent choice; the orbital radial velocities (`orbit.sky_positions`)
  reproduce β Aur's K₁/K₂ and put the secondary receding at the
  ascending node.

## 7. Conclusions

1. **The SPAD Lambda as delivered cannot measure closure phases of these
   binaries** on any large telescope: its time-tag link caps the photon
   rate 200–800× below what the stars deliver, and even with a
   correlator readout its 320 channels need ~150 nights on the best
   case (Spica, VLT).  The same link caps every two-telescope g² row
   at SNR ≈ 5–7/√h regardless of aperture (§5f).
2. **Spectral resolution plus real-time correlation is the enabling
   hardware**: an R = 5000 backend brings the Spica template detection on
   the VLT to 36 minutes (7 with a polarizing beamsplitter), a factor of
   ~2000 in time over 320 channels.
3. **Detection and imaging are different measurements.** The template
   amplitude comes in minutes; R = 100 closure-phase curves take nights
   per epoch (8.6 on the VLT for Spica, 14 h with a beamsplitter);
   single 0.1 nm channels are out of reach. Image
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
6. **Model atmospheres change the photon budget, not the verdicts**
   (§5d): with the NewEra tables the sensitivities move by −20 % to
   +14 %; the Hβ/Hγ channels of Algol are the one place the fringe
   itself changes (×3.4). δ Vel, at its correct 25.1 pc, is a g² target
   for the 1–4 m class (C2PU, EON-SII: §5f). Three EON-SII units detect
   its closure-phase template in ~20 nights over an orbit, but the
   asymmetric part of its closure phases is out of reach on every array
   considered here (§5g).

## References

- Malvimat, V., Wucknitz, O. & Saha, P. 2013, MNRAS (arXiv:1304.3391)
- Nuñez, P. D. & Domiciano de Souza, A. 2015, MNRAS (arXiv:1507.07635)
- Zmija, A. et al. 2025, MNRAS (arXiv:2512.13485) — first astrophysical
  g³ measurement (H.E.S.S.), sensitivity analysis, CTA projection
- Rai, Basak & Saha 2021 (arXiv:2105.09532); Guerin et al. 2025
  (arXiv:2503.22446) — pair formalism this work extends
