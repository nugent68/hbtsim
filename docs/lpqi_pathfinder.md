# The LPQI-Pathfinder in hbtsim

The La Palma Quantum Interferometer (LPQI; [lapalmaqi.es](https://lapalmaqi.es),
IAA-CSIC / IAC, PI F. Prada) is a photon-counting intensity interferometer at
the Roque de los Muchachos Observatory. Its first phase, the **LPQI-Pathfinder**,
correlates photon detections between the Nordic Optical Telescope (2.56 m) and
the Telescopio Nazionale Galileo (3.58 m) "550 metres apart", with 64×64 SPAD
cameras at the telescope foci, fibre-linked White Rabbit time stamps and
narrow-band filters; the planned extension adds GTC (10.4 m), WHT (4.2 m) and
INT (2.54 m) for baselines to 1.5 km ("50 µas"). First science is announced for
spring 2028 (angular sizes of the brightest stars, "160 µas").

This document records what the catalog carries for it, which numbers are
published and which are assumptions to confirm with the team, and what the
Pathfinder does for hbtsim's targets. Catalog: `sites/orm_not`,
`telescopes/{not_2p56m,tng_3p58m,gtc_10p4m,wht_4p2m,int_2p54m}`,
`detectors/{lpqi_spad64_i2cass,lpqi_spad64_i2cass_nextgen,lpqi_spad1_mpd,lpqi_spad1_mpd_blue}`,
`spectrographs/filter_lpqi_*_1nm`, `backends/lpqi_*`,
`arrays/{lpqi_pathfinder,lpqi_orm}`, campaigns `g2_binaries_lpqi_pathfinder`
and `mc_sirius_b_lpqi_pathfinder` (`hbtsim catalog show array lpqi_pathfinder`
prints the provenance).

## Sources

- Quintana, González-de-Rivera, López-Buedo & Prada 2026, Sensors 26, 5757
  ([doi:10.3390/s26185757](https://doi.org/10.3390/s26185757)): the Pathfinder
  SPAD array (IMSE/I2CASS 64×64, LiDAR heritage) and its readout — the only
  publication with instrument numbers: 24.5 µm pitch, fill factor 3.5 %, peak
  PDP 75 % (effective PDE ≈ 2.6 %), dead time 5–10 ns, median dark-count rate
  1.68 Hz per pixel at 27.8 °C, on-chip AER ceiling 40 Mevents/s, the
  implemented capture ≈ 6.7 Mevents/s, **no per-event time stamps yet** (the
  roadmap targets ≈ 500 ps end to end with White Rabbit; 10–20 ps synchronization
  jitter, 81–96 ps demonstrated on a previous Zynq); plate scales 0.18″ (NOT)
  and 0.13″ (TNG) per pixel at the natural focus; ORM dark sky 21.70 mag/arcsec²
  (V) → ≈ 3 sky photons/s/pixel.
- Cosentino et al. 2026, SPIE AS 14149AN: Peltier-cooled camera head at −20 °C,
  FPGA acquisition (abstract).
- Prada et al. 2026, "The Extremely Large Telescope Interferometer"
  (arXiv:2603.05589): SPAD + narrow filter at the focus, instrumental efficiency
  α = 0.5, dt = 10 ps for a future detector; cites LPQI and the megapixel-SPAD
  procurement for the GTC.
- lapalmaqi.es, IAC and IAA press releases (2026): telescopes, 550 m, 1.5 km,
  "50 µas", "160 µas", spring 2028.
- Telescope positions: NOT technical-details page (GPS: 28°45′26.2″ N,
  17°53′06.3″ W, 2382 m; focal length 28.16 m, f/11); Caporali & Barbieri,
  "The astronomic and geodetic coordinates of the Telescopio Nazionale Galileo,
  Canary Island" (WGS84 geodetic 28°45′14.4″ N, 342°06′39.4″ E from a GPS
  survey on 1996-11-22, ~5 m; the astronomic coordinates are the local-vertical
  values and are not used); ING `whtcoord.html` (28°45′38.3″ N, 17°52′53.9″ W,
  2332 m, GPS 1993) and `intcoord.html` (28°45′43.4″ N, 17°52′39.5″ W, 2336 m);
  gtc.iac.es (28°45′24″ N, 17°53′31″ W, "about 2300 m"; 2267 m from
  Wikipedia). GTC FITS headers carry LATITUDE +28:45:43.2, LONGITUD
  +17:52:39.5, which is the INT's position to within 6 m, not the GTC's, and
  were not used. Apertures and collecting areas from Wikipedia.
- F. Prada (LPQI PI), email of 2026-10-07: the two Pathfinder detector cases
  (the next IMSE-LPQI 64×64 array with microlenses, and the MPD single-pixel
  back-up) and the filter set with a fifth filter at 425 nm.

## Parameters and status

<!-- catalog:lpqi_parameters -->
|  | value | source / status |
|---|---|---|
| site | Roque de los Muchachos, Nordic Optical Telescope (LPQI reference point): 28.7573°, -17.8851°, 2382 m | Wikipedia (NOT infobox) |
| Pathfinder pair | NOT 2.56 m + TNG 3.58 m, B = 532 m at PA 227° | surveyed positions (NOT GPS; TNG geodetic survey, ~5 m); the project quotes 550 m |
| five-telescope network | NOT 2.56 m, TNG 3.58 m, GTC 10.4 m, WHT 4.2 m, INT 2.54 m; baselines 409–1518 m | NOT, WHT, INT GPS pages; TNG survey; GTC web page (arc-second) |
| telescope throughput | 0.3, 0.3 | hbtsim default (**assumed**) |
| detector (Pathfinder: lensed IMSE-LPQI 64×64) | PDE 0.50 flat 400–550 nm, 0.20 at 650 nm (microlenses); 500 ps FWHM, 4 pixels, dead 10 ns, dark 1.68 cps/pixel, readout ≤ 6.7e+06 cps | F. Prada 2026-10-07; PDE shape and timing **assumed** |
| detector (Pathfinder: MPD single pixel, back-up) | PDE 0.30 flat 400–700 nm, 35 ps FWHM (275 ps below 470 nm), dead 55 ns, dark 50 cps, 1 pixel, link ceiling unknown | F. Prada 2026-10-07; flat PDE **assumed**, ceiling **unknown** |
| bare IMSE array (reference only) | PDE 0.026 = fill factor 3.5 % × PDP 75 %, otherwise as the lensed array with 25 pixels | Quintana et al. 2026, Sensors 26, 5757; not a Pathfinder option |
| filters | 425.0 nm (1.0 nm); 500.0 nm (1.0 nm); 550.0 nm (1.0 nm); 656.3 nm (1.0 nm); 486.1 nm (1.0 nm); throughput 0.9; one per night | wavelengths **to be confirmed**; width and one-per-night from the user |
<!-- /catalog -->

The filter set (Hα, Hβ, 500, 550 and 425 nm, 1 nm wide, one per night) was
confirmed by F. Prada on 2026-10-07; the exact central wavelengths and widths
remain working values. Two detectors are modelled, the two the team is
building: the **lensed IMSE-LPQI 64×64 array** (microlenses bring the fill
factor from 3.5 % to ~100 %: PDE 0.50 over 400–550 nm, 0.20 at 650 nm;
500 ps White Rabbit timing target; 4 pixels per star; the 6.7×10⁶ cps
capture ceiling of the current readout) and the **MPD single-pixel 50 µm
SPAD back-up** (PDE 0.30, 35 ps FWHM above 470 nm but 250–300 ps at 400 nm,
so the 425 nm filter uses a 275 ps variant; dead time 55 ns, 50 cps dark,
time-tag ceiling unknown and therefore not applied). The bare 3.5 %-fill-factor
array characterized in the Sensors paper stays in the catalog
(`lpqi_spad64_i2cass`, backends `lpqi_*`) as the published reference only.
The arrays' ENU positions come from the surveyed coordinates above (the NOT
and TNG ends of the Pathfinder to GPS / 5 m precision; GTC at arc-second
precision): NOT–TNG = 531.8 m at position angle 226.7°, 18 m short of the
project's rounded "550 m" (`load_array("lpqi_pathfinder", baseline_m=550.0)`
uses the published figure), and the five-telescope network spans 409 m
(TNG–GTC) to 1518 m (GTC–INT), the project's "1.5 km". The earlier Wikipedia
longitude of the TNG was 97 m too far east (NOT–TNG 471 m).

## What the Pathfinder does for hbtsim's targets

**Resolution.** At B = 532 m and 500 nm the first null of a uniform disk sits
at θ = 1.22 λ/B = 0.24 mas, so every star in the catalog is resolved out:
the binary components (0.45–1.1 mas), Sirius A (6.0 mas) and Vega (3.3 mas)
all sit many nulls beyond. The Pathfinder is built for 50–200 µas sources
(white dwarfs, hot compact stars); for our binaries the one-night |V|² is
10⁻⁴–10⁻³ (table below), not a measurement.

**Sensitivity.** A 1 nm filter gives a coherence time τ_c = λ²/(cΔλ) ≈ 0.8 ps
at 500 nm, against a pair timing width of 300 ps (both IMSE arrays, 500 ps
FWHM per detector) or 21 ps (MPD, 35 ps FWHM); the g² S/N scales with the
photon rate (so with the PDE) and with the coherence time over the square root
of the timing width, and a single channel has no multiplexing gain. Two
things follow in the tables below. The lensed array's 50 % PDE does not reach
the photons: on these bright stars its rates (10⁷–10⁸ cps per telescope
through a 1 nm filter) run into the 6.7×10⁶ cps capture ceiling and are
attenuated by ×0.1–0.3, so it ends up only a factor 2–8 per night ahead of the
bare 2.6 % array it replaces; for the Pathfinder's bright targets the readout,
not the sensor, is the limit. The MPD single pixel, with a lower PDE but 35 ps
timing and no ceiling in the model, is the more sensitive of the two by a
factor 10–20 — except through the 425 nm filter, where its 275 ps timing costs
a factor 3.6.

**One filter per night.** The backends `lpqi_<filter>` are exclusive: a night on
one filter is a night not spent on the others. The g2 runner's
`one_backend_per_night` option therefore **adds** the nights of the filter set
(it never combines them in quadrature), and prints the rule.

### The four binaries on NOT + TNG (`hbtsim run g2_binaries_lpqi_pathfinder`)

One night along the uv track (30-min blocks above 30°; NewEra atmospheres where
the grid covers the stars, the blackbody + Claret model otherwise), the two
detectors; "nights" is the time to a 3σ detection of |V|² at the night's
mean value. δ Vel (dec −54.7°) never rises above 30° from La Palma. The full
table (30 rows) is `output/campaigns/g2_binaries_lpqi_pathfinder/table.md`.

| target | SED | filter | detector | B [m] | |V|² over the night | SNR/night | nights (3σ) |
|---|---|---|---|---|---|---|---|
| β Aur | NewEra | 550 nm | lensed IMSE (readout-limited) | 289–532 | 6×10⁻⁵–2×10⁻³ | 6.2×10⁻³ | 2.4×10⁵ |
| β Aur | NewEra | 550 nm | MPD single pixel | 289–532 | 6×10⁻⁵–2×10⁻³ | 0.031 | 9.4×10³ |
| β Aur | NewEra | Hα | MPD single pixel | 289–532 | 3×10⁻⁵–3×10⁻³ | 0.045 | 4.4×10³ |
| β Aur | NewEra | 425 nm | MPD (275 ps) | 289–532 | 1×10⁻⁵–2×10⁻⁴ | 1.6×10⁻³ | 3.4×10⁶ |
| Algol | NewEra (A) | 550 nm | lensed IMSE (readout-limited) | 300–532 | 4×10⁻⁵–2×10⁻³ | 0.011 | 7.7×10⁴ |
| Algol | NewEra (A) | Hα | MPD single pixel | 300–532 | 1×10⁻⁵–5×10⁻³ | 0.069 | 1.9×10³ |
| Spica | blackbody | Hα | lensed IMSE (readout-limited) | 289–531 | 4×10⁻⁴–3×10⁻³ | 0.024 | 1.5×10⁴ |
| Spica | blackbody | Hα | MPD single pixel | 289–531 | 4×10⁻⁴–3×10⁻³ | 0.19 | 261 |
| Spica | blackbody | 550 nm | MPD single pixel | 289–531 | 5×10⁻⁶–3×10⁻³ | 0.11 | 808 |

Filter set (five filters, one per night, nights added): β Aur 2.3×10⁷ nights
(lensed IMSE) / 3.4×10⁵ + 3.4×10⁶ (MPD, the 425 nm night dominating); Algol
1.2×10⁶ / 2.3×10⁴ + 2×10⁵; Spica 9×10⁵ / 4.8×10³ + 3.9×10⁴. Every lensed-IMSE
row is readout-limited (×0.1–0.3 at the 6.7×10⁶ cps capture ceiling); the MPD
has no ceiling in the model because none has been specified. For reference, the
bare 2.6 % array of the Sensors paper would need 6×10⁸, 4×10⁷ and 2×10⁶ nights
on the three targets.

### Sirius B on NOT + TNG (`hbtsim run mc_sirius_b_lpqi_pathfinder`)

The Pathfinder's own use case: a 28.5 µas white dwarf (V = 8.44), 10 h at
zenith angles 46°, 52.5°, 60° (Sirius culminates at 45.5° from the NOT), one
500 nm / 1 nm filter, unpolarized, geometric-mean aperture √(A_NOT A_TNG), projected baselines 488–531 m.

| detector | pair σ_t | analytic σ(θ)/θ in 10 h | hours to 10 % | with PBS at the 2400 m optimum |
|---|---|---|---|---|
| lensed IMSE 64×64 | 300 ps | 99 | 9.7×10⁶ | 6.8×10⁴ |
| MPD single pixel | 21 ps | 42 | 1.8×10⁶ | 1.3×10⁴ |

At V = 8.44 the rates are far below any ceiling, so here the sensor counts:
the lensed array gets the full factor 19 in precision that its PDE gives over
the bare one (which would need 3.6×10⁹ h), and the MPD's timing makes it the
better of the two. Neither of them
measures Sirius B with one 1 nm channel: the Poisson Monte Carlo cannot fit a
diameter from these histograms (the coincidence excess is below the
accidentals' shot noise by orders of magnitude) and the table reports the
analytic matched-filter precision. The EON-SII pair reaches 10 % on the same
star in 1.5 h (MCP-PMT) because it has 1000 channels, 27 ps and a 20 % PDE:
the Pathfinder needs a spectrograph (hundreds of channels) on top of the MPD
class of timing to approach it.

## Northern targets: hot stars at the first null (`hbtsim run g2_singles_lpqi_pathfinder`)

The Pathfinder's niche in the northern sky is the overlap of two constraints:
it resolves out everything larger than ≈ 0.25 mas, and with one 1 nm channel
on 2.6 + 3.6 m apertures it reaches V ≲ 3.5. That overlap is the bright, hot,
compact end of the sky: O and early-B stars at 150–400 pc with θ = 0.2–0.4 mas,
none of which has an interferometric diameter (CHARA and the VLTI do not
resolve them in the visible). Over a night the projected NOT–TNG baseline sweeps
300–530 m, so for these stars the track crosses the first null of the disk
visibility: the null position gives the diameter nearly model-independently
and the sidelobe level the limb darkening. The nights below are the time to a
5 % diameter from the Fisher information of |V|²(θ) along the track (nights
add), uniform-disk limb darkening from the Spica Claret law, blackbody fluxes
anchored to V (NewEra covers nothing above 12 000 K), the catalog throughput
0.3 × 0.9, and the MPD without a link ceiling.

| star | type | V | θ (mas) | B over the night [m] | |V|² (Hα) | nights to 5 % θ: MPD Hα | MPD 550 nm | lensed array Hα | lensed array 550 nm | all five filters, MPD |
|---|---|---|---|---|---|---|---|---|---|---|
| lambda Ori A | O9 III | 3.7 | 0.238 | 392–532 | 7.1e-02–0.28 | **3.7** | 7.8 | 105 | 33 | 7.9 |
| beta Cep | B1 IV, β Cep pulsator | 3.23 | 0.320 | 334–507 | 1.2e-03–0.18 | **4.0** | 14 | 106 | 56 (rl) | 11 |
| eta Ori Aa | B1 V, eclipsing 7.99 d | 3.42 | 0.201 | 323–532 | 1.8e-01–0.56 | **1.9** | 2.3 | 50 | 9.3 | 2.5 |
| epsilon Per | B0.5 III, SB 14 d | 2.88 | 0.363 | 302–532 | 3.4e-06–0.16 | **6.2** | 23 | 152 | 156 (rl) | 16 |
| delta Ori Aa1 | O9.5 II, eclipsing 5.73 d | 2.23 | 0.319 | 334–531 | 2.8e-06–0.18 | **1.4** | 4.4 | 28 | 75 (rl) | 3.7 |
| gamma Peg | B2 IV, β Cep pulsator | 2.84 | 0.349 | 421–532 | 5.4e-05–0.02 | **26** | 120 | 629 | 873 (rl) | 64 |
| zeta Tau | B2 IIIpe shell star | 3.01 | 0.365 | 390–531 | 1.2e-05–0.02 | **43** | 186 | 1.1×10³ | 1.0×10³ (rl) | 97 |
| gamma Cas | B0.5 IVe, disk | 2.39 | 0.532 | 297–527 | 7.8e-05–0.01 | **35** | 86 | 748 | 1.2×10³ (rl) | 79 |

(rl) = readout-limited at the 6.7×10⁶ cps capture ceiling. The last column
rotates the five filters one per night on the MPD; the 425 nm filter, where the
MPD runs at 275 ps, costs the set roughly a factor two.

What this says:

- **Diameters of hot stars in a few nights.** η Ori Aa (θ ≈ 0.20 mas), δ Ori Aa1,
  λ Ori A and β Cep reach 5 % in 1–4 nights per filter with the MPD; ε Per in a
  week. These radii anchor the upper main sequence and test hot-star limb
  darkening where no measurement exists. The η Ori and δ Ori figures use the
  whole system's light and are optimistic by about a factor two in rate.
- **Pulsation and binarity in the same stars.** β Cep and γ Peg are the prototype
  β Cephei pulsators: a few-percent radius change moves the null by tens of
  metres, within reach over a season. ε Per, δ Ori Aa and η Ori Aa are close
  hot binaries whose components are themselves below the null; modelled as
  binaries they give fringes in |V|² versus projected baseline, the two-telescope
  version of the closure-phase science elsewhere in hbtsim.
- **Hα against the continuum.** For the Be stars ζ Tau and γ Cas the decretion
  disk (several mas) is gone at 300 m, so the Hα filter measures the fraction of
  Hα light from the unresolved photosphere against the 550 nm continuum, night by
  night. The same contrast on Alnilam (ε Ori, 0.7 mas, V 1.7) would probe its
  wind, but its second lobe sits at |V|² ≲ 0.004.
- **Not reachable:** white dwarfs and hot subdwarfs have the right size and the
  wrong magnitude (V > 8); every star brighter than V ≈ 1.5 in the north is a
  cool or A-type star already resolved out; all of this needs the MPD-class
  timing — the lensed array at 500 ps is 5–15× slower on the same stars.

Caveats: θ is 2R/d from literature radii and distances (δ Ori's distance is
disputed at the factor-two level; λ Ori A's parallax is 15 % uncertain); the
multiples' light is attributed to one star; the MPD link ceiling is unknown; a
field stop must exclude λ Ori B (4″) and η Ori's companions.

## Movies (`hbtsim run movie_delori_lpqi_night`, `movie_betcep_lpqi_pulsation`)

Two campaigns of the `nightmovie` runner animate the measurement, one frame per
5-min block, with the MPD behind the 550 nm filter (whose first null at 1.22 λ/θ
≈ 430 m lies inside the night's 330–530 m sweep):

- **δ Ori Aa1: one night, then twenty** (108 frames): act one is the night, one
  frame per 5-min block — the projected NOT–TNG baseline on the (u, v) plane
  with the first-null circle; the 30-min-binned |V|² measurements walking along
  the disk curve from beyond the null back up the first lobe, with the θ ± 5 %
  curves they discriminate; and the coincidence histogram's excess over the
  accidentals, matched-filtered, in units of its shot noise. The g² bump stays
  inside the noise after one night (+0.7σ expected). Act two adds the same
  track night after night, one frame per night: the bins average down as
  1/√n, the bump rises to +3σ after 20 nights (147 h on source), and the
  diameter's error band narrows from 10 % to 2.3 %.
- **δ Ori Aa1 with a different filter each night** (`movie_delori_lpqi_filters`,
  20 nights rotating Hα, 550, 500, Hβ, 425 nm on the MPD): the same 330–530 m
  ground sweep samples the disk curve at πθB/λ from the first lobe (Hα, null
  at 518 m) through the null (550, 500 nm) into the sidelobe (Hβ, 425 nm);
  the middle panel plots every night against the equivalent baseline Bλ₀/λ so
  the filters fall on one curve, the third panel tracks σ(θ)/θ. The answer to
  "does changing B/λ help" is no for raw precision: the bluer filters sit
  beyond the null where |V|² and its slope are tiny (and the 425 nm night
  costs the MPD its timing), so 20 rotated nights give 2.4 % against 1.3 %
  for 20 nights in Hα alone. What the rotation buys is the shape of the curve
  across the null and sidelobe, i.e. a test of the limb-darkening model that
  a single filter cannot make. A weighted schedule (mostly Hα, a few 550 nm
  nights for the null) keeps most of both.
- **β Cep, the pulsating diameter** (190 frames): act one is the same night with
  θ(t) breathing by 1 % on the 0.19048 d period (amplitude assumed from the
  radial-velocity curve); act two folds 60 nights on pulsation phase. The honest
  result is negative: after 60 nights the phase-binned diameter has σ(θ)/θ ≈ 6 %
  per bin against a 1 % amplitude, so the Pathfinder cannot resolve β Cep's
  pulsation with one 1 nm channel (it would take ~10⁴ nights); a spectrograph's
  multiplexing is what would make it possible.

The movies are written to `output/campaigns/<campaign>/<campaign>.mp4`
(PNG frames when ffmpeg is absent); `--no-figures` computes the numbers only.

## Assumptions to confirm with the LPQI team

1. The filter list is confirmed (Hα, Hβ, 500, 550, 425 nm; 1 nm; one per
   night); the exact central wavelengths, widths and the 0.9 throughput are
   still working values.
2. The NOT–TNG baseline: 532 m from the two surveyed positions against the
   project's "550 m"; the GTC position (arc-second web value only; its FITS
   headers carry the INT's coordinates) and whether the WHT / INT ground-floor
   points are the ones the fibre links terminate at.
3. The lensed IMSE-LPQI array: the flat 50 % PDE over 400–550 nm and the 20 %
   at 650 nm (shape assumed), the 500 ps White Rabbit timing target, 4 pixels
   per star, the cooled dark count rate, and above all whether the 6.7×10⁶ cps
   capture ceiling stays — it is what limits this array on every bright target
   in the tables.
4. The MPD single-pixel module: its time-tag link ceiling (unknown; none
   applied), the flat 30 % PDE, and the 250–300 ps blue timing (275 ps used
   for the 425 nm filter).
5. Telescope + relay throughput to the camera (0.3 × 0.9 assumed; Prada et al.
   use an overall 0.5). The NOT (f/11, 28.16 m) and TNG (Nasmyth, 38.5 m)
   focal lengths set the plate scales; which focus the cameras use is to be
   confirmed.
6. The field stop at the camera: λ Ori B (V 5.6 at 4.4″), η Ori's and β Cep's
   companions must be excluded for the hot-star diameters above.
