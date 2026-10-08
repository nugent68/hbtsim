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
| filters | 425.0 nm (1.0 nm); 500.0 nm (1.0 nm); 550.0 nm (1.0 nm); 650.0 nm (1.0 nm); 656.3 nm (1.0 nm); 486.1 nm (1.0 nm); throughput 0.9; one per night | wavelengths **to be confirmed**; width and one-per-night from the user |
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

## Movies (`hbtsim run movie_delori_lpqi_night`, `movie_betcep_lpqi_pulsation`, `movie_etaori_lpqi_orbit`, `movie_iotaori_lpqi_orbit`)

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
- **δ Ori Aa1, the weighted schedule** (`movie_delori_lpqi_weighted`): 12 nights
  in Hα for the diameter, then 4 at 550 nm for the null and 2 each at 500 nm
  and Hβ for the sidelobe, none at 425 nm. It ends at σ(θ)/θ = 1.6 % against
  1.3 % for Hα alone and 2.4 % for the uniform rotation, with an 11σ
  detection in Hα and the null and sidelobe sampled at 2σ and ~1σ: most of
  the precision and the shape test. `options.schedule` lists the backend for
  every night; without it the backends rotate.
- **δ Ori Aa1, is it oblate?** (`movie_delori_lpqi_oblateness`, Hα on the MPD,
  40 nights): the target now carries an `ellipse` (axis ratio 1.03 from
  v sin i = 114 km/s on R = 13.1 R☉, M = 17.8 M☉; major axis at an assumed
  PA of 0°, the spin axis's sky orientation being unknown). The movie colours
  each measurement by the baseline's position angle and compares the oblate
  disk along the track with the equal-area circular disk; the third panel is
  the significance of q − 1 with θ marginalized and the orientation held
  fixed. The answer is negative: the NOT–TNG track sweeps only 15°–53° in
  position angle (not the 90° a longer track would give), the oblate and
  circular curves differ by at most 0.01 in |V|² against a per-block σ of
  0.19, and θ and q are 87 % correlated, so σ(q) = 0.45 per night and a 3σ
  detection of a 3 % flattening would take ~2000 nights (0.4σ after 40). The
  Pathfinder measures a mean diameter; oblateness needs either a second
  baseline at a different position angle (the five-telescope network) or the
  spectrograph's multiplexing. The tidal distortion by the 5.73 d companion,
  of similar size and rotating with orbital phase, is not modelled.
- **β Cep, the pulsating diameter** (190 frames): act one is the same night with
  θ(t) breathing by 1 % on the 0.19048 d period (amplitude assumed from the
  radial-velocity curve); act two folds 60 nights on pulsation phase. The honest
  result is negative: after 60 nights the phase-binned diameter has σ(θ)/θ ≈ 6 %
  per bin against a 1 % amplitude, so the Pathfinder cannot resolve β Cep's
  pulsation with one 1 nm channel (it would take ~10⁴ nights); a spectrograph's
  multiplexing is what would make it possible.

- **η Ori Aa through one orbit** (`movie_etaori_lpqi_orbit`, Hα on the MPD,
  eight consecutive nights, 400 frames): the binary mode of the runner
  (`hbtsim/runners/nightmovie_binary.py`), on a new binary target
  `etaori_ab` (B1 V + B3 V, P = 7.98763 d, i = 87.62°, a = 0.2171 AU from the
  masses; at the Hipparcos 300 pc that is a = 0.724 mas with the stars
  0.201 and 0.149 mas across). Eight nights step the orbit by 0.125 in phase:
  six nights of fringes with λ/ρ = 190–260 m swept by the 317–532 m
  baseline, and two eclipse nights (phases 0.25 and 0.75) where the disks
  overlap, the fringe is gone and the rendered two-disk image is the model.
  The panels are the pair on the sky with the baseline direction, the night's
  |V|² fringes with the 30-min bins, and the 68 % region of that night's
  fitted separation on the apparent orbit, with the orbit drawn at a ± 1σ
  from the Fisher information accumulated so far on (ln a, Ω). η Ori Ac
  (V 5.65 at 44 mas) is inside any field stop: its 21 % of the light adds to
  the rates and dilutes the fringe by 0.62 (`options.third_light_fraction`);
  η Ori B (V 4.95 at 1.7″) is assumed excluded. The honest numbers: σ(|V|²)
  = 0.51 per 30-min bin against fringe amplitudes of 0.1–0.3, so a single
  night detects its fringe at only ~2σ and its separation region is a set
  of stripes (one baseline direction fixes the separation only along itself,
  a fringe period apart), but the orbit's scale accumulates: σ(a)/a = 28 %
  after night 1, 16 % after two, 8.5 % after the eight. Since a in AU is
  known from the spectroscopic masses, that is a geometric distance to
  ±8.5 % from one orbit, against Hipparcos's 32 % and no Gaia parallax at
  all; a second orbit would bring it to 6 %, the lensed IMSE array (σ = 4.8
  per block) would need ~30 orbits. The unknown node angle matters: Ω = 45°
  (node along the NOT–TNG baseline) gives 3.7 %, Ω = 0, 90, 135° give
  8.5–8.9 %; the movie uses the assumed Ω = 0.
  Those are local (Fisher) precisions. The right panel therefore shows
  the global fit instead (`options.global_fit`): the χ² of every
  out-of-eclipse block of the nights so far over the orbit's angular scale
  (0.5–1.5 × the assumed a, all angular sizes scaling with it) and the
  node-angle offset (0–180°), the two unknowns of a spectroscopic pair seen
  with one baseline, converted to a distance through a in AU. After each
  night the deepest solution is drawn with its 68 % range and every other
  solution surviving at 95 % (the fringe aliases) as a hollow marker, against
  the model distance and the Hipparcos band (`options.reference_distances`).
  The closing act puts the χ² map in the left panel and the family of orbits
  allowed at 68 % in the middle, and `results.json` carries `global_fit` and
  `cumulative_fit`. For the rendered realization the first night leaves the
  distance unconstrained (200–600 pc), the second gives 405 pc with eight
  alias solutions, and all eight nights give a = 0.673 mas (68 % in its
  island 0.579–0.724), d = 323 pc (300–375) with 3 solutions left. Over 20
  noise realizations the truth lies in the deepest island only a quarter of
  the time and the rms error of the best-fit scale is 15 %, not 8.5 %.
  Two follow-up questions, answered the same way:
  - *A second orbit months later.* Nothing rotates: the uv track of a
    fixed declination is the same every night (only the transit time moves,
    so a season later the star may be observable for half the window, which
    costs a factor two in information). A second eight-night orbit adds its
    photons, √2 in Fisher terms (6.0 %), and thins the aliases: rms 6.2 %,
    2.5 islands; four orbits give 4.7 % rms, two islands, 95 % range
    0.89–1.14 in scale. Sampling different orbital phases is not what helps
    (the shape is already known from spectroscopy); the quadrature nights
    carry the information, so repeating them is as good as filling in.
  - *Another filter.* Alone, every bluer filter is worse: the disks are
    more resolved, so the fringe contrast C falls faster than the photon
    rate rises (Hβ/500/550 nm 10.5–11.6 %, 425 nm with the 275 ps MPD
    jitter 77 %). Alternating two filters within one orbit is worse than Hα
    alone (10.6 % Fisher, 24 % rms, the alias area quadrupled). A second
    orbit in 550 nm instead of Hα is a wash (7.5 % rms against 6.2 %, with
    the truth in the deepest island slightly more often). The B/λ leverage
    that helps a single star's null does not help the binary's scale, which
    is set by SNR on a fringe whose period scales with λ for every
    separation. What would break the stripes is a second baseline at a
    different position angle (TNG–GTC, NOT–WHT), not a second wavelength.
    One caveat on Hα itself: a 1 nm filter on a B1 V star sits in the
    photospheric Hα absorption, which the blackbody model ignores; a red
    continuum filter beside the line would be strictly better.
- **η Ori Aa with the node rotated by 90°** (`movie_etaori_lpqi_orbit_node90`,
  target `etaori_ab_node90`, which extends `etaori_ab` with `node_pa_deg` 90):
  the same eight nights with the apparent orbit East–West instead of
  North–South, since the node angle is unknown. The fringe geometry changes
  visibly: at quadrature the NOT–TNG track now sweeps a full fringe (|V|²
  0.02–0.56 against 0.08–0.14 at PA 0) because the baseline's east component
  is what varies most through the night, and the stripes of the per-night
  region run across the orbit instead of along it. The precision does not:
  σ(a)/a = 8.8 % (Fisher) after the eight nights against 8.5 %, with the
  global check giving a 10 % rms scale error and 4 alias islands for one
  orbit, 4.9 % and 2 islands for two (the rendered realization's global
  fit: a = 0.630 mas, 68 % 0.536–0.731, node offset +38°, 3 islands). The
  scale information comes from how
  the fringe period changes along the orbit, which a single baseline samples
  about equally at either orientation; the 3.7 % at Ω = 45° is the special
  case of the node along the baseline.
- **Two orbits** (`movie_etaori_lpqi_2orbits`, `movie_etaori_lpqi_2orbits_node90`,
  sixteen consecutive nights, 20 fps): with P = 7.98763 d the nightly phase
  step is 0.1252, so night 9 falls at phase 1.0016 and the second orbit
  repeats the first's phases. A gap of whole nights cannot change that
  (0.0002 of phase per day of gap; half a step after 323 days), the offsets
  reachable within a season (≤ 0.03 after 150 days) sample phases nearer the
  eclipses that carry less fringe information, and the 20-realization check
  agrees: 16 consecutive nights give a 6.2 % rms scale error with the truth
  in the deepest island 60 % of the time, against 7.8 % / 45 % for a 150-day
  gap and 6.2 % / 45 % for an 11-month one. So consecutive nights are the
  right second orbit. The rendered realizations: at PA 0 the global fit is
  a = 0.717 mas (68 % 0.659–0.746), d = 303 pc, node offset 0°, two 95 %
  islands left; at PA 90, a = 0.717 mas (68 % 0.659–0.789), d = 303 pc,
  node offset 3°, three islands. The Fisher σ(d)/d after sixteen nights is
  6.1–6.3 %. The alias that caught the one-orbit PA 90 fit at a node offset
  of 38° is gone; the surviving islands are the ones a second baseline
  direction would remove.
- **Two orbits 323 days apart, node at PA 90** (`movie_etaori_lpqi_2orbits_gap_node90`,
  `options.gap` {after_night 8, days 323}): 323 / 7.98763 = 40.44 orbits, so
  the second eight nights sample the orbit half a step (0.0625 in phase)
  from the first, the largest offset a whole-night gap can give, and the
  season is six weeks earlier, still with the full window. This time the
  offset pays. The second run has no eclipse nights (eight fringe nights at
  ρ = 0.27 and 0.67 mas instead of six), and over 20 realizations the PA 90
  gapped plan gives a 4.5 % rms scale error with 2.2 islands and the truth
  in the deepest one 55 % of the time, against 6.4 %, 3.0 and 35 % for
  sixteen consecutive nights (a 150-day gap, offset 0.029, sits between at
  4.9 %). The PA 0 orientation did not gain from the offset because its
  near-eclipse phases, separated North–South, are swept poorly by the
  NOT–TNG track; with the pair East–West they are swept well. The rendered
  realization's global fit is a = 0.709 mas (68 % 0.637–0.789), d = 306 pc,
  node offset 3°, three islands.

- **ι Ori Aa, the better distance target** (`movie_iotaori_lpqi_orbit`, target
  `iotaori_aa`, eight nights every 3.64 d, 650 nm red continuum on the MPD):
  a search of the bright northern double-lined pairs for the Pathfinder's
  figure of merit (V ≲ 4, both stars unresolved at 530 m, comparable fluxes,
  separation 1–3 mas so the sweep crosses several fringes, a disputed
  distance) singled out Hatysa: O9 III + B0.8 III, P 29.13 d, e 0.745,
  a = 132.3 R☉ from the BRITE eclipsing-orbit solution (Pablo et al. 2017,
  which derives no distance). ι Ori A has no Gaia parallax. Its distance
  rests on the wide companions: Gaia DR3 gives B (V 7.0, 11″) 2.787 ± 0.048
  mas = 359 ± 6 pc and C (A0, 49″) 2.606 ± 0.024 mas = 384 pc, which
  disagree by 3.4σ (25 pc, far more than any physical depth of the triple);
  B sits in the glare of a V 2.8 star and bright-star DR3 errors run ~1.3×
  underestimated, so the realistic value is the B + C mean, 2.64 mas =
  378 pc, with the error inflated for the reduced χ² of ~11 to ±20–25 pc
  (5–7 %). Maíz Apellániz & Barbá 2020 quoted 412 +14/−13 pc from B's DR2
  parallax alone; Hipparcos's 1.40 ± 0.22 mas = 714 pc for A is probably
  corrupted by the multiplicity. No one has measured the distance of the
  pair itself. At 378 pc a = 1.628 mas with the stars 0.224 and 0.122 mas
  across and separations of 0.2–2.1 mas around the orbit. It is
  1.7 mag brighter than η Ori (σ(|V|²) = 0.23 per 10-min block against
  0.89), and the eccentric orbit sampled every P/8 spreads the fringe
  spacing from 70 to 900 m, which is what kills the aliases. The other
  candidates fail: δ Ori Aa has the most dramatic distance dispute (212 vs
  382 pc) but its 0.32 mas primary is resolved out and the fringe contrast
  is 0.016; σ Ori Aa,Ab would be excellent but CHARA already has its
  dynamical distance to 0.3 % (Schaefer et al. 2016); Spica, Mizar A and
  β¹ Sco have resolved disks and precise parallaxes; ζ Ori's 36 mas orbit
  has a 4 m fringe period; ε Per is single-lined.
  Filters: with the primary at 0.2 mas every bluer filter is worse (the disk
  is twice as resolved at Hβ, the 18 300 K secondary's share drops, and the
  MPD's extra photons are cancelled by the shorter coherence time): fringe-
  only σ(a)/a over eight nights 2.0 % at Hα, 2.4 % at 550, 3.5 % at Hβ,
  13 % at 425 nm with the 275 ps blue MPD; alternating red and blue nights is
  worse than red alone on every parameter (scale, node, primary diameter,
  secondary temperature), and it doubles the alias islands. The O9 III
  primary has a wind and the pair is a colliding-wind X-ray source, so the
  movie uses a proposed 1 nm filter at 650 nm (`filter_lpqi_650_1nm`,
  backend `lpqi_mpd_650`), red continuum 6 nm blueward of Hα, for all
  nights. The movie: separations and fringe periods change night by night,
  the best night detects its fringe at 4σ (η Ori: 1.5σ), and the distance
  panel converges against the Gaia B + C band, the B and C lines and the
  Hipparcos band: 386 pc (344–434) with three alias solutions after night
  one, 382 pc (367–398) with five after four, a single solution from night
  six on, and after all eight a = 1.628 mas with 68 % range 1.595–1.660
  (±2 %), d = 378 pc (371–386); the Fisher σ(d)/d is 1.6 % (fringe only
  2.0 %). Over 20 noise realizations
  the rms scale error is 1.5 % with the truth in the deepest island 85 % of
  the time. Hipparcos's 714 pc would give a fringe period 1.9 times longer
  and is excluded in one night; the 359 and 384 pc of the two companions
  differ by 7 % and are separated after four nights; the result is the
  first direct distance of the pair, independent of the companions. Caveats: node angle assumed 0 (90° gives
  ~1 %); third light 7 % from Ab at 155 mas; the infobox parameters should
  be checked against Pablo et al. 2017; the 650 nm filter is not in the
  confirmed LPQI set.

The movies are written to `output/campaigns/<campaign>/<campaign>.mp4`
(PNG frames when ffmpeg is absent); `--no-figures` computes the numbers only.

## Assumptions to confirm with the LPQI team

1. The filter list is confirmed (Hα, Hβ, 500, 550, 425 nm; 1 nm; one per
   night); the exact central wavelengths, widths and the 0.9 throughput are
   still working values. A 650 nm red continuum filter (1 nm, beside Hα) is
   proposed here for binaries with wind-affected Hα (ι Ori).
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
6. The field stop at the camera: λ Ori B (V 5.6 at 4.4″), η Ori B (V 4.95 at 1.7″; the binary movie assumes it excluded, Ac at 44 mas cannot be), β Cep's
   companions must be excluded for the hot-star diameters above.
