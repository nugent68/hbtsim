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
`detectors/{lpqi_spad64_i2cass,lpqi_spad_nextgen}`,
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
- Wikipedia infoboxes (read 2026-10-06) for the five telescopes' coordinates,
  altitudes, apertures and collecting areas.

## Parameters and status

<!-- catalog:lpqi_parameters -->
|  | value | source / status |
|---|---|---|
| site | Roque de los Muchachos, Nordic Optical Telescope (LPQI reference point): 28.7573°, -17.8851°, 2382 m | Wikipedia (NOT infobox) |
| Pathfinder pair | NOT 2.56 m + TNG 3.58 m, B = 550 m at PA 219° | 550 m published (lapalmaqi.es); coordinates give 471 m — **to be confirmed** |
| five-telescope network | NOT 2.56 m, TNG 3.58 m, GTC 10.4 m, WHT 4.2 m, INT 2.54 m; baselines 447–1547 m | Wikipedia coordinates, arc-second precision |
| telescope throughput | 0.3, 0.3 | hbtsim default (**assumed**) |
| detector (published) | IMSE 64×64: PDE 0.026 (fill factor 3.5 % × PDP 75 %), 500 ps FWHM, dead 10 ns, dark 1.68 cps/pixel, 25 pixels, readout ≤ 6.7e+06 cps | Quintana et al. 2026, Sensors 26, 5757; timing = White Rabbit target (**assumed**), n_pixels **assumed** |
| detector (next-gen) | PDE 0.50 peak (SPAD Lambda curve), 100 ps, readout ≤ 4e+07 cps | **assumed** (the paper names a higher-efficiency sensor as the next generation) |
| filters | 500.0 nm (1.0 nm); 550.0 nm (1.0 nm); 656.3 nm (1.0 nm); 486.1 nm (1.0 nm); throughput 0.9; one per night | wavelengths **to be confirmed**; width and one-per-night from the user |
<!-- /catalog -->

The filter wavelengths (Hα, Hβ, 500 and 550 nm, 1 nm wide) are a working
assumption chosen with the user on 2026-10-06; nothing published names them.
The arrays' ENU positions come from arc-second Wikipedia coordinates (≈ 30 m
per coordinate): they give NOT–TNG = 471 m against the project's 550 m, and
GTC–INT = 1547 m, the project's "1.5 km". `arrays/lpqi_pathfinder` adopts the
published 550 m along the coordinate-derived position angle (219°);
`load_array("lpqi_pathfinder", baseline_m=471.0)` uses the coordinate length.

## What the Pathfinder does for hbtsim's targets

**Resolution.** At B = 550 m and 500 nm the first null of a uniform disk sits
at θ = 1.22 λ/B = 0.23 mas, so every star in the catalog is resolved out:
the binary components (0.45–1.1 mas), Sirius A (6.0 mas) and Vega (3.3 mas)
all sit many nulls beyond. The Pathfinder is built for 50–200 µas sources
(white dwarfs, hot compact stars); for our binaries the one-night |V|² is
10⁻⁴–10⁻³ (table below), not a measurement.

**Sensitivity.** A 1 nm filter gives a coherence time τ_c = λ²/(cΔλ) ≈ 0.8 ps
at 500 nm, against a pair timing width of 300 ps (published array, 500 ps FWHM
per detector) or 60 ps (next-gen assumption); the g² S/N scales with the
photon rate (so with the PDE) and with the coherence time over the square root
of the timing width, and a single channel has no multiplexing gain. In the
campaign below the assumed next-generation array (PDE 0.5 peak, 100 ps) gains
a factor 25–40 per night over the published one on the same filter; both are
orders of magnitude below what the same telescopes would do behind a
320-channel spectrograph.

**One filter per night.** The backends `lpqi_<filter>` are exclusive: a night on
one filter is a night not spent on the others. The g2 runner's
`one_backend_per_night` option therefore **adds** the nights of the filter set
(it never combines them in quadrature), and prints the rule.

### The four binaries on NOT + TNG (`hbtsim run g2_binaries_lpqi_pathfinder`)

One night along the uv track (30-min blocks above 30°; NewEra atmospheres where
the grid covers the stars, the blackbody + Claret model otherwise), the
published array and the assumed next-generation one; "nights" is the time to a
3σ detection of |V|² at the night's mean value. δ Vel (dec −54.7°) never rises
above 30° from La Palma.

| target | SED | backend | B [m] | |V|² over the night | SNR/night | nights (3σ) |
|---|---|---|---|---|---|---|
| β Aur | NewEra | 1 nm Hα, IMSE 64×64 | 314–550 | 1×10⁻⁴–2×10⁻³ | 8×10⁻⁴ | 1.3×10⁷ |
| β Aur | NewEra | 1 nm 550 nm, IMSE 64×64 | 314–550 | 2×10⁻⁵–2×10⁻³ | 1.4×10⁻³ | 4.7×10⁶ |
| β Aur | NewEra | 1 nm 550 nm, next-gen (assumed) | 314–550 | 2×10⁻⁵–2×10⁻³ | 0.057 | 2.8×10³ |
| Algol | NewEra (A) | 1 nm Hα, IMSE 64×64 | 331–550 | 8×10⁻⁶–3×10⁻³ | 1.4×10⁻³ | 4.4×10⁶ |
| Algol | NewEra (A) | 1 nm 550 nm, next-gen (assumed) | 331–550 | 4×10⁻⁶–2×10⁻³ | 0.058 | 2.7×10³ |
| Spica | blackbody | 1 nm Hα, IMSE 64×64 | 314–550 | 1×10⁻⁴–3×10⁻³ | 0.007 | 1.8×10⁵ |
| Spica | blackbody | 1 nm Hα, next-gen (assumed) | 314–550 | 1×10⁻⁴–3×10⁻³ | 0.21 | 207 |
| Spica | blackbody | 1 nm 550 nm, next-gen (assumed) | 314–550 | 2×10⁻⁵–3×10⁻³ | 0.17 | 306 |

Filter set (four filters, one per night, nights added): β Aur 4×10⁸ nights
(IMSE) / 2.7×10⁵ (next-gen); Algol 3×10⁷ / 2×10⁴; Spica 1×10⁶ / 1.8×10³. The
full table (all 24 observable rows) is `output/campaigns/g2_binaries_lpqi_pathfinder/table.md`.
The next-generation rows on Spica are readout-limited at the 4×10⁷ cps ceiling
(×0.5–0.6); the published array never is (its rates are 10⁶–10⁷ cps per
telescope through a 1 nm filter).

### Sirius B on NOT + TNG (`hbtsim run mc_sirius_b_lpqi_pathfinder`)

The Pathfinder's own use case: a 28.5 µas white dwarf (V = 8.44), 10 h at
zenith angles 46°, 52.5°, 60° (Sirius culminates at 45.5° from the NOT), one
500 nm / 1 nm filter, unpolarized, geometric-mean aperture √(A_NOT A_TNG).

| detector | rate per telescope | pair σ_t | analytic σ(θ)/θ in 10 h | hours to 10 % |
|---|---|---|---|---|
| IMSE 64×64 (published) | 2.3×10³ cps | 300 ps | 1.8×10³ (no measurement) | 3.4×10⁹ |
| next-gen (assumed) | 4.4×10⁴ cps | 60 ps | 42 | 1.8×10⁶ |
| next-gen + PBS + 2400 m | — | 60 ps | 3.7 | 1.4×10⁴ |

The Poisson Monte Carlo cannot fit a diameter from these histograms (the
coincidence excess is below the accidentals' shot noise by orders of
magnitude); the analytic matched-filter precision is what the table reports.
The EON-SII pair reaches 10 % on the same star in 1.5 h (MCP-PMT) because it
has 1000 channels, 27 ps and a 20 % PDE: with one 1 nm channel the Pathfinder
needs ≈ 10 ps timing *and* a high-PDE array *and* a spectrograph to approach it.

## Assumptions to confirm with the LPQI team

1. The filter list: central wavelengths, widths (1 nm assumed), throughput, and
   whether more than one filter can be used per night (one assumed).
2. The NOT–TNG baseline (550 m published vs 471 m from the dome coordinates)
   and the station positions of the five-telescope network.
3. The Pathfinder detector's PDE curve (2.6 % flat assumed), the time-stamp
   resolution actually reached with White Rabbit (500 ps target assumed), the
   number of pixels the seeing disk covers (25 assumed), and the cooled dark
   count rate.
4. The next-generation sensor (`lpqi_spad_nextgen` is a placeholder: SPAD
   Lambda PDE, 100 ps, 4×10⁷ events/s).
5. Telescope + relay throughput to the camera (0.3 × 0.9 assumed; Prada et al.
   use an overall 0.5).
