# The ISI as a red / near-infrared intensity interferometer

The Infrared Spatial Interferometer (ISI, UC Berkeley: three 1.65 m
Pfund-type telescopes on semi-trailers, 11 µm heterodyne, Mt. Wilson
1988–2010s) re-equipped for photon counting: one free-beam Micro Photon
Devices PDM 100 µm silicon SPAD per telescope at prime focus behind a 1 nm
filter at 650 or 900 nm, the trucks driven to a flat California site
(Palomar stands in) where the baseline can be changed between nights from
2.5 m to 1 km. Because the array moves, a programme can sample |V|²(B) from
1 through the first null into the sidelobes night by night, which is what the
`baseline_schedule` / `schedule` options of the g2 and night-movie runners do.
(An InGaAs ID230 at 1550 nm was modelled first and dropped: its 2 µs dead
time capped every giant at the same 6σ per night; see the git history.)

## Sources

- Hale et al. 2000, ApJ 537, 998: the ISI (three 1.65 m telescopes, fully
  mobile, 4–70 m at Mt. Wilson); Wikipedia: Mt. Wilson coordinates.
- PicoQuant / MPD PDM series datasheet (February 2018, 25 °C): spectral
  range ~400–1000 nm peaking near 550 nm; PDE 24 % at 400, 49 % at 550, 37 % at
  650 nm (free-beam; the fibre version loses up to 20 % absolute); 100 µm active
  area (~4″ at the ISI prime focus); NIM timing ~50 ps FWHM (TTL 250 ps);
  dead time 77 ns (7–8 % loss at 1 Mcps, 25 % at 4 Mcps); afterpulsing ≤ 3 %
  (0.13 % typical); dark-count grades for 100 µm down to < 25 cps. The ASU
  intensity-interferometry group's 100 µm units: 32 ps jitter, 40 % at 589 nm
  through the fibre connector, 50 cps dark.
- P. Nugent, 2026-10-21: Palomar, baselines to 1 km, the PDM 100 µm free-beam
  module (replacing the ID230), 1 nm filters at 900 nm and, since silicon does
  not see 1550 nm, 650 nm; targets red giants and supergiants, red clump stars
  and binaries.
- Targets: Montargès et al. 2014 (Betelgeuse 42.28 mas), Richichi &
  Roccatagliata 2005 (Aldebaran 20.58), Lacour et al. 2008 (Arcturus 21.05),
  Perrin et al. 2005 (μ Cep 14.1), Anugu et al. 2026 and Gaia DR3 (ε Tau
  2.59 from R/d), Torres et al. 2015 (Capella's orbit); Ducati 2002 and 2MASS
  for the V/H/K anchors.

## Parameters and status

<!-- catalog:isi_parameters -->
|  | value | source / status |
|---|---|---|
| site | Palomar Mountain (generic flat California site for the relocated ISI, baselines to 1 km): 33.356°, -116.865°, 1712 m | Palomar 200-inch coordinates; a stand-in for any flat California site (**assumed**) |
| telescopes | three ISI 1.65 m Pfund telescopes on trucks; throughput 0.5 including the fibre coupling into the ID230 | Hale et al. 2000, ApJ 537, 998; throughput **assumed** |
| configurations | pair (`isi_pair`, default 100 m East–West, min spacing 2.5 m) and equilateral triangle (`isi_triangle`, default 100 m); the campaigns' `baseline_schedule` / `schedule` set a configuration per night, e.g. 2.5, 3, 4, 5, 7, 10, 14, 20, 35, 60, 100, 200, 400, 700, 1000 m | the trucks (user, 2026-10-21) |
| detector (MPD PDM 100 µm, free-beam) | silicon SPAD, PDE 0.49 at 550 nm, 0.37 at 650 nm, 0.11 at 900 nm; 50 ps FWHM (NIM), dead 77 ns, dark 50 cps, operating cap 4e+06 cps (load 0.31) | PDM datasheet (Feb 2018); PDE beyond 650 nm, timing of the 100 µm unit and the cap **assumed** |
| filters | 650 nm (1.0 nm); 900 nm (1.0 nm); throughput 0.9 | 900 nm from the user (2026-10-21), 650 nm replacing the 1550 nm filter the silicon SPAD cannot see; widths and throughput **assumed** |
| targets | Betelgeuse 42.3 mas, Aldebaran 20.58, Arcturus 21.05, μ Cep 14.1, ε Tau 2.59, HD 360 0.91; Capella (56 mas binary) | interferometric diameters (Montargès 2014, Richichi 2005, Lacour 2008, Perrin 2005); ε Tau from R/d; Torres et al. 2015 |
<!-- /catalog -->

## What the dead time means

The PDM's 77 ns dead time and an operating cap of 4×10⁶ incident counts per
second per detector (the user's "a few Mcps per channel"; load r·τ = 0.31,
25 % dead-time loss, the non-paralyzable model applied) still mean that every
giant brighter than R ≈ 2 is attenuated to the same rate: Betelgeuse at 900 nm
by ×0.16, Aldebaran and Arcturus by ×0.2–0.4, ε Tau barely, HD 360 not at all.
So the bright giants share one signal-to-noise per night: with τ_c = 2.7 ps at
900 nm against a pair timing of 30 ps (50 ps FWHM per detector), σ(|V|²) =
0.145 per 30-min block and 0.036 per night, i.e. 38σ at |V|² ≈ 0.7 and the
first sidelobe at |V|² ≈ 0.009 at 0.25σ per night. That is 16 times the
ID230's cap and three times its timing, which is what moves the diameters
from 13 % to 1 % per season; it is still not enough for the sidelobes.

## The surveys (`hbtsim run g2_giants_isi`, `g2_binaries_isi`)

Fifteen configurations from 2.5 m (the trucks' minimum spacing) to 1 km
East–West, one night per filter each, 30-min blocks above 30°, blackbody +
rough linear limb darkening (no model atmosphere exists below 4700 K), the
Fisher information on the diameter θ and on the linear limb-darkening
coefficient u combined over the nights (`table_combined.md`):

| target | θ (mas) | S/N over 30 nights | σ(θ)/θ, θ alone | σ(θ)/θ jointly with u | σ(u) |
|---|---|---|---|---|---|
| Betelgeuse | 42.3 | 57 | 1.1 % | 10 % | 0.80 |
| Aldebaran | 20.58 | 123 | 0.95 % | 8.6 % | 0.66 |
| Arcturus | 21.05 | 121 | 0.94 % | 10 % | 0.91 |
| μ Cep | 14.1 | 69 | 2.0 % | 18 % | 1.3 |
| ε Tau | 2.59 | 60 | 4.5 % | 48 % | 4.0 |
| HD 360 | 0.91 | 6.3 | 52 % | — | — |

The diameters come from the nights inside and just beyond the first null
(Betelgeuse 2.5–7 m at 900 nm, Aldebaran and Arcturus 3–14 m, μ Cep 5–20 m,
ε Tau 35–100 m) and reach about 1 % per season for the four bright giants,
2 % for μ Cep and 4.5 % for the red-clump star ε Tau. The limb darkening is
not reached: the first sidelobe of a 20–40 mas star at 900 nm sits at |V|² ≈
0.009 against a per-night σ of 0.036, so u comes out at ±0.7–0.9 over a season
and the joint diameter degrades to 9–10 %. The 650 nm nights are three times
weaker than the 900 nm ones (τ_c 1.4 ps, the stars more resolved) but reach
the TiO-band size of the supergiants, so the 650 versus 900 nm comparison is a
few-percent measurement per season. The binaries (`g2_binaries_isi`, 5 m to
1 km): Capella's fringes are detected at 29–40σ per night at 5–12 m and gone
beyond 35 m (two 6–8 mas giants resolved), 69σ over the eleven nights but no
orbit; β Aur, Algol and Spica are detected at 77–95σ over the schedule, their
information in the 300–1000 m nights, their g/i anchors extrapolated to the
red.

## The movie (`hbtsim run movie_betelgeuse_isi`)

Betelgeuse at 900 nm, the trucks stepping the East–West baseline 2.5, 3.5,
4.5, 5.5, 7, 9, 12, 16 and 25 m on nine nights of 5-min blocks (log-scale
|V|² panel): the measurements walk from |V|² ≈ 0.8 at 2.5 m through the 5.4 m
null, the 7 m night still detects the fringe at 3σ, and the 9–25 m nights,
in the sidelobes, see nothing. The diameter settles at σ(θ)/θ = 1.2 % after
nine nights (nine nights at the single best baseline, 3.5 m, would give
0.7 %); the joint fit with u stays at 18 % with σ(u) = 1.4.

## What would change the verdict

- The sidelobes need σ(|V|²) ≈ 0.003 per night, ten times better: more
  nights at 7–12 m (a season of 100 nights at one baseline gets there), a
  wider filter (10 nm: ten times the photons, but then the cap bites harder),
  or a detector without the cap: splitting the beam onto several PDMs raises
  the 4 Mcps ceiling in proportion.
- A measured IRF of the 100 µm unit at 900 nm: 32 ps (ASU) instead of the
  50 ps datasheet figure is a factor 1.5 in signal-to-noise; a diffusion tail
  is the other way.
- The PDE at 900 nm, assumed 0.11 at the edge of the silicon response, sets
  the whole 900 nm programme; the 650 nm filter is on firmer ground (0.37
  measured) but sees the molecular layers of the supergiants rather than the
  photosphere.

Open assumptions are flagged in the catalog files: the PDM PDE curve between
and beyond the datasheet's three points, the 100 µm unit's timing and the cap
convention; the telescope throughput (0.5, free-beam, coatings built for
11 µm); the giants' H/K anchors (2MASS saturates) and distances (Betelgeuse
168 vs 222 pc, μ Cep 421 +57/−42); ε Tau's diameter from R/d; the rough
M-giant limb-darkening law; and the sense of Capella's node angle in hbtsim's
convention.
