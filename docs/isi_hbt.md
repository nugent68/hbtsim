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
| targets | Betelgeuse 42.3 mas (×1.12 at 650 nm, ×1.02 at 900), Aldebaran 20.58, Arcturus 21.05, μ Cep 14.1, ε Tau 2.59, HD 360 0.91; Mira 32 (×1.4 at 650); Altair 3.67 × 2.96 and Regulus 1.65 × 1.25 (oblate); Capella (56 mas binary), Spica (e = 0.108) | interferometric diameters (Montargès 2014, Richichi 2005, Lacour 2008, Perrin 2005, Woodruff 2004, Monnier 2007, McAlister 2005); ε Tau from R/d; Torres et al. 2015; Harrington 2016; the TiO-band factors **assumed** |
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

## The red-clump movie (`hbtsim run movie_epstau_isi`)

ε Tau at 900 nm, nine nights at 20, 30, 45, 60, 75, 90, 110, 140 and 180 m:
a 2.59 mas K0 III red-clump giant whose |V|² leaves 0.9 at 20 m, crosses the
87 m null and vanishes into a first sidelobe of 0.01. Unlike the bright
giants, ε Tau sits only marginally at the PDM's cap, so its σ(|V|²) of 0.86
per 5-min block is set by its own V 3.5 photons: the first nights detect the
fringe at 9–11σ, the null night at 1.5σ, and the diameter reaches 5.9 % after
the nine nights (3.8 % at the single best baseline, 45 m, every night); the
sidelobe nights add nothing and the limb darkening is unconstrained. A season
of 30 nights at 30–75 m would tie this Hyades red-clump calibrator to about
2 %, which is the level at which the red-clump radius scale becomes useful.

The same nine nights at 650 nm with the baselines scaled to that filter's
63 m null (`movie_epstau_isi_650`: 14, 22, 33, 44, 55, 66, 80, 100, 130 m)
give 4.0 % instead of 5.9 %: ε Tau sits below the PDM's cap, so the 0.37
efficiency at 650 nm against the assumed 0.11 at 900 nm outweighs the
shorter coherence time and the star's colour (the fringe at 16σ on the first
night, the null night at 2.3σ). A 550 nm filter, where the PDM peaks at 0.49,
would be worse than either (5.2 %): the coherence time of a 1 nm band scales
as λ² and the giants are red, so for the uncapped red-clump star 650 nm is the
best filter and for the capped bright giants the reddest filter always wins.

## The third truck: three baselines a night (`movie_epstau_isi_line`, `g2_epstau_isi_line`)

With the trucks in an East–West line at spacings (a, b) the night measures
the baselines a, b and a + b at once (`arrays/isi_line`, a `line` generator;
schedule entries `{"spacings_m": [a, b]}`; both runners take every pair of
the night's array). For ε Tau at 650 nm the nights 25+35, 35+40 and 45+50 m
(baselines 25/35/60, 35/40/75, 45/50/95 m, bracketing the 63 m null and
reaching the first sidelobe) give, twice over:

| plan | nights | σ(θ)/θ | σ(u) |
|---|---|---|---|
| pair, 9 nights 14–130 m | 9 | 4.0 % | 4.5 |
| pair, 3 nights 33/44/55 m | 3 | 4.7 % | — |
| line, 3 nights | 3 | 3.1 % | 4.9 |
| line, 6 nights (the movie) | 6 | 2.2 % | 3.5 |
| line, a season: 20 nights at 650 nm, 10 at 900 nm | 30 | 1.15 % | 1.2 |

Three trucks for three nights beat two trucks for nine. The movie shows the
three pairs' points filling the curve together: the short pair on the first
lobe, the long pair past the null, the night's fringe detected at 22σ on the
first night and 11σ on the sidelobe-reaching third. The limb darkening is the
honest negative of the ISI with the PDM: even thirty nights leave σ(u) = 1.2
on a coefficient of 0.6, because the sidelobe of a 2.6 mas star sits at
|V|² ≈ 0.01 and the per-night σ(|V|²) is 0.06; reaching σ(u) = 0.1 would take
~4000 such nights, or a detector without the dead-time cap. The diameter, at
1 % in a season, is the measurement the trucks deliver.

## The first week: four science cases

What the three trucks can do in a week that nobody has done -- each a
campaign of its own (`hbtsim run movie_<name>`), each a season (Spica in
spring; Betelgeuse, Mira and Capella in winter; Altair in summer–autumn;
Regulus in winter–spring), each movie ending on the fitted quantity. Numbers
from the runs with seed 7; "Fisher" is the expected precision, "fit" the
realized one.

**The first intensity-interferometric stellar shape** (`movie_altair_isi_oblate`,
`movie_regulus_isi_oblate`). The Narrabri and all later intensity
interferometry fitted circular disks. Altair (3.67 × 2.96 mas, Monnier et al.
2007) and Regulus (1.65 × 1.25 mas, McAlister et al. 2005) are 24 % and 32 %
oblate; the equilateral triangle gives three position angles 60° apart every
block, and a second night with the triangle rotated by 30° samples six.
Altair at 650 nm on 30 m sides: 22σ fringes per night, σ(q) = 0.070 after one
night and 0.053 after two (q − 1 = 0.24 at 4.5σ), σ(θ_major)/θ 2.4 %, 170° of
position angle covered; Regulus on 70 m sides: 20σ per night, σ(q) 0.091 →
0.057 (q − 1 = 0.32 at 5.6σ). The oblateness mode (`options.oblateness`) now
takes any array and a schedule of configurations; a collinear line is ~4×
worse than the triangle because every pair of a line sees the same position
angle. Gravity darkening (Altair's pole 1600 K hotter than its equator) is not
modelled: the measured shape is the photometric one.

**Betelgeuse in the TiO bands against the continuum** (`movie_betelgeuse_isi_chromatic`).
A target's `chromatic_diameter` table draws a wavelength-dependent disk (×1.12
at 650 nm, ×1.02 at 900 nm, **assumed** from the optical TiO-band 45–55 mas
against the 42–44 mas near-infrared continuum); the rotation runner fits the
diameter per wavelength and reports the ratio. Three nights -- 650 nm at
2.5+2.5 m, 900 nm at 2.5+3 m, 650 nm at 2.5+3 m (the 47 mas TiO disk nulls at
3.45 m, so the trucks at their 2.5 m minimum sit on the first lobe and the
5–5.5 m pair past the null) -- give θ(650) to 2.3 % and θ(900) to 1.5 %:
ratio 1.098 ± 0.030, the molecular layer at 3.2σ. The ratio measures the
layer above the photosphere directly, in one week, with the same instrument
at both wavelengths.

**Spica's orbit and its apsidal motion since 1971** (`movie_spica_isi_orbit`).
Spica's orbit was the only intensity-interferometric binary orbit ever made
(Herbison-Evans et al. 1971, Narrabri, a = 1.54 ± 0.05 mas). With e = 0.108
and an apsidal period of 139 ± 6 yr (Robinette & Aufdenberg) the periastron
has advanced ~142° since. The three trucks in a 60 m triangle at 650 nm (the
0.91 mas primary nulls at 180 m; the pair's fringe period λ/a is 78 m at
quadrature, so the triangle sits inside the first lobe) for seven consecutive
nights from phase 0.2 (1.75 orbits, never within hbtsim's near-eclipse margin:
i = 63°) give 20–25σ fringes per night and, by Fisher, σ(a)/a 1.7 %,
σ(Ω) 2.3°, σ(ω) 2.3°. The global χ² over (scale, node, periastron) of the run
finds a = 1.763 mas (68 % 1.729–1.814; truth 1.711), the node to 2° and
ω = 50° (46–54; truth 52): |V|² is even in the separation, so ω is measured
modulo 180° and the spectroscopic orbit picks the half. Against the assumed
1971 ω = 90° ± 10° that is an advance of +140 ± 11° on the expected branch --
the uncertainty is the 1971 orbit's, not ours. The collinear line (40+60 m)
was tried first and gives 2.7 % on a with the per-night (dx, dy) fit on a
lattice of aliases; the triangle locks it. The 1971 ω and the convention of
Harrington et al.'s 232° (secondary) are flagged in the catalog.

**The first HBT size of a long-period variable, and Capella's orbit**
(`movie_mira_isi`, `movie_capella_isi_orbit`). Mira (32 mas **assumed** at
900 nm near maximum, Woodruff et al. 2004) sits just under the PDM's cap, so
its S/N is its own: the line at 2.5+3, 3+4 and 4+5 m (first lobe, the 7.1 m
null, the first sidelobe) gives 52, 41 and 28σ per night and 1.8 % on the
diameter per night, 1.0 % in three -- a snapshot of a star whose diameter
breathes by ±10 % over 332 days and whose TiO-band disk is ~40 % larger
(`chromatic_diameter`; a 650 nm night would measure the layer). Capella
(a = 56.4 mas, P = 104 d) is the test of a binary orbit against a known one:
the fringe period λ/ρ = 3.3 m at 900 nm is twice the 1.65 m pupils, so the
point-sampled |V|² (0.05 at 7 m) is wrong by a factor four and the pupil-
averaged model (`options.pupils`, now in the binary runner, with the fringe
amplitude weighted across the pupil) is the one to fit: 0.21. Seven nights on
the 5+7 m line (baselines 5, 7, 12 m; the 8.5 and 6.2 mas giants are resolved
beyond ~25 m) move the pair 24° around the orbit and give σ(a)/a 0.23 % and
σ(Ω) 0.12° by Fisher, the global fit landing 0.3 % (1σ) from the assumed a --
an independent 13.12 pc (68 % 13.09–13.16) check of Torres et al. 2015's
orbital parallax (13.159 pc, 0.04 %), not an improvement on it; Capella never eclipses, so no
image grid is needed (its 8192-pixel grid was what stopped the earlier
runner).

| case | campaign | nights | configuration | result |
|---|---|---|---|---|
| Altair's shape | `movie_altair_isi_oblate` | 2 | triangle 30 m, 650 nm, rotated 30° | q = 1.24 at 4.5σ |
| Regulus's shape | `movie_regulus_isi_oblate` | 2 | triangle 70 m, 650 nm, rotated 30° | q = 1.32 at 5.6σ |
| Betelgeuse TiO/continuum | `movie_betelgeuse_isi_chromatic` | 3 | line 2.5–5.5 m, 650 and 900 nm | θ(650)/θ(900) = 1.098 ± 0.030 |
| Spica's apsides | `movie_spica_isi_orbit` | 7 | triangle 60 m, 650 nm | a 1.7 %, ω ± 2.3° (mod 180) |
| Mira's size | `movie_mira_isi` | 3 | line 2.5–9 m, 900 nm | θ to 1.0 % |
| Capella's orbit | `movie_capella_isi_orbit` | 7 | line 5+7 m, 900 nm, pupil-averaged | a 0.23 %, Ω 0.12° |

Sources: Monnier et al. 2007, Science 317, 342 (Altair); McAlister et al.
2005, ApJ 628, 439 (Regulus); Woodruff et al. 2004, A&A 421, 703 (Mira);
Herbison-Evans et al. 1971, MNRAS 151, 161 and Harrington et al. 2016, ApJ
827, 104 (Spica); Robinette & Aufdenberg (Spica's apsidal period); Torres et
al. 2015, ApJ 807, 26 (Capella).

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
