# The ISI as a near-infrared intensity interferometer

The Infrared Spatial Interferometer (ISI, UC Berkeley: three 1.65 m
Pfund-type telescopes on semi-trailers, 11 µm heterodyne, Mt. Wilson
1988–2010s) re-equipped for photon counting: one ID Quantique ID230
InGaAs/InP SPAD per telescope behind a 1 nm filter at 900 or 1550 nm, the
trucks driven to a flat California site (Palomar stands in) where the
baseline can be changed between nights from 3 m to 1 km. Because the array
moves, a programme can sample |V|²(B) from 1 through the first null into the
sidelobes night by night, which is what the new `baseline_schedule` /
`schedule` options of the g2 and night-movie runners do.

## Sources

- Hale et al. 2000, ApJ 537, 998: the ISI (three 1.65 m telescopes, fully
  mobile, 4–70 m at Mt. Wilson); Wikipedia: Mt. Wilson coordinates.
- ID Quantique ID230 brochure (G.192.0115-PB-1.0, June 2026): 900–1700 nm,
  efficiency 10/15/20/25 % calibrated at 1550 nm, jitter 150 ps typical and
  200 ps maximum at 25 %, dead time 2–100 µs, dark < 80 Hz at 10 % and
  < 200 Hz at 20 %, SMF or MMF 62.5 µm input.
- P. Nugent, 2026-10-21: Palomar, baselines to 1 km, ID230 at 25 % with the 2 µs
  dead time, 1 nm filters at 900 and 1550 nm, targets red giants and
  supergiants, red clump stars and binaries.
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
| telescopes | three ISI 1.65 m Pfund telescopes on trucks; throughput 0.2 including the fibre coupling into the ID230 | Hale et al. 2000, ApJ 537, 998; throughput **assumed** |
| configurations | pair (`isi_pair`, default 100 m East–West, min spacing 2.5 m) and equilateral triangle (`isi_triangle`, default 100 m); the campaigns' `baseline_schedule` / `schedule` set a configuration per night, e.g. 3, 5, 8, 12, 20, 35, 60, 100, 200, 400, 700, 1000 m | the trucks (user, 2026-10-21) |
| detector (ID Quantique ID230) | InGaAs/InP SPAD, PDE 0.25 at 1550 nm and 0.10 at 900 nm, 150 ps FWHM, dead 2 µs, dark 200 cps, operating cap 2.5e+05 cps (half of 1/dead time) | ID230 brochure (June 2026); PDE away from 1550 nm and the cap **assumed** |
| filters | 1550 nm (1.0 nm); 900 nm (1.0 nm); throughput 0.9 | the user (2026-10-21); widths and throughput **assumed** |
| targets | Betelgeuse 42.3 mas, Aldebaran 20.58, Arcturus 21.05, μ Cep 14.1, ε Tau 2.59, HD 360 0.91; Capella (56 mas binary) | interferometric diameters (Montargès 2014, Richichi 2005, Lacour 2008, Perrin 2005); ε Tau from R/d; Torres et al. 2015 |
<!-- /catalog -->

## What the dead time means

The ID230's 2 µs dead time is the instrument. With the non-paralyzable model
r/(1 + rτ) trustworthy only below r·τ ≈ 1, the catalog carries an
*operating cap* of 2.5×10⁵ incident counts per second (half of 1/τ, load 0.5,
1.7×10⁵ detected), which the runners apply as a neutral-density attenuation
(`readout_scale`): Betelgeuse at 1550 nm is cut by ×0.009, Aldebaran and
Arcturus by ×0.02–0.04, ε Tau by ×0.7, and only HD 360 runs uncapped. So every
giant brighter than H ≈ 1 gives the same count rate and the same
signal-to-noise: with τ_c = 8 ps (1550 nm, 1 nm) against a pair timing of
90 ps, that is about 2.2 per hour at |V|² = 1, 6 per night; at 900 nm (τ_c
2.7 ps, PDE 0.10 assumed) 2 per night. The programme is detector-limited, and
the brightness of the targets buys nothing: the information is in geometry.

## The surveys (`hbtsim run g2_giants_isi`, `g2_binaries_isi`)

Twelve configurations from 3 m to 1 km East–West, one night per filter each,
30-min blocks above 30°, blackbody + rough linear limb darkening (no model
atmosphere exists below 4700 K), the Fisher information on the diameter θ and
on the linear limb-darkening coefficient u combined over the nights
(`table_combined.md`):

| target | θ (mas) | S/N over 24 nights | σ(θ)/θ, θ alone | σ(θ)/θ jointly with u | σ(u) |
|---|---|---|---|---|---|
| Betelgeuse | 42.3 | 5.8 | 15 % | 135 % | 13 |
| Aldebaran | 20.58 | 10 | 13 % | 131 % | 13 |
| Arcturus | 21.05 | 10 | 13 % | 158 % | 16 |
| μ Cep | 14.1 | 13 | 12 % | 113 % | 11 |
| ε Tau | 2.59 | 17 | 15 % | 149 % | 15 |
| HD 360 | 0.91 | 2.7 | 119 % | — | — |

The diameters come from the nights near the first null (Betelgeuse 3–9 m,
Aldebaran and Arcturus 5–20 m, μ Cep 8–35 m, ε Tau 60–200 m) and reach
12–15 % after a season; the sidelobes, at |V|² ≈ 0.01 against a per-night
σ(|V|²) of 0.16, are never seen, so the limb darkening is unconstrained
(σ(u) ≫ 1) and the "full g² curve" is in practice the null and the first lobe.
At 900 nm the same nights are three times weaker (PDE 0.10 assumed, τ_c three
times shorter), so the 900 vs 1550 nm size comparison of the supergiants'
molecular layers is at the 30 % level per season. The binaries
(`g2_binaries_isi`, 5 m to 1 km): Capella's fringes are detected at 3–4.6σ per
night at 5–12 m and gone beyond 35 m (two 6–8 mas giants resolved), a 3σ
detection in 1.7 nights but no orbit; β Aur, Algol and Spica are detected in
0.6–1 night on the 300–1000 m configurations, their g/i anchors extrapolated
to the near-IR.

## The movie (`hbtsim run movie_betelgeuse_isi`)

Betelgeuse at 1550 nm, the trucks stepping the East–West baseline 3, 5, 7, 9,
12, 16, 25 and 40 m on eight nights of 5-min blocks (log-scale |V|² panel):
the measurements walk from |V|² ≈ 0.9 at 3 m through the 9.2 m null; the
16–40 m nights show nothing (0.0σ each), and the diameter settles at
σ(θ)/θ = 13 % with the first four nights, the joint fit with u never closing
(146 %). Eight nights at the single best baseline (5 m) would give 8 %.

## What would change the verdict

- A detector without the 2 µs dead time. A superconducting nanowire (the
  ID281: < 25 ps, > 95 %, no comparable dead time) or splitting the beam onto
  several ID230s removes the cap and lets the giants' 10⁷–10⁸ cps count; the
  numbers then scale with the rate, i.e. by 10–100 in S/N.
- Multimode-fibre coupling: a 10–30 m MMF adds 0.5–1.5 ns of modal
  dispersion, which would make the 150 ps jitter irrelevant and cost another
  factor of several; the throughput 0.2 assumes short fibres.
- The ID230's PDE at 900 nm (0.10 assumed at the InP window edge) decides
  whether the 900 nm filter is useful at all.

Open assumptions are flagged in the catalog files: the ID230 PDE curve, dark
rate at 25 % and cap convention; the telescope throughput; the giants' H/K
anchors (2MASS saturates) and distances (Betelgeuse 168 vs 222 pc, μ Cep
421 +57/−42); ε Tau's diameter from R/d; the rough M-giant limb-darkening law;
and the sense of Capella's node angle in hbtsim's convention.
