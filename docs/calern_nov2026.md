# C2PU at H-alpha, 16-20 November 2026: what 20-30 hours on the 15 m pair can do

Written 2026-11-04 from F. Izraelevitch's email of 2026-11-01 (OCA/INPHYNI, with the Delft
SNSPD team). Catalog entries: telescope `c2pu_1m` (0.737 m² effective, transmission
0.93 × 0.93 × 0.8), array `c2pu_pair` (15 m east-west), detectors `snspd_delft` (PDE 0.80,
100 ps RMS) and `apd_calern` (PDE 0.60, 500 ps RMS), spectrographs `calern_halpha_1nm_snspd`
/ `_apd` (656.28 nm, 1 nm, the branch optics), backends `calern_snspd_halpha` and
`calern_apd_halpha` (each `single_pol`: one output of the polarizing beam splitter), campaign
`g2_calern_halpha_nov2026` (runner `g2`, both branches combined in quadrature). The site is
the C2PU placard (06°55′23″ E, 43°45′13″ N, 1270 m) and the mount's pointing limits from the
Epsilon chart (|hour angle| < 5.3 h, declination < 60°, altitude > 21°) are on the telescope
(`max_hour_angle_h`, `max_dec_deg`, `min_alt_deg`, applied to every track). Atmospheric
transmission 0.90 and the detectors' dead times, dark rates and maximum count rates are
assumptions (see the JSON).

## The instrument in one line

At 15 m and 656 nm a star reaches the first null at 11 mas; a 3 mas star is 20 % resolved
and a 5-8 mas star sits at |V|² 0.1-0.7. The SNSPD branch carries 90 % of the sensitivity
(the APD branch, with five times the jitter, adds 4 % in quadrature). The two polarizations
on separate detectors give full g² contrast on each, at half the flux, which is why
`single_pol` and the 0.48 of the splitter are not both applied.

## One night per target, both branches

Night hours are those inside the mount's limits (altitude > 21°, |H| < 5.3 h) during the
astronomical night of 16-20 November at Calern (LST 21.3 h to 9.1 h); the hour-angle cap,
not the horizon, bounds the circumpolar targets at 10.6 h. S/N is on |V|² per hour of integration; "hours to 1 %" is the Fisher
precision on the diameter with the limb darkening fixed and the instrumental contrast known.

| target | θ or ρ [mas] | |V|² at 15 m | night hours | S/N per hour (both) | σ(θ)/θ in 5 h | hours to 1 % |
|---|---|---|---|---|---|---|
| Vega | 3.33 | 0.72-0.87 | 2.6 | 25 | 3.7 % | 69 |
| Procyon A | 5.40 | 0.41-0.83 | 5.8 | 17 | 2.3 % | 27 |
| Rigel | 2.76 | 0.80-0.90 | 6.7 | 24 | 5.2 % | 136 |
| Pollux | 7.98 | 0.13-0.62 | 6.6 | 6.9 | 3.2 % | 51 |
| Hamal | 6.83 | 0.24-0.76 | 10.1 | 4.9 | 6.2 % | 192 |
| Aldebaran (main lobe only at |H| > 4 h, B < 6 m) | 20.6 | 0.00-0.13 | 9.6 | 1.2 | 6.6 % | 220 |
| Deneb | 2.36 | 0.85-0.92 | 4.7 | 11 | 16 % | 1210 |
| Mirfak | 3.19 | 0.74-0.84 | 10.6 | 7.3 | 12 % | 750 |
| Capella (ρ 56 mas, 6-8 mas giants; attenuated to the 10⁷ cps ceiling) | fringes, period 2.4 m | 0.05-0.29 | 9.1 | 5.2 | binary | - |
| β Aur (ρ 3.3 mas, P 3.96 d) | fringes, period 41 m | 0.25-0.98 over the orbit | 9.0 | 4.5 | binary | - |
| Algol (ρ 2.2 mas) | fringes, period 61 m | 0.92-0.99 | 10.6 | 4.6 | binary | - |
| ε, γ, δ¹, θ¹ Tau (red clump) | 2.3-2.7 | 0.81-0.98 | 9.7-9.9 | 1.7-2.2 | 75-135 % | 28 000-90 000 |
| Regulus | 1.44 | 0.94-0.99 | 3.8 | 9.4 | 57 % | 16 000 |
| ζ Tau (Be; H-alpha disk 3.1 × 1.0 mas) | disk FWHM 3.14 | 0.72-1.01 (effective) | 8.8 | 3.9 | disk FWHM to 30 % | - |
| γ Cas (Be; H-alpha disk 3.6 × 2.1 mas; declination limit overridden) | disk FWHM 3.59 | 0.55-0.82 (effective) | 9.0 | 9.1 | disk FWHM to 6 % | - |

Count rates on the SNSPD branch reach 1.0-1.4 × 10⁷ cps per telescope on Vega and Capella
(dead-time load 0.3-0.4 at the assumed 30 ns). The ceiling is 10⁷ cps (R. Kaiser,
2026-11-06: "optimistic that the SNSPD can handle 10 Mcps", to be confirmed); it was an
assumed 2 × 10⁷ before. Only Capella is over it: attenuated by 28 %, its S/N falls from 6.5
to 5.2 per hour. Vega sits at the ceiling; every other target is below it.

**SNSPD against APD.** The SNSPD branch has 3.6 times the S/N of the APD branch at low
rates (3.1-3.2 on Vega and Capella, where the dead time bites): 1.63 from the efficiency
(0.80 against 0.82 × 0.60, the APD branch having one more 0.82 optic) times √5 = 2.24 from
the jitter (100 against 500 ps RMS, both far above the 1.4 ps coherence time, so S/N goes
as jitter^-1/2). In quadrature the APDs add 4 % to the SNSPD S/N; "90 % of the sensitivity"
above is the SNSPD's share of S/N², the statistical weight.

## What the week can do

1. **The HBT peak itself**, in minutes: 3 σ on Vega and Rigel in one minute, on γ Cas in seven,
   on Capella in twenty, on the Hyades giants in two hours. Every bright star in the list is a commissioning
   target for the SNSPD/APD pair.
2. **Capella's fringes in one night**: the projected baseline runs from 11 to 15 m, two
   fringe periods, with |V|² from 0.05 to 0.32 at S/N 5 per hour (attenuated to the 10⁷ cps ceiling). The 104-day orbit is known
   to 0.1 % (Torres et al. 2015), so this is a clean end-to-end test of the baseline vector,
   the pupil smearing and the sign of the modulation.
3. **β Aur's orbital modulation across the week**: |V|² swings between 0.26 and 0.98 over
   the 3.96-day orbit at the 41 m fringe period; 2-3 hours a night at S/N 4.7 per hour maps
   it at 10 % per point through five nights. This is the "right location of the modulation"
   case: the pair is always inside the first fringe, so the modulation is in orbital phase,
   not in hour angle.
4. **A diameter at a few percent on a 3-8 mas star**: Procyon to 2 % in 5 h (morning,
   5 h a night), Vega to 2 % in its 15 h of evening visibility, Pollux to 3 % in 5 h, Rigel to
   5 %. These are stars whose diameters are known to 0.5 % (Procyon, Vega) or 1-2 % (Pollux,
   Hamal), so the result is a calibration of the instrumental contrast rather than new
   astrophysics; it needs an unresolved calibrator (Regulus at S/N 9 per hour in its 3.8 morning hours, or ζ Tau)
   observed in the same configuration, because at |V|² 0.4-0.7 a 2 % diameter needs the
   contrast to 2 %.

## What it cannot do

- **Red-clump giants**: the Hyades giants are 20 % resolved at 15 m and give S/N 2 per
  hour: 25 h detects their g² at 10 σ and measures the diameter to 70 %. The 1 % blue
  diameters need 40-50 m at 425 nm (`docs/hyades_ld.md`). HD 17652 never rises above 30°
  at Calern; HD 360 (0.9 mas) is unresolved.
- **Eclipsing-binary diameters**: β Aur's components (1 mas) and Algol's are unresolved;
  the week gives the binary fringe, not the stellar disks.
- **Algol's modulation**: 3-8 % on |V|² at S/N 4.6 per hour, below 1 σ per hour.
- **Supergiant H-alpha**: Deneb and Rigel are 10-20 % resolved; the line-forming wind
  would need a dedicated line-versus-continuum model, which the catalog does not have.
- **γ Cas inside the official limits**: it is north of the mount's 60° declination limit
  (see the Be-star section for the unofficial case).


## Be-star disks at H-alpha: γ Cas and ζ Tau (added 2026-11-06)

R. Kaiser (2026-11-06): γ Cas is formally beyond the declination limit, but it was observed
with one C2PU telescope in 2018 with J.-P. Rivet overriding the mount's safeties. Campaign
`g2_calern_halpha_be_disks` runs it with telescope `c2pu_1m_override` (declination limit 61°,
not an official configuration; the hour-angle range reachable there is unknown and ± 5.3 h
is kept), together with ζ Tau.

**Model** (`hbtsim.single.LineDisk`, `line_disk_vis2`; target entry `line_disk`). In a
channel that contains the line the star is the photosphere plus an elliptical Gaussian disk
that carries a Gaussian emission line; l(λ) is the line-to-continuum ratio. One detector
integrates the channel, so the g² signals of its wavelengths add against a shared noise and
the photon budget takes

    |V|²_eff = ⟨|V_star + l V_disk|²⟩ / ⟨1 + l⟩²,   count rate × ⟨1 + l⟩.

At zero baseline |V|²_eff = ⟨(1 + l)²⟩ / ⟨1 + l⟩² > 1: the longer coherence time of a
spectrum concentrated in a line (Matthews et al. 2023, AJ 165, 117, Eq. 4). Channels off the
line see the bare photosphere, so the blue MAGIC and LPQI continuum campaigns are unchanged.
The disk's own continuum (< 1 mas) is neglected.

| | γ Cas | ζ Tau |
|---|---|---|
| disk (Gaussian FWHM, minor/major, PA of the major axis) | 3.59 mas, 0.58, 31° (Tycner et al. 2006) | 3.14 mas, 0.31, −62° (Tycner et al. 2004) |
| line in the model | peak 5 × continuum, FWHM 340 km/s, 32 Å in the core (Matthews et al. 2023, Fig. 3, January 2022) | 13 Å (9-16 Å in 2022-2024, Nazé et al. 2024), FWHM 400 km/s assumed |
| line / continuum photons in the 1 nm filter | 2.8 (disk carries 74 %) | 1.1 (52 %) |
| rate per telescope, SNSPD branch | 3.7 × 10⁶ cps | 1.2 × 10⁶ cps |
| projected baseline over the track | 13-15 m | 6-15 m |
| \|V\|²_eff over the track | 0.55-0.82 | 0.72-1.01 |
| S/N per hour, both branches | 9.1 | 3.9 |
| disk FWHM in 5 h / one 9 h night / 25 h | 6 % / 4.5 % / 2.7 % | 30 % / 22 % / 13 % |

**Check against the group's own data.** With the January 2022 line in the ~1.05 nm I2C filter
the model gives a coherence time of 1.46 ps (top-hat filter) against 1.41 ps expected and
1.35 ± 0.05 ps measured by Matthews et al. Their 13-18 m point (0.59 ± 0.10 of the
zero-baseline area) lies between the model's 0.44 along the major axis and 0.74 across it;
their 18-21 m and 32-38 m points (0.22 ± 0.06, 0.05 ± 0.03) sit at the major-axis values
(0.29, 0.08).

**Reading.** γ Cas is the best-matched science target of the week: four times the continuum
count rate, a disk the 15 m baseline half resolves, 3 σ in seven minutes and the disk size
to 5 % in a night. With one east-west baseline the measurement is the extent along the
projected baseline (59° from the major axis at transit), so the 5 % is on the size with the
axis ratio and position angle taken from NPOI; it is a repeat of the 2022 I2C result at
higher S/N, not a new shape. ζ Tau is fainter and its line weaker: a detection in an hour
and a 13 % size in 25 h. Both forecasts scale with the line strength in November 2026, which
is not known: the γ Cas line is that of 2022, ζ Tau's is the middle of a variable range. A
spectrum taken during the run fixes l(λ), which the analysis needs in any case.
