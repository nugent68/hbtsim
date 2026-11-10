# R. Kaiser's O/B dwarfs on the ORM telescopes, spring/summer 2027 (`g2_kaiser_orm_2027`)

R. Kaiser (INPHYNI, email 2026-11-09, "Targets La Palma") will bring two or three time-tag
cameras to La Palma in 2027 and asked which ORM telescope combinations (2 or 3 telescopes) give
the best S/N on λ Ori A, υ Ori, 10 Lac and λ Lep. His instrument: a 1 nm filter at 417 nm (or
431 nm), a polarizing beam splitter with one detector per polarization on each telescope, 20 %
detection efficiency, 35 ps rms jitter for the whole chain. In the catalog: detector
`inphyni_lapalma` (82 ps FWHM taken as the single-detector value, dead time / dark rate ASSUMED,
no count-rate ceiling: Robin wrote on 2026-11-10 that their limit is in integrated counts, not
rate), spectrographs
`filter_inphyni_417_1nm` and `_431_1nm` (throughput 0.9 ASSUMED), backends `inphyni_417_pbs`
and `inphyni_431_pbs` (`polarization_mode` pbs: half the flux and full contrast per channel,
the two channels in quadrature, √2 over unpolarized detection).

The telescopes are the five of `lpqi_orm` (effective areas with the central obstruction:
NOT 4.80, TNG 9.12, GTC 73.0, WHT 12.70, INT 4.41 m²; all with the catalog's ASSUMED optical
throughput 0.3), pairwise 409 m (TNG–GTC) to
1518 m (GTC–INT). The runner gives every pair's |V|² and S/N along one night's track above 30°;
`scripts/kaiser_orm_combos.py` adds the pairs of each 2- or 3-telescope combination in quadrature
(S/N) and sums their Fisher information on θ (nights to 5 % on the diameter).

## The targets

| star | type | V | θ_LD (mas) | first null at 417 nm | source of θ |
|---|---|---|---|---|---|
| λ Ori A | O9 III | 3.70 | 0.238 | 441 m | 2R/d; CHARA/PAVO 0.226 ± 0.015 (Gordon et al. 2018) |
| λ Lep | B0.5 V | 4.29 | 0.161 | 652 m | 2R/d (R 4.5 R☉, 260 pc) |
| υ Ori | O9.7 V | 4.62 | 0.133 | 790 m | 2R/d (R 5.5 R☉, 386 pc) |
| 10 Lac | O9 V | 4.88 | 0.125 | 840 m | 2R/d (R 7.4 R☉, 552 pc); CHARA 0.11 ± 0.02 |

All four are far below CHARA's 0.2 mas practical limit at 331 m: any fringe detected at the
400–700 m ORM baselines is a higher-resolution measurement than VLTI/CHARA have made. But the
stars are so small that the longest baselines (TNG–WHT, GTC–WHT, TNG–INT, GTC–INT, 700–1500 m)
sit beyond the first null with |V|² < 0.01: they contribute nothing in a night. The useful
baselines are the short ones, 250–450 m, where |V|² is 0.2–0.8.

## Ranking (431 nm filter; 417 nm is 5–10 % worse)

Nights to 5 % on θ, one night = the target's track above 30°:

| telescopes | λ Ori A | υ Ori | 10 Lac | λ Lep | sum |
|---|---|---|---|---|---|
| **NOT + TNG + GTC** | 0.85 | 2.2 | 1.8 | 1.8 | **6.6** |
| TNG + GTC + WHT | 0.92 | 3.1 | 2.2 | 2.0 | 8.2 |
| TNG + GTC + INT | 0.93 | 3.4 | 2.6 | 2.0 | 8.9 |
| **TNG + GTC** (two telescopes) | 0.93 | 3.4 | 2.6 | 2.0 | **8.9** |
| NOT + WHT + INT | 7.6 | 12.4 | 13.1 | 8.3 | 41 |
| NOT + GTC (two telescopes) | 11.2 | 8.1 | 6.6 | 28 | 54 |
| WHT + INT (two telescopes) | 9.0 | 27 | 31 | 16 | 83 |
| NOT + WHT (two telescopes) | 50 | 25 | 24 | 18 | 117 |
| NOT + TNG (the Pathfinder pair) | 222 | 33 | 34 | 28 | 318 |

(Run `g2_kaiser_orm_2027_nocap`, 2026-11-10: obstructed areas, no count ceiling. The 2026-11-09
version of this table, with unobstructed mirrors and a 10 Mcps ceiling applied to both telescopes
of a pair, read 4.2 / 2.4 / 1.7 / 3.1 for NOT+TNG+GTC.)

S/N per night on |V|² (431 nm, all pairs of the combination): NOT+TNG+GTC 5.4 (λ Ori A),
10.4 (υ Ori), 8.2 (10 Lac), 11.1 (λ Lep); TNG+GTC alone 5.2, 10.1, 7.6, 11.0. Every target is a
3σ detection in a fraction of a night (λ Ori A in 0.3 nights: the TNG–GTC baseline, 270–420 m,
sits at and beyond its null, |V|² 0.01–0.25, so it lives on the GTC's full count rate).

What the ranking says:

- **The GTC is the whole game.** Its 73 m² collects 8× the TNG's photons; the TNG–GTC pair
  (409 m, the shortest on the mountain, 137–420 m projected) carries 90–97 % of the Fisher
  information of any combination that contains it. Two cameras on TNG + GTC already beat every
  three-telescope set without the GTC by a factor 2.5 or more.
- **The third camera belongs on the NOT**, not on the WHT or INT: NOT–TNG (300–530 m) and
  NOT–GTC (300–680 m) are the next-shortest baselines and still see |V|² 0.05–0.6 on the three
  dwarfs, cutting the nights by 10–35 % over TNG + GTC alone. A third camera on the WHT or INT
  adds only the 700–1500 m baselines, where these stars are resolved out (< 5 % gain).
- **Without the GTC**, the best pair is WHT + INT (422 m, 230–420 m projected: the same
  geometry as TNG–GTC with 1/6 of the photons), 8–12× slower; NOT + WHT + INT is the best
  GTC-less triple at 41 nights for the set. The Pathfinder pair NOT + TNG is the wrong baseline
  for stars this small (300–530 m is past the null for λ Ori A and λ Lep).
- **Count rates.** Per telescope, both channels, 431 nm: GTC 24 / 14 / 10 / 8 Mcps for
  λ Ori A / λ Lep / υ Ori / 10 Lac, TNG 3.0 / 1.7 / 1.3 / 1.0, WHT 4.1 / 2.4 / 1.8 / 1.4, NOT and
  INT 0.5–1.6 Mcps. A 10 Mcps per-telescope ceiling (run `g2_kaiser_orm_2027_cap10fix`, the
  attenuation applied to the GTC only) would cost 2× on λ Ori A (1.7 nights with NOT+TNG+GTC),
  27 % on λ Lep (2.2) and nothing on υ Ori and 10 Lac, which sit at or under it. The counts a
  10 % diameter costs with NOT+TNG+GTC are 6–7 × 10¹⁰ per channel on the GTC, 8–9 × 10⁹ on the
  TNG and 4–5 × 10⁹ on the NOT, nearly the same for all four stars.

## When

Dark hours with the target above 30° from the NOT on the 1st of each month of 2027
(astronomical night):

| month | λ Ori A | υ Ori | 10 Lac | λ Lep |
|---|---|---|---|---|
| Jan | 8.2 | 6.7 | 2.1 | 5.9 |
| Feb | 6.0 | 5.2 | 0 | 4.6 |
| Mar | 3.8 | 3.0 | 0 | 2.4 |
| Apr | 1.3 | 0.5 | 0 | 0 |
| May–Jun | 0 | 0 | 0.7–2.3 | 0 |
| Jul | 0 | 0 | 4.4 | 0 |
| Aug | 0 | 0 | 6.8 | 0 |
| Sep | 1.4 | 0.7 | 8.5 | 0.6 |
| Oct | 3.7 | 3.0 | 7.8 | 2.8 |

The three Orion/Lepus stars are a winter programme: "spring 2027" means February at the latest
(3–6 h per night, falling to nothing by April). 10 Lac is the summer target (July–September,
4–8 h per night, dec +39 so it passes near the zenith). A spring/summer 2027 run therefore
means 10 Lac, with the Orion stars either in February or in the autumn.

## Assumptions to confirm

Robin's detector dead time and dark rate; whether the 35 ps rms is per detector (taken: pair
49.5 ps) or already the pair's resolution (S/N 1.19× higher); the S/N prefactor (the code uses the
matched Gaussian filter, 0.53 in Guerin et al. 2025's form; their recommended three-parameter fit
gives 0.44, i.e. 1.46× more time); the filter/relay throughput (0.9 taken); the ORM telescopes' optical throughput to a
Nasmyth/Cassegrain focus at 420 nm (0.3 taken for all five); the three 2R/d diameters (no
interferometric measurement; the 5 % forecast is on whatever the true value is).

## Changes on 2026-11-10

- Effective areas with the central obstruction for all five telescopes (NOT 4.80, TNG 9.12,
  GTC 73.0, WHT 12.70, INT 4.41 m²; commits `cbd37f9`, `7381318`): 15–25 % more time everywhere.
- The INPHYNI count ceiling removed (Robin: the limit is integrated counts, not rate): λ Ori A
  with NOT+TNG+GTC goes from 4.2 to 0.85 nights.
- The g2 runner attenuated *both* telescopes of a pair by the stricter of the two readout
  scales (`scale = min(...)` applied through `mag_eff`); it now attenuates each telescope by its
  own detector's requirement. With a ceiling on the GTC this is a 2.4× difference in time for
  λ Ori A on TNG+GTC. The same shared-scale pattern remains in `snr.py` (spectral), `snr3.py`
  and the movie runners.
- Hours of on-target time to a 10 % diameter (what Robin was sent): NOT+TNG+GTC 1.7 / 3.7 / 4.3 /
  2.6 h for λ Ori A / υ Ori / 10 Lac / λ Lep; TNG+GTC 1.9 / 5.6 / 6.3 / 3.0; NOT+WHT+INT 15 / 20 /
  32 / 12; WHT+INT 18 / 45 / 77 / 23.
