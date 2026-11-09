# R. Kaiser's O/B dwarfs on the ORM telescopes, spring/summer 2027 (`g2_kaiser_orm_2027`)

R. Kaiser (INPHYNI, email 2026-11-09, "Targets La Palma") will bring two or three time-tag
cameras to La Palma in 2027 and asked which ORM telescope combinations (2 or 3 telescopes) give
the best S/N on λ Ori A, υ Ori, 10 Lac and λ Lep. His instrument: a 1 nm filter at 417 nm (or
431 nm), a polarizing beam splitter with one detector per polarization on each telescope, 20 %
detection efficiency, 35 ps rms jitter for the whole chain. In the catalog: detector
`inphyni_lapalma` (82 ps FWHM, dead time / dark rate / 10 Mcps ceiling ASSUMED), spectrographs
`filter_inphyni_417_1nm` and `_431_1nm` (throughput 0.9 ASSUMED), backends `inphyni_417_pbs`
and `inphyni_431_pbs` (`polarization_mode` pbs: half the flux and full contrast per channel,
the two channels in quadrature, √2 over unpolarized detection).

The telescopes are the five of `lpqi_orm` (NOT 2.5 m, TNG 3.5 m, GTC 10.4 m, WHT 4.2 m,
INT 2.5 m; all with the catalog's ASSUMED optical throughput 0.3), pairwise 409 m (TNG–GTC) to
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
| **NOT + TNG + GTC** | 4.2 | 2.4 | 1.7 | 3.1 | **11.4** |
| TNG + GTC + WHT | 4.6 | 3.3 | 2.1 | 3.6 | 13.6 |
| TNG + GTC + INT | 4.7 | 3.7 | 2.4 | 3.7 | 14.4 |
| **TNG + GTC** (two telescopes) | 4.7 | 3.7 | 2.4 | 3.7 | **14.5** |
| NOT + WHT + INT | 6.3 | 10.7 | 11.3 | 7.1 | 35 |
| WHT + INT (two telescopes) | 7.2 | 21.7 | 25.1 | 12.7 | 67 |
| NOT + WHT (two telescopes) | 47 | 23 | 22 | 17 | 108 |
| NOT + GTC (two telescopes) | 57 | 8.8 | 6.2 | 52 | 124 |
| NOT + TNG (the Pathfinder pair) | 222 | 33 | 34 | 28 | 318 |

S/N per night on |V|² (431 nm, all pairs of the combination): NOT+TNG+GTC 2.4 (λ Ori A),
10.0 (υ Ori), 8.4 (10 Lac), 8.2 (λ Lep); TNG+GTC alone 2.3, 9.7, 7.8, 8.1. Every target but
λ Ori A is a 3σ detection in a fraction of a night; λ Ori A needs 1.6 nights because the
TNG–GTC baseline (270–420 m) sits at and beyond its null (|V|² 0.01–0.25).

What the ranking says:

- **The GTC is the whole game.** Its 78 m² collects 8.6× the TNG's photons; the TNG–GTC pair
  (409 m, the shortest on the mountain, 137–420 m projected) carries 90–97 % of the Fisher
  information of any combination that contains it. Two cameras on TNG + GTC already beat every
  three-telescope set without the GTC by a factor 2.5 or more.
- **The third camera belongs on the NOT**, not on the WHT or INT: NOT–TNG (300–530 m) and
  NOT–GTC (300–680 m) are the next-shortest baselines and still see |V|² 0.05–0.6 on the three
  dwarfs, cutting the nights by 25–35 % over TNG + GTC alone. A third camera on the WHT or INT
  adds only the 700–1500 m baselines, where these stars are resolved out (< 5 % gain).
- **Without the GTC**, the best pair is WHT + INT (422 m, 230–420 m projected: the same
  geometry as TNG–GTC with 1/6 of the photons), 4–10× slower; NOT + WHT + INT is the best
  GTC-less triple at 35 nights for the set. The Pathfinder pair NOT + TNG is the wrong baseline
  for stars this small (300–530 m is past the null for λ Ori A and λ Lep).
- The GTC saturates the ASSUMED 10 Mcps per-channel ceiling on all four stars (1.1–2.6 × 10⁷
  cps per channel): the forecasts attenuate it to the ceiling, so a detector that takes more
  gains up to 1.6× in S/N on the GTC pairs.

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

Robin's detector dead time, dark rate and maximum count rate (the 10 Mcps ceiling matters on the
GTC); the filter/relay throughput (0.9 taken); the ORM telescopes' optical throughput to a
Nasmyth/Cassegrain focus at 420 nm (0.3 taken for all five); the three 2R/d diameters (no
interferometric measurement; the 5 % forecast is on whatever the true value is).
