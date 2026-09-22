# The bending height model

An **option** of the depth stage, off by default. With
`depth_3d.height_model: "bending"` each filament is given a height every few
pixels along its length instead of one height, under a minimum bend radius
that follows from the bundle's own stiffness and adhesion. The over/under
order, the crossings, the evidence and everything two-dimensional are exactly
what the flat model produces; only the heights change.

This document is the method, the measurements behind its constants, what it
does on real fields, and how to switch it off or remove it. The
implementation is [`plecta/bending.py`](plecta/bending.py); the tests are
[`tests/test_bending.py`](tests/test_bending.py).

## Why

`solve_metric_z` gives every filament ONE height (`assumptions.planar_instances:
dz/ds = 0`). A filament that passes over one neighbour and under the next
cannot be drawn that way, so every transitive over/under step becomes global
stack height, and the film grows with the crossing graph rather than with the
specimen. On B58-B3-S2_100 the flat model reads 121 nm at seven stack levels;
the sharpest example filament (id 100, 501 px long) sits 81.6 px above the
floor because it is over two neighbours that are themselves stacked, while
elsewhere along its length it is under others. Bent, its mean height is 21.8
px and the film is 64 nm.

## The physics

**What bends.** A bundle of (10,10) single-wall tubes, free to shear past one
another (the sliding bound of `plecta_dem.rule`, the softer of the two
bundle models), so its bending stiffness is the sum of its tubes':

    EI = N EI_1,    EI_1 = pi C (d_tube / 2)^3 = 3.38e-25 N m^2,
    N = (pi/4)(d - gap)^2 / ((sqrt 3 / 2) a0^2)   (hexagonal packing)

with C = 345 N/m the graphene in-plane stiffness (Kudin, Scuseria & Yakobson
2001), d_tube = 1.356 nm, a0 = 1.67 nm and gap = a0 - d_tube = 0.317 nm.
EI_1 sits between the two values PLECTA-DEM transcribed for the same tube,
K_bnd = 3.04e-25 N m^2 (Volkov & Zhigilei 2010) and Y I = 3.58e-25 (Ostanin
et al. 2013); `tests/test_bending.py` pins it there.

**What holds it down.** Adhesion to whatever it lies on, w_a per unit length
lifted: three tube-pair lines of contact at the Girifalco (2000) well depth,
w_a = 0.44 nN. This is a tube-scale, radius-blind floor -- the same choice
PLECTA-DEM keeps as its sensitivity floor ("tube-pair").

**The balance.** A ramp of rise h over arclength L, clamped flat at both
ends (the cubic z = h (3u^2 - 2u^3)), costs bending energy 6 EI h^2 / L^3 and
gives up adhesion w_a L. The sum is least at

    L^4 = 18 EI h^2 / w_a.

With h = d (one diameter) and L = rho d this is the dimensionless control

    rho^4 = 18 EI / (w_a d^2),

and because N grows as d^2 the diameter cancels: rho = 8.02 at 8 nm, 8.09 at
13.4 nm, 8.14 at 30 nm. The ramp's end curvature 6h / L^2 is independent of
the rise, and is the constraint the solver applies:

    R_min = rho^2 d / 6 = sqrt(EI / (2 w_a)).

The second form is the classical peel radius -- the curvature at which
bending energy per unit length EI kappa^2 / 2 equals the adhesion per unit
length w_a -- and does not depend on the ramp shape assumed above. It is
proportional to d: 146 nm on a 13.4 nm bundle, 61 nm on the thinnest and 286
nm on the thickest bundle of the reference field. rho's literal reading
("diameters of arclength to rise one diameter") is the clamped-cubic value;
the shape the solver actually produces, two arcs of radius R_min, rises one
diameter in 2 sqrt(R_min d) = 0.82 rho d.

**Against buckling.** A single (10,10) tube buckles at R_cr = 27.5 nm (Volkov
& Zhigilei 2010). Every bend this model allows is at least 61 nm, so the
harmonic bending energy the derivation assumes is valid throughout. PLECTA-DEM
scales R_cr with the bundle radius (272 nm for a 13.4 nm bundle), which is the
welded picture; under the sliding picture used here each tube buckles at its
own 27.5 nm. The two projects are consistent within their own assumptions and
differ in this one, which is stated rather than resolved.

**The constants are a choice among literature values**, and rho goes as the
fourth root of every one of them:

| adhesion w_a per unit length | source | rho |
|---|---|---|
| 0.44 nN (three tube pairs, Girifalco 2000) | shipped | 8.09 |
| 1.07 nN (three tube pairs at Ostanin's 2.23 eV/nm) | PLECTA-DEM's transcribed well depth | 6.5 |
| 0.69 nN (Derjaguin, two parallel 13.4 nm bundles) | PLECTA-DEM's base-case adhesion model | 7.2 |

and taking K_bnd for EI_1 instead of the shell value moves 8.09 to 7.88. The
film thickness moves faster than rho does: on the reference field 58 nm at rho
4, 64 nm at 8.09, 80 nm at 16, 105 nm at 40, and the flat model's 121 nm as
rho -> infinity. None of these was tuned; 8.09 is what the shipped constants
give, and the table is what the alternatives give.

## What is solved

For filament n, heights h_n,k at knots spaced d_n apart along its in-plane
arclength (about `bend_sample_px`, with a knot at each end). Between knots the
profile is the cubic Hermite curve with central-difference slopes
(Catmull-Rom). That curve is **linear in the knot heights**, so its curvature
at both ends of every interval -- and hence everywhere, since the second
derivative of a cubic is linear -- is a linear function that the LP bounds
directly. The filament that is drawn is the filament that was constrained.

    minimise    sum_n d_n sum_k h_n,k                  the lifted area: the flat
                                                        model's compact prior,
                                                        integrated along each
                                                        filament
    tie-break   sum_n,k |h_n,k+1 - h_n,k|              flatness, at 1e-3 px^2
                                                        per px

    subject to  h_n,k >= r_n                           resting on the substrate
                z_hi(s*) - z_lo(t*) >= r_hi + r_lo + gap
                                                        at every constrained
                                                        crossing, on the samples
                                                        bracketing it, with the
                                                        chord margin (d/q)^2/(8R)
                z_hi(a) - z_lo(b) >= sqrt(sep^2 - delta_ab^2)
                                                        at every pair of samples
                                                        of the two filaments
                                                        within sep in projection
                |z_n''(s)| <= 1 / R_min(d_n)           everywhere on the drawn
                                                        curve
                z_n'(0) = z_n'(L_n) = 0                flat at both ends

Directions come from `PredCrossing.over` where the evidence decided and from
the global order where it abstained, read exactly as `solve_metric_z` reads
them. The solver writes to no crossing, no order and no centreline.

The flatness term is a tie-break, not a trade: on the reference field it
leaves the lifted area within 1e-9 of the two-stage lexicographic minimum
(area first, then variation) and the variation within 4e-5 of it, at one
ninth of the cost. `_FLATNESS_WEIGHT = None` runs the two-stage solve, and
`test_two_stage_lexicographic_solve_agrees_with_the_weighted_one` checks it.

The flat ends are a boundary condition: nothing lifts an end, so a filament
is horizontal where it stops or leaves the frame. They are what makes rho ->
infinity **exactly** the flat model (zero curvature alone permits a straight
but tilted rod, and the compact prior prefers one: without the condition the
stiff limit came out at 134 nm with 113 tilted filaments against the flat
model's 121 nm).

The substrate convention differs from the flat model's in one respect: here
the filaments that rest on the substrate rest on it bottom-flush (h >= r),
whereas the flat sweep puts their centrelines coplanar (z >= 0). With equal
radii the two are the same solution shifted; with unequal radii they differ by
at most the spread of the bottom radii, and `test_unequal_radii_in_the_stiff_limit_differ_only_by_the_substrate`
pins that difference to the least element of the bending constraints.

Cost: 9,612 height variables, 67,000 rows and 1.3 s on the 307-filament
reference field at the default 6 px spacing (HiGHS through `scipy.optimize.
linprog`); deterministic, and bit-identical on re-run.

## What changes, and what never does

Changes: `z` per instance (now the mean of its profile), the film thickness
(measured over the profiles' envelope), the per-instance extent, and the
report fields `height_model`, `bend_rho`, `bend_min_radius_px`,
`bend_sample_px`, `film_thickness_px`. `assumptions.planar_instances` reads
`|d2z/ds2| <= 1/R_min(d), dz/ds = 0 at the ends`. The interpenetration gate
`n_interpenetrating` evaluates each profile where the pair meets.

Never changes: every crossing and its evidence, `over`, `flipped`, the global
order and its report fields, `layer` (K), the centrelines, the radii, and every
number of the default `compact_stack` run, which is bit-identical to the
package before the option existed (`tests/test_bending.py::TestRunScene::
test_default_output_carries_no_bending_field`, and the byte-for-byte
comparison of `pred_depth.json` on the three skeleton fields below).

## On real fields

All at the defaults (rho 8.09, 6 px, crossings constrained, grazing pairs
free), 1.08 nm/px:

| field | filaments | crossings | flat film | bent film | R_min (median d) | K | added time |
|---|---|---|---|---|---|---|---|
| B58-B3-S2_100 manual annotation | 307 | 414 | 121.3 nm | **64.3 nm** | 146 nm | 6 | 1.3 s |
| b58_100 skeleton | 282 | 342 | 121.9 nm | **68.6 nm** | 163 nm | 7 | 1.7 s |
| b58_110 skeleton | 385 | 507 | 131.4 nm | **67.8 nm** | 143 nm | 7 | 2.3 s |
| b58_300 skeleton | 309 | 606 | 199.1 nm | **81.7 nm** | 164 nm | 9 | 5.6 s |

"Added time" is the bending solve on the manual annotation and, for the
skeleton fields, the wall-clock difference between `python -m plecta.depth`
runs with and without the option (4.1-5.4 s without, grouping included).
The manual annotation slices into three levels by height (mean height in
median diameters) against seven by order.

With grazing pairs constrained too (`clear_grazing_overlaps`), the manual
annotation reads 154.3 nm flat and 81.1 nm bent (859 constrained pairs, 3.0 s).

On the manual annotation, 165 of 307 filaments bend and 140 never leave their
level; the largest rise is 45 px; the drawn curves' tightest radius equals the
bound (ratio 1.000, minimum and median) and no constrained pair is short of its
clearance where it meets. Across the whole projected overlap of every
constrained pair the residual interpenetration between samples is at most
0.09 px (0.29 px with one sample per knot, 0.03 px with four). Grazing pairs
left free interpenetrate more often than under the flat model (325 against
165 of 445), because the film is more compact; the gate reports it, as it
does for the flat model.

**Spacing.** `bend_sample_px` is a discretisation and the film converges from
above, because a coarser chain cannot realise the tightest allowed ramp:

| spacing | 1.5 px | 3 px | 4 px | 6 px | 9 px | 12 px |
|---|---|---|---|---|---|---|
| film | 60.8 nm | 61.1 | 61.8 | **64.3** | 67.1 | 67.4 |
| time | 26.7 s | 5.6 | 2.8 | **1.3** | 0.7 | 0.4 |

6 px is the interactive compromise and reads 6 % above the converged value;
3 px is within 0.5 % of it.

## What the tests prove

Known answers (`TestKnownAnswers`): a filament over one neighbour and under
the next gives a two-diameter film where the flat model gives three, with the
upper filament back on the substrate one ramp 2 sqrt(R h) away and its drawn
curve never tighter than R_min; a weave over five neighbours is two diameters
deep and mirror-symmetric; two neighbours closer than a ramp hold the filament
level between them; rho of 1e3, 1e5 and 1e7 reproduce the flat sweep to a
tolerance that follows the cap and vanishes with it; unequal radii in that
limit differ from the flat sweep only by the substrate convention.

Properties (`TestProperties`): the order, the crossings and the centrelines
are byte-identical before and after; every constrained pair is the right way
up and clear where it meets; across the whole overlap zone the residual is
under 0.15 px; every drawn curve respects its reported radius; the film is
never thicker than the flat model's; the film converges from above in the
spacing; the solve is deterministic; the two-stage lexicographic solve agrees
with the weighted one.

Degenerate geometry (`TestDegenerate`): no crossings, one crossing, a closed
loop, a 0.5 px filament, a zero-length filament, a one-point filament, an
instance without a radius, coincident crossings between different pairs, and
an abstained pair with no order position.

Constants (`TestConstants`): rho is 8.09 and moves by under 0.07 across
8-30 nm; R_min = rho^2 d / 6 equals sqrt(EI / 2 w_a) to 1e-12; EI_1 lies
between the two transcribed values; R_min scales with d.

## Limitations

- The curvature bounded is the out-of-plane part only. The in-plane curvature
  comes from the image and cannot change; a filament that is already bent in
  the plane is allowed the same out-of-plane radius as a straight one.
- The compact prior is a prior. Nothing in the image measures a height, so
  the profiles are the lowest configuration consistent with the order and
  the stiffness, not a measurement.
- The adhesion is tube-scale and radius-blind; the bundle-scale alternatives
  above give rho 6.5-7.2 and a thinner film. rho is exposed for exactly this
  reason.
- Grazing pairs are cleared only when `clear_grazing_overlaps` is on, as in
  the flat model; free, they interpenetrate more often here.
- The residual interpenetration across an overlap zone is the sub-sample
  spacing's, at most 0.09 px at the defaults, not the flat model's zero.
- The film at 6 px is 6 % above the converged value; 3 px is within 0.5 %.
- The order is decided by the flat model (the `compact` costing of the two
  candidate arrangements uses the flat sweep); bending never influences which
  filament is on top.

## Switching it off, or removing it

Off: `depth_3d.height_model: "compact_stack"` (the default). Nothing else is
needed; the two other keys are then inert.

Removing the option entirely means deleting `plecta/bending.py`,
`tests/test_bending.py`, this file, and every block fenced by
`# -- bending option --` ... `# -- end bending option --` in `plecta/depth.py`
(the import, the three `DepthParams` fields, the branch in `run_scene`, the
gate branch, the `assumptions` line and the profile branch in `tube_mesh`) and
in `plecta/parameters.yaml` (the three `depth_3d` keys). Equivalently, revert
the merge commit that brought the option in:

    git revert -m 1 <merge commit>

`tests/test_parameters.py` will confirm the parameter file and the dataclass
still agree afterwards.
