# Method

How PLECTA groups a mask into instances, what was measured, and what was
measured and **rejected**. Sections marked with a result carry the numbers
behind it.

**The method**
&nbsp;&nbsp;[Contract](#contract) ·
[Graph construction](#graph-construction) ·
[Linking](#linking) ·
[Fixed configuration](#fixed-configuration)

**What it scores, and where it applies**
&nbsp;&nbsp;[Evaluation](#evaluation) ·
[Scope](#scope)

**A measured negative result**
&nbsp;&nbsp;[Joint mode](#joint-mode-a-measured-negative-result)

**The optional depth stage**
&nbsp;&nbsp;[Stacking undecided crossings](#stacking-the-crossings-the-evidence-could-not-decide) ·
[Options](#depth-stage-options) ·
[Global ordering can reverse a relation](#global-ordering-can-reverse-a-relation-nothing-contradicts) ·
[Grazing pairs](#grazing-pairs-evidence-not-clearance)

**The over/under evidence**
&nbsp;&nbsp;[Overview](#depth-stage-the-overunder-evidence) ·
[The feature, exactly](#the-feature-exactly) ·
[The sharpness channel](#the-sharpness-channel-and-why-the-shipped-rule-has-none) ·
[The score is the primitive](#the-score-is-the-primitive-not-p) ·
[Three scoring rules](#three-scoring-rules)

## Contract

PLECTA accepts a 2-D binary foreground or centreline mask and returns a sparse
multilabel stack of filament instances. Pixels at a crossing may belong to more
than one instance. The evaluated grouping stage uses mask geometry only.

## Graph construction

The input is binarized at `>0`, skeletonized, and pruned of spurs up to 3 px.
Adjacent junction pixels form nodes. Short connectors are handled before
matching: connectors up to 5 px are absorbed, connectors up to 14 px join nearby
nodes while remaining real arms, and free debris up to 2 px is absorbed.
Remaining connected pieces become ordered arms with one stub at each end.

Each stub receives an outward tangent and signed curvature from an arclength
polynomial fit. A frame is reliable only when at least three samples span at
least 3 px.

## Linking

Eight coupled rounds alternate two exact matching problems:

1. At each junction, maximum-weight matching pairs compatible stubs from
   different arms. Cost combines direct tangent reversal, chord-turn geometry,
   and curvature continuity. Leaving a stub unmatched is allowed; an edge is
   eligible only when its cost is below twice the unmatched price (1.24 at full
   strictness).
2. Among remaining stubs, one global matching bridges gaps. Candidates require
   reliable frames, different arms, different junctions, distance at most 85 px,
   tangent angle at most 0.40 rad, and chord-turn angle at each end at most
   0.28 rad. Cost also includes gap length; eligibility requires cost below
   0.60 at full strictness.

Cycles are broken after each matching step. Later rounds fit geometry over the
assembled chains (55 px support rather than 24 px locally), so longer context
can revise ambiguous local links. The fixed annealing schedule reaches full
strictness only in its last round; therefore the evaluated configuration returns
that round rather than selecting among several full-strictness rounds.

Connected components of accepted arm links define instances. Each instance
contains its arm pixels and every touched junction cluster, which creates the
overlap-aware crossing representation. Isolated one-arm pieces shorter than
6 px are omitted. Gap links affect grouping but are not painted into the core
output.

## Fixed configuration

[`plecta/parameters.yaml`](plecta/parameters.yaml) materializes every effective
value and is what the executable reads. It holds all six sections —
`grouping_2d`, `depth_3d`, `joint_greyscale`, `width_sampling`, `width_profile`
and `rendering` — and `plecta/parameters.py` builds each stage's parameter
object from it (`plecta.parameters.CONFIG_PATH`,
`plecta.predict.load_params`). Nothing in the code carries a tuned value of its
own: the dataclass defaults exist only so the classes are constructible in
isolation, and are overwritten from the file on every load. The file is required;
a missing or incomplete section fails loudly rather than falling back to
defaults.

[`plecta/params.json`](plecta/params.json) is the frozen record of the 26
`grouping_2d` values that produced every published number — that section only,
not the whole configuration. `plecta.parameters.verify_against_frozen()` asserts
that `parameters.yaml` still agrees with it, and the test suite calls that
check, so a stray edit to the grouping section fails the tests instead of
silently invalidating the results. Its SHA-256 is recorded with the aggregate
results in [`evidence/`](evidence/).

## Evaluation

The main outcome is common-fragment pairwise F1. The metric is
[`plecta/score.py`](plecta/score.py); see its module docstring for the fragment
derivation and the pairwise TP/FP/FN formulae, rather than a second description
of them here. Precision, recall, instance recovery, adjusted Rand index, and
variation-of-information split and merge components are also stored.
Development used 84 scenes. The held-out evaluation used 128 disjoint-seed
scenes from the same generator and had no failures:

| Metric | Mean |
|---|---:|
| F1 | 0.8545 |
| Precision | 0.8676 |
| Recall | 0.8430 |
| Recovery | 0.7994 |
| ARI | 0.8508 |
| VI split / merge (bits) | 0.2562 / 0.2165 |

One-at-a-time development ablations identify chord-turn evidence, gap bridging,
and crossing matching as the largest contributors. These are diagnostic, not
held-out comparisons.

## Scope

The evidence supports 2-D overlap-aware reconstruction on synthetic scenes from
the development generator. It does not establish transfer to independent image
distributions, calibrated physical widths, or general real-image performance.
SEM width measurement and ribbon rendering are downstream optional stages;
default-off silhouette absorption changes grouping and was absent from the
held-out result.

## Joint mode: a measured negative result

`joint_greyscale` is the only section of `parameters.yaml` that configures
nothing the pipeline runs. Joint mode lets a registered greyscale image
influence which arm continues into which *inside* the 2-D junction matching, at
the pairs geometry finds ambiguous. It was built and measured on 2026-08-20,
and rejected: mean F1 **−0.0129** as shipped over 12 synthetic development
scenes and 3 real annotated scenes — never positive on any set — and **+0.0005**,
i.e. neutral, after repairing three implementation defects, so the loss was not
merely a bug. On the ambiguous stub pairs it acts on, appearance separates
right from wrong at AUC **0.63** pooled and **~0.53** on real SEM, against
geometry's own **0.78–0.82** on the same pairs; the pre-registered stop
criterion of AUC 0.75 was met. The module is disabled and imported by nothing —
`plecta/predict.py` never touches it, there is no console script, and the only
way to run it is `python -m plecta.joint`. **No 2-D improvement is claimed
here or anywhere.** It is kept as the record of a rejected mechanism so the
finding does not have to be re-derived; see `plecta/joint.py`.

The shipped `lambda_appearance: 1.5` is **deliberate and is not a live
default**. `lambda_appearance: 0.0` would disable the mechanism exactly, but
setting it to 0 would also destroy the record of what was measured: 1.5 is the
weight the −0.0129 result was obtained at, and the file's job here is to record
the rejected configuration, not to ship a safe one. Nothing loads this section
unless someone runs the module by hand, so the value cannot leak into any
result. Anyone re-running the module to confirm the negative should leave it at
1.5; anyone wanting the geometry-only core should not run the module at all.

## Stacking the crossings the evidence could not decide

An abstained crossing still means two instances overlap in projection, so they
must clear each other in z -- but nothing in the evidence says which way round.
`DepthParams.undecided_order` chooses how that tie is broken.

`id_based` is the historical behaviour: the direction comes from the order the
crossing-graph components happen to fall in, which for singleton components is
instance-id rank. It is deterministic and it never contradicts the evidence,
but id rank has no reason to put two instances that do not overlap each other
on the same level. Six filaments in a row, each overlapping only its neighbour,
come out as six levels where two suffice.

`compact` (the default) states the problem properly and solves it as graph
colouring with precedence:

* every crossing pair is a **conflict**: the two may not share a level;
* every **decided** relation is a precedence constraint: the upper instance
  must sit strictly higher;
* minimise the number of levels.

That is the same "minimal K" objective `assign_layers` already applies to the
decided relations, extended to the abstained ones. `compact_order` solves it
with DSATUR -- take the vertex with the most distinct neighbouring levels next,
ties by conflict degree then by id, lowest feasible level each time -- in an
order that respects precedence.

Colouring minimises the level *count*, but what a reader quotes is the film
thickness, and thickness is the longest **weighted** chain. With unequal radii
an arrangement with fewer levels can still be thicker; measured on a real
307-instance annotation, a colouring at the same eight levels came out 113.5 px
against id rank's 111.8 px. So `run_scene` solves both arrangements and keeps
the thinner film. `compact` is therefore never thicker than `id_based`, and
`solver_report.undecided_placement` records which one was used.

Nothing else moves: the per-crossing probabilities, the abstentions, the
corrected over/unders and the layer count K are identical under both. On the
same annotation, thickness falls from 134.9 px to 130.7 px with every decision
unchanged.

## Depth stage options

Everything the depth stage is allowed to use is a field on `DepthParams`.
Every default below preserves the behaviour the stage had before the option
existed, except `undecided_order` and the evidence rule itself: `scoring`,
`core_mode` and `core_px` changed together on 2026-08-22, and the previous
behaviour is reproduced by naming all three (see "Three scoring rules").

| field | default | what it decides |
|---|---|---|
| `use_image_evidence` | `True` | whether to read the image at all. Off, or with `image=None`, every crossing abstains: the geometry still solves and nothing that is constrained interpenetrates, but the vertical order becomes a deterministic tie-break rather than a measurement. `solver_report.image_evidence` records which happened. |
| `radius_mode` | `"measured"` | `"measured"` uses the radii the caller supplied (or the FWHM widths); `"fixed"` gives every instance `default_radius_px`, which is what a mask-only run has to do. |
| `scoring` | `"noise_floored"` | how the evidence becomes one signed score. The shipped rule is `2·F_inf`, a **single** channel with a noise floor under its denominator; `"winsorized_linear"` (`2·F_I + clip(F_S, −1, 1)`) is the rule shipped before 2026-08-22 and `"weight_of_evidence"` (w = 2.446, 1.576) is the smooth-bounded variant the stored held-out aggregates were produced under. All three share a numerator, so they never disagree about **direction**. See "Three scoring rules" below. |
| `core_mode` | `"flat"` | how the evidence core's half-size is set. `"flat"` makes it the constant `core_px`, so **no radius enters the evidence path at all**; `"radius_scaled"` is the historical `max(core_px, 1.1·(r_i+r_j)/2 + 2)`, retained so stored records reproduce. Radii are still used everywhere else — metric z, grazing detection, the exported tube. |
| `core_px` | `7.5` | that half-size in px under `"flat"`, and the floor of the formula under `"radius_scaled"` — so reproducing a pre-2026-08-22 record needs `core_px = 6.0` as well as `core_mode = "radius_scaled"`. |
| `radius_fallback` | `"median"` | what an instance with **no usable width measurement** gets. `"median"` gives it the median of whatever *was* measured in the same scene; `"default"` gives it `default_radius_px`, a constant unrelated to the scene. `solver_report.n_radius_imputed` and `radius_fallback_px` record how many were filled and with what, and each such instance carries `"d_imputed": true`. |
| `clear_grazing_overlaps` | `False` | whether pairs that overlap in projection *without* their centrelines crossing are cleared as well. Off keeps the historical constraint set — crossings only — and reports the interpenetration that leaves. |
| `grazing_evidence` | `False` | whether to put the occlusion question to those pairs instead of merely clearing them, at their point of closest approach, so each earns a direction or abstains on its own merits. See "Grazing pairs: evidence, not clearance" below. |
| `undecided_order` | `"compact"` | how pairs the evidence could not decide are stacked; see the section above. |
| `z_gap_px` | `0.0` | clearance added to `r_i + r_j`. |
| `abstain_band` | `0.15` | **legacy rules only.** `\|p - 0.5\|` below this is called unidentifiable, applied as the exactly equivalent score threshold 0.6190. Kept so `winsorized_linear` and `weight_of_evidence` stay bit-identical to their records. |
| `abstain_score` | `0.40` | **`noise_floored` only**, and in score units rather than as a probability band, because `\|s\| = \|2·F_inf\|` is a noise-normalised margin and not a log-odds. 0.40 was set to put the rule at the 82.5% **pooled** decision rate the calibration targets — a target carried in from the mixed development domains, and *not* what the threshold delivers on any one set: on the clean held-out optical set the same 0.40 decides **87.1%** of crossings. Every threshold in 0.35–0.47 measures the same accuracy there, so it is a plateau, not a peak. See "Three scoring rules" for how the 82.5% comparison row and the shipped 87.1% relate. |

`solver_report` now also carries `image_evidence`, `scoring`,
`abstain_threshold` (the effective threshold in score units, whichever
parameterisation supplied it), `core_mode`, `core_px`, `radius_mode`,
`radius_fallback`, `n_radius_imputed`, `radius_fallback_px`,
`n_grazing_overlaps`, `grazing_constrained`, `grazing_evidence`,
`n_grazing_decided`, `n_interpenetrating`, `undecided_order` and
`undecided_placement` — so a stored run describes the evidence rule and the
sampling geometry that produced it, rather than depending on the
`parameters.yaml` of the day.

A radius is still not a cosmetic detail: it decides which pairs graze, sets the
clearance `r_i + r_j` every stacked pair must satisfy, and becomes the exported
tube's radius. Falling through to a constant is therefore worth avoiding where
anything at all is measurable — on a real 282-instance field, 56 instances had
no usable width and were modelled at 5.0 px while every rod that *could* be
measured came out between 6.3 and 16.8 px, i.e. thinner than anything actually
present in the scene.

What it no longer does is take part in the **evidence**. It used to size the
evidence core and window and place the edge samples the sharpness channel read;
the shipped rule has no sharpness channel and a constant core, so an over/under
call now depends on no radius at all. That matters because the radius the old
geometry wanted is an oracle quantity on synthetic data and an estimate
everywhere else — and dropping it costs nothing measurable (see "Three scoring
rules").

`n_interpenetrating` comes from `clearance_violations`, which is measured on the
**solved heights** and over **both** pair sets whatever was constrained — so
leaving grazing pairs free reports the geometry that costs, instead of hiding
it. On a 307-instance annotation: 414 crossings and a further 445 grazing pairs,
of which 149 interpenetrate when they are left free.

Command line: `--no-image-evidence`, `--clear-grazing`,
`--grazing-evidence`, `--fixed-radius R`,
`--undecided-order {compact,id_based}`.

### Global ordering can reverse a relation nothing contradicts

`resolve_global_order` states its objective as a minimum-weight feedback arc
set, and a minimum FAS never has to reverse an edge that lies on no cycle. The
pass does not have that property: it rebuilds a whole connected component's
total order at once and re-derives every relation from the single order that
comes out, so a relation nothing contradicts can still come back reversed. Fed
every ground-truth crossing with its true direction — input that is acyclic by
construction, and that a minimum-FAS solver should therefore leave untouched —
it still reverses 24 relations across the 20 held-out scenes, capping layer ARI
at 0.886 instead of 1.000. On real evidence, 37 of the 102 relations it
reverses lie on no cycle at all.

A condense-then-order variant — topologically order the condensation of the
strongly connected components, and run the reordering only inside genuine
cycles — reverses none of those and reaches ARI 1.000 at that ceiling, but
measured worse on the actual noisy predictions: Kendall tau −0.024, mean K
5.80 → 7.45 against a true 5.95, depth RMSE 18.95 → 23.80 px. The broader
flipping incidentally suppresses some erroneous local relations: it makes 60
more reversals than the condensing variant and 34 of those 60 land correct,
57%, a coin toss that happened to come up heads on this set. That advantage is
an accident rather than a mechanism and is expected to invert as the local
evidence improves. It was implemented and removed rather than kept switchable;
the implementation is in git history at commit 7d369d8.

### Grazing pairs: evidence, not clearance

`clear_grazing_overlaps` separates the pairs that overlap without
crossing, but with an arbitrary direction, which measures worse than
leaving them alone. `grazing_evidence` instead asks the image about them:
`crossing_evidence` at the point of closest approach, where the two
strokes really do overlap even though the centrelines never meet. A pair
that earns a direction then constrains and is reported as a crossing (with
`"grazing": true`); a pair that abstains is left exactly as before.
`solver_report.n_grazing_decided` counts the first kind.

**Every number in the rest of this subsection was measured under the previous
evidence rule** — `winsorized_linear` / `weight_of_evidence` with the
radius-scaled core (`core_px` = 6.0), on 2026-08-21, before the 2026-08-22
change. They have **not** been re-run under `noise_floored`. The recipe under
"Reproducing the previous behaviour" reproduces the rule they were made with
exactly, so they still describe *a* configuration this code can run, but they
are not the shipped stage's numbers. The direction of the conclusion is not in
doubt — all three rules share a numerator and never disagree about direction —
but the accuracies, coverages, tau and ARI below are **pending re-measurement**
and should be quoted with this attribution attached.

The sampling geometry is sound only where the pair is a real crossing that
2-D fragmentation broke apart. With PLECTA's own instances, 196 of 484
grazing pairs across the held-out scenes are exactly that, and on those
the evidence is right 0.716 of the time (0.848 on crossings that stayed
intact). On the other 288 — genuine side-by-side runs and end abutments,
which have no unoccluded flank near the contact to compare the overlap
against — it is right 0.537 of the time, and the abstention band does not
separate the two populations: it decides about 69% of both. So the switch
buys crossing coverage 0.766 → 0.849, accuracy over all eligible crossings
0.543 → 0.607 and interpenetrating pairs 207 → 41, while costing Kendall
tau 0.354 → 0.340 and layer ARI 0.256 → 0.228. Off by default; worth
turning on for predicted-instance runs that care about crossing coverage
and physical validity more than about the layer partition.

## Depth stage: the over/under evidence

The cue is occlusion. At a projected crossing the shared core shows the *upper*
rod's appearance uninterrupted, while the lower rod's ridge is interrupted
there. The question asked is "how far is the core from this rod's own flanks",
and the shipped rule asks it of **one** channel, the grayscale.

Sampling geometry is flat and reads no radius: the core half-size is the
constant `core_px` = 7.5 px and the window `max(22, 3·core)`, so flank samples
sit clear of the other rod's stroke without anyone having to know how wide
either rod is. It used to be `max(6, 1.1·(r_i+r_j)/2 + 2)` px — see
`core_mode`, and "A flat core" below for what dropping the radius measures at.

### The feature, exactly

Three medians are measured per crossing — each rod's own grayscale appearance
on its flanks, and the appearance inside the shared core:

| | grayscale |
|---|---|
| rod *i* flanks | `m_i` |
| rod *j* flanks | `m_j` |
| core | `m_c` (both rods' core samples pooled) |

The numerator asks which rod's appearance the core matches, and is shared by
every scoring rule in this file — which is why they never disagree about
*direction*, only about confidence:

```
numerator = |m_c − m_j| − |m_c − m_i|          positive favours rod i
```

It is how much closer the core sits to rod *i*'s brightness than to rod *j*'s.
The denominator is what the rules differ on:

```
F_I    = numerator / max(|m_i − m_j|, 0.02)                    previous rule
F_inf  = numerator / max(|m_i − m_j|, 2·sigma_hat)              shipped rule
```

`F_I` divides by how different the two rods look in the first place. That
normalisation is what makes it scale-free, and it is why two rods of the *same*
brightness cannot produce a strong reading: the numerator shrinks with the
denominator. But it is also **bounded to [−1, +1]** by construction and reaches
±1 whenever the core matches one rod outright — including when the two rods
differ by less than the noise, where the match means nothing.

How often that happens has been measured twice, on **different populations**,
and the two figures must not be read as two estimates of one quantity:

| population | saturated at ±1 |
|---|---:|
| the reference annotation — one real SEM field, 307 instances, 414 crossings; the same scene every other "reference annotation" number in this file is measured on | 285 of 414, **68.8%** |
| the rule-change audit, pooled over crossings by domain (recorded outside this repository; the scene list is not in this repo) | **32.3%** of synthetic, **61.5%** of real |

68.8% is one field; 61.5% is a pooled real-domain rate over a larger and
differently composed set that this repository does not enumerate. They are
consistent in what they say — saturation is the majority behaviour on real
crossings — and are not comparable as measurements.

`F_inf` puts a floor under that denominator at twice the flank noise, so the
feature reports a margin in units of "how much bigger than the noise is this"
and stops claiming certainty the picture cannot supply. Above the floor the two
are identical.

```
sigma_hat = 1.4826 · MAD of the WITHIN-ROD flank residuals
```

i.e. centre each rod's flank grayscale samples on **that rod's own** median,
pool the two residual sets, take the MAD. The ordering matters and getting it
wrong is silent: the MAD of the raw **pooled flank values** is taken over a
bimodal set — two rods, two brightnesses — so it tracks `|m_i − m_j|` and
becomes a *contrast* floor rather than a noise floor, which exactly cancels the
effect the floor exists to produce. Both are "a MAD of the flank samples"; only
one is a noise scale. `tests/test_depth.py` pins the difference on a hand-built
case where the two are a factor of 30 apart.

Both `sigma_hat` and `feat_noise_floored` are written into every crossing's
`features`, alongside `feat_intensity`, so a stored record can be re-read under
either rule without the image.

### The sharpness channel, and why the shipped rule has none

The two legacy rules add a second channel read off the gradient-magnitude
image, `gradient_magnitude` (a 1 px Gaussian, then Sobels). Its samples are
taken at `centreline ± r·normal` — a ridge's gradient is ~0 on its own axis by
symmetry, so sharpness has to be measured on the edges, which is the last thing
in the evidence path that needed a radius.

**Sharpness** asks whose edges survived the core:

```
cont_k          = | core_edge_k − edge_k |            per rod: how much its own
                                                      edge signal changed
feat_sharpness  = ( cont_j − cont_i ) / max(edge_i + edge_j, 0.02)
```

A small `cont` means that rod's edges continued through the crossing, which is
what being on top looks like. Positive favours *i*, negative favours *j*.

It is **not** in the shipped rule. Measured on the clean held-out optical set
(below), an axial variant of it is inert at every deployable operating point,
and the edge variant costs a per-scene Gaussian plus two Sobels that only it
consumes. Under `scoring = "noise_floored"` the gradient image is therefore not
built at all, and `feat_sharpness` / `d_sharpness` are **absent** from a
crossing's `features` rather than reported as zeros nobody measured.

Under the legacy rules both channels are squashed by `tanh` before weighting,
so no single saturated feature can run away with the decision:

```
z = 2.446·tanh(feat_intensity) + 1.576·tanh(feat_sharpness)
p = sigmoid(z)
```

A worked crossing from the reference annotation (rods 22 × 327), under that
rule:

```
feat_intensity  +1.0000  → tanh +0.7616 × 2.446 = +1.8629
feat_sharpness  +0.5592  → tanh +0.5074 × 1.576 = +0.7997
                                             z  = +2.6625
                                   p = sigmoid(z) = 0.9348
```

and the weight that crossing carries into the ordering is `|z| = 2.6625`. That
is why weights are log-odds and not probabilities: log-odds are additive (adding
them multiplies odds), so summing them over relations is a meaningful quantity,
whereas summing probabilities is not.

Every rule abstains rather than guesses: too few clean flank samples, or a
score inside the abstention band, and the crossing is reported as undecided.

### The score is the primitive, not `p`

`sign(z)` decides and `|z|` weights; nothing consumes `p`. It used to be formed
as `sigmoid(z)` and then undone as `|logit(p)|` in the solver, a round trip that
cancelled exactly. `p` is still reported, because the output schema and the
evaluator's calibration metric expect it, but it is a **monotone display
transform of the score, not a calibrated probability** — held-out ECE is ~0.21,
so it ranks well and should not be read as a confidence.

Abstention is parameterised per rule, because the two families do not mean
the same thing by their score and one number cannot honestly serve both.
`abstain_threshold()` remains the documented band → score converter for the
legacy rules: their band is stated as `|p − 0.5| < 0.15` and applied as the
exactly equivalent score threshold `|z| < log((0.5+band)/(0.5−band)) = 0.6190`,
which keeps them bit-identical to their records. The shipped rule has no
probability anywhere in it — `|s| = |2·F_inf|` is a noise-normalised margin,
not a log-odds — so it takes `abstain_score = 0.40` directly in score units.
`decision_threshold(params)` returns whichever applies, `apply_abstention`
takes `params=` and works for all three, and `solver_report.abstain_threshold`
records the number actually used.

### Three scoring rules

`DepthParams.scoring` chooses how the evidence becomes one score. All three are
odd in their features and carry no intercept, so swapping *i* and *j* flips the
score exactly, and all three share the numerator, so they never disagree about
which rod is on top — only about how sure they are.

| rule | score | weights | abstains at |
|---|---|---|---|
| `noise_floored` (default) | `w_I·F_inf` | 2.0 | `abstain_score` = 0.40 |
| `winsorized_linear` | `w_I·F_I + w_S·clip(F_S, −1, 1)` | 2.0, 1.0 | `abstain_band` = 0.15 → 0.6190 |
| `weight_of_evidence` | `w_I·tanh(F_I) + w_S·tanh(F_S)` | 2.446, 1.576 | `abstain_band` = 0.15 → 0.6190 |

**What the shipped rule measures, and what its seeds are clean for.** A clean
held-out optical set was generated for this question: 20 scenes, 1355
evidence-bearing crossings, seeds disjoint from every tuning set and from every
burned or reserved series. What that buys, stated precisely rather than as a
blanket claim:

* the **rule** is clean on it. The choice of `noise_floored` over
  `winsorized_linear`, the single weight `w_I = 2`, the `2·sigma_hat` floor and
  the abstention threshold `abstain_score = 0.40` were all fixed *before* this
  set existed and were only *checked* against it.
* **`core_px` = 7.5 is not.** It was selected by a sweep run on these very
  scenes — see "A flat core" below, which says so explicitly. It is a
  held-out-selected constant, not a pre-frozen one.

So the comparison below is leakage-free for the choice of *rule*, and carries
whatever optimism one constant chosen on it is worth — bounded, in this case,
by the sweep being flat and by 7.5 being the value the radius-free fallback
already produced. Scene-cluster bootstrap, 4000 resamples, against
`winsorized_linear`:

| comparison | `noise_floored` − previous | 95% CI |
|---|---:|---|
| accuracy at the previous rule's own decision rate (89.5%) | +0.0025 | [−0.0042, +0.0079] |
| AUC of correctness against `\|s\|` (0.7772 → 0.8698) | **+0.0926** | [+0.0563, +0.1321] |
| accuracy at 85.0% coverage | +0.0130 | [+0.0041, +0.0203] |
| accuracy at 82.5% coverage | **+0.0188** | [+0.0080, +0.0254] |
| accuracy at 80.0% coverage | **+0.0240** | [+0.0114, +0.0299] |

Read that first row carefully. **At the previous rule's own operating point the
change is worth nothing** — the CI straddles zero. What changed is the
*ordering*: the rule knows which of its own calls to distrust, and that
converts into accuracy only once you actually abstain. The honest headline is
"same accuracy, better-ordered confidence, simpler evidence path", not "more
accurate".

**Where the shipped default actually runs, and how the coverage rows relate to
it.** The last three rows are *coverage-matched comparisons* — both rules
forced to the stated decision rate — and none of them is the shipped operating
point:

| | decision rate | accuracy |
|---|---:|---:|
| shipped default, `abstain_score` = 0.40, on this set | **87.1%** | 0.9347 |
| previous rule at its own threshold, on this set | 89.5% | 0.9217 |
| the 85.0 / 82.5 / 80.0% rows above | forced, below the default | — |

So run as a drop-in at its own threshold the shipped rule reads 0.9347 at 87.1%
against 0.9217 at 89.5%: +0.0131 [+0.0074, +0.0195] of accuracy bought with
2.4 points of decision rate, and both halves of that trade are real. Reaching
82.5% or 80.0% means abstaining **more** than the default does — a higher
threshold than 0.40 — so "+0.0188 at 82.5% coverage" is what the rule is worth
to a caller willing to abstain that much, not a description of the default.
The 82.5% figure itself is the *pooled* decision rate the threshold was
calibrated to across the mixed development domains; that pooled calibration is
recorded outside this repository, and on this clean set 0.40 decides 87.1%
instead. The two numbers are a target and a delivery on a different
distribution, not a disagreement.

**A second channel was built and measured inert.** An axial variant — each
rod's core arc read on its own axis rather than its edges — scores −0.0008
[−0.0059, +0.0048] against the single-channel rule at the shipped operating
point, and −0.0072 [−0.0258, +0.0142] on AUC. It reverses the direction of 26
of the 1355 crossings and is right on 20 of them, but the largest `|s|` among
those 26 is 0.3474 — **none** of them clears any deployable threshold, and
among the crossings both rules decide the number of reversals is zero. It earns
its place only at full coverage with no abstention at all (+0.0103 [+0.0015,
+0.0196]), and it costs a second pass over both rods' cores. It is deliberately
not implemented.

**A flat core, and where 7.5 px comes from.** The only place a radius entered
the intensity path was the core half-size. Replacing it with a constant and
sweeping 6–14 px **on the clean held-out set itself**: a flat core is never
significantly worse than *oracle* per-rod radii anywhere in that range. 7.5 px
matches the oracle's decision rate (88.0% against 87.5%) and edges its accuracy
(+0.0017 at matched coverage); 9.0 px is indistinguishable. `core_px = 10.0`,
chosen in earlier work on the three *tuning* domains, does **not** replicate
(−0.0051 [−0.0141, +0.0050] against the oracle, at 1.8 points less coverage).

That sweep ran on the same scenes the comparison table above is measured on, so
**this one constant was selected on held-out data** and the default cannot be
described as pre-frozen. Two things bound how much that is worth worrying
about, and neither is a defence of the procedure:

* the sweep is **flat** — nothing in 6–14 px is significantly worse than the
  oracle, and 9.0 px is indistinguishable from 7.5 — so the selection is
  picking a point on a plateau rather than a peak;
* 7.5 px is **exactly** what the radius-free fallback `max(6.0, 1.1·5 + 2)`
  already computed from `default_radius_px = 5.0`. The value pre-existed the
  sweep; what the sweep did was confirm it and retire the non-replicating 10.0.

The value was therefore not invented by the sweep, but it *was* confirmed on
the clean set, and the 6–14 px range examined was too. Read the held-out table
as clean for the rule and one-constant-optimistic for the geometry.

**Reproducing the previous behaviour.** Three settings moved together, so
three have to be named — `abstain_band` did not move and still gates the rule
you are selecting:

```
scoring: "winsorized_linear"   core_mode: "radius_scaled"   core_px: 6.0
```

`core_px` is the *floor* of the radius-scaled core, so leaving it at the new
7.5 would change the sampling. With those three set, the stage reproduces the
previous output bit for bit — every score, probability, abstention, direction,
flip, layer and height — verified on held-out optical scenes; the record only
gains the two new informational feature keys (`sigma_hat`,
`feat_noise_floored`) and the three new `solver_report` fields. Use
`weight_of_evidence` with w = (2.446, 1.576) and the same three geometry
settings for the earlier record.

**The stored held-out depth aggregates predate this change and need
regenerating.** Every depth aggregate — accuracy, coverage, Kendall tau, layer
ARI, mean K, depth RMSE, the crossing-level calibration — recorded outside this
repository **and also the ones quoted inside it**, namely the grazing-evidence
figures in "Grazing pairs: evidence, not clearance" above (196 of 484,
0.716 / 0.848 / 0.537, coverage 0.766 → 0.849, 207 → 41 interpenetrating,
tau 0.354 → 0.340, ARI 0.256 → 0.228) and their duplicate in the
`grazing_evidence` comment in `parameters.yaml`, was produced under
`weight_of_evidence` or `winsorized_linear` with the radius-scaled core. Those numbers still describe
*a* rule this code can still run, and the recipe above reproduces them exactly,
but they are **not** the shipped stage's numbers any more and must not be
quoted as such until they are re-run. The clean held-out optical table above is
the only measurement of the shipped rule on unseen data that currently exists
— leakage-free for the rule, with the `core_px` caveat above attached.
Likewise superseded: the earlier "+0.0137 accuracy on optical" figure, which
compared two thresholds at different decision rates on rows that had chosen one
of those thresholds; at matched coverage on unseen optical data the gain is
+0.0025 [−0.0042, +0.0079].

**Two properties of the legacy rules, kept for the record.** `winsorized_linear`
drops both squashes: `F_I` is already bounded to [−1, +1] by construction, so a
squash on it is redundant, and `F_S` — the unbounded one — is bounded just as
effectively by clipping. Measured on **2649 held-out crossings** across the
in-domain and shifted sets, those two rules change the direction of **no**
crossing; the winsorized one decides about 3% fewer of them and is 0.4–0.6
points more accurate on the ones it does decide (in-domain 0.8479 against
0.8424; shifted 0.7852 against 0.7811).

And a consequence of clipping: because `|clip(F_S)| ≤ 1`, the decision is
locked whenever `2·|F_I| − 1 > 0.619`, i.e. `|F_I| > 0.81` — the sharpness
channel then cannot change the direction or force an abstention. That covers
**52.9%** of in-domain and 45.0% of shifted crossings. Under the smooth variant
a strongly opposing sharpness reading could still veto a saturated intensity
reading (score +0.288 → abstain, against +1.000 → decided here). On the
held-out sets sharpness changed **no** direction either way (0 of 2649) and
moved the decide/abstain gate on 3–4% of crossings; in-domain it costs 0.002
accuracy, on the shifted set it buys 1.3 points of coverage. It was a
distribution-shift backstop, not a load-bearing channel — which is the finding
the shipped rule acts on by dropping it.
