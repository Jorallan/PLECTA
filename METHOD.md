# Method

How PLECTA turns a binary mask into overlap-aware filament instances, and what
each stage is and is not allowed to use.

This document is **descriptive**: it carries no measurements. Every number
PLECTA has measured is recorded in [`evidence/`](evidence/) as
machine-readable JSON, indexed by [`evidence/README.md`](evidence/README.md).
Every effective parameter value lives in
[`plecta/parameters.yaml`](plecta/parameters.yaml), which is what the
executable actually reads.

## Contract

PLECTA accepts a 2-D binary foreground or centreline mask and returns a sparse
multilabel stack of filament instances. Pixels at a crossing may belong to more
than one instance.

The evaluated grouping stage uses **mask geometry only**. It does not read the
source image, the ground truth, or a clean reference mask. Everything else in
the repository — depth, width measurement, rendering — runs after the grouping
is fixed and cannot revise which pixels were grouped together.

## Graph construction

The mask is binarized, skeletonized, and pruned of short spurs. Adjacent
junction pixels are merged into nodes, so one crossing is one node rather than
a cluster of them.

Short connectors are resolved before matching, on a three-way distinction the
parameters name: the shortest are absorbed outright, intermediate ones join
nearby nodes while remaining real arms in their own right, and free debris
below a threshold is discarded. The remaining connected pieces become ordered
**arms**, each with one **stub** at either end. Stubs are the units the method
has to pair up.

Each stub receives an outward tangent and a signed curvature from a polynomial
fit taken in **arclength** rather than pixel count, so a diagonal arm gets the
same support as a horizontal one. A frame is marked reliable only when enough
samples span enough arclength to justify it; unreliable frames are excluded
from the candidate sets that depend on direction.

## Linking

Several coupled rounds alternate two exact matching problems.

**Crossing matching.** At each junction, maximum-weight matching pairs
compatible stubs from different arms. The cost combines direct tangent
reversal, chord-turn geometry, and curvature continuity. Leaving a stub
unmatched is always allowed, and an edge is eligible only when its cost falls
below a multiple of that unmatched price — so the matcher declines rather than
forcing an implausible pairing.

**Gap matching.** Among the stubs still free, one global matching bridges gaps
in the mask. Candidates are gated before they are priced: both frames reliable,
different arms, different junctions, and distance, tangent angle and chord-turn
angle all within bounds. The cost then also charges for gap length, and
eligibility again requires it to clear a ceiling.

Cycles are broken after each matching step.

The rounds are coupled because **every accepted join lengthens a chain, and a
longer chain estimates direction better**. Later rounds therefore refit geometry
over the assembled chains rather than locally, so accumulated context can revise
links that were ambiguous when only a short arm was visible. A fixed annealing
schedule tightens the gates across rounds and reaches full strictness only in
the last one; the evaluated configuration returns that final round rather than
selecting among several.

Connected components of accepted arm links define the instances. Each instance
contains its arm pixels and every junction cluster it touches — which is what
makes the output overlap-aware, since a crossing pixel is written into *every*
instance running through it. Isolated one-arm fragments below a minimum length
are omitted. Gap links affect grouping but are not painted into the core
output.

## Configuration

[`plecta/parameters.yaml`](plecta/parameters.yaml) materializes every effective
value across all six sections — `grouping_2d`, `depth_3d`, `joint_greyscale`,
`width_sampling`, `width_profile` and `rendering` — and
[`plecta/parameters.py`](plecta/parameters.py) builds each stage's parameter
object from it.

Nothing in the code carries a tuned value of its own. The dataclass defaults
exist only so the classes are constructible in isolation and are overwritten
from the file on every load; the file is required, and a missing or incomplete
section fails loudly rather than falling back. That is what makes "the YAML is
the configuration" a property rather than an aspiration, and
[`tests/test_parameters.py`](tests/test_parameters.py) enforces it in both
directions.

[`plecta/params.json`](plecta/params.json) is the frozen record of the
`grouping_2d` section alone — that section only, not the whole configuration.
`plecta.parameters.verify_against_frozen()` asserts `parameters.yaml` still
agrees with it and the test suite calls that check, so a stray edit to the
grouping section fails the tests instead of silently invalidating published
results. Its SHA-256 is the provenance anchor for [`evidence/`](evidence/).

## What is measured

The main outcome is **common-fragment pairwise F1**. The metric is
[`plecta/score.py`](plecta/score.py); its module docstring gives the fragment
derivation and the pairwise TP/FP/FN formulae, and is the single description of
them. Cutting both prediction and truth into the same atomic fragments and
scoring agreement about which pairs share an instance makes the score
independent of how either side numbered its instances.

Precision, recall, instance recovery, adjusted Rand index, and the
variation-of-information split and merge components are recorded alongside it.

Development and held-out scene sets are drawn from disjoint seeds, and a third
set is locked and left unevaluated. The split, its seeds and its executable
verifier are described in
[`evidence/README.md`](evidence/README.md#evaluation-split); the aggregate
results are in
[`evidence/benchmark_summary.json`](evidence/benchmark_summary.json) and the
per-scene rows behind them in
[`evidence/heldout_per_scene.json`](evidence/heldout_per_scene.json).

## Scope

The evidence supports 2-D overlap-aware reconstruction on synthetic scenes from
the development generator. Held-out scenes are independent draws from that same
generator, so they measure **sampling robustness, not transfer** to an
independent image distribution.

It does not establish calibrated physical widths or general real-image
performance. SEM width measurement and ribbon rendering are downstream optional
stages, and default-off silhouette absorption changes grouping and was absent
from the held-out result.

Two limitations of the optional depth stage are worth stating plainly, since
they are not visible in the aggregates:

- The shipped over/under rule was **checked** on a clean optical set whose
  seeds are disjoint from every tuning set, and the choice of rule, its weight,
  its noise floor and its abstention threshold were all fixed before that set
  existed. **One geometry constant — the evidence core's half-size — was
  selected by a sweep on that same set**, so the default cannot be described as
  pre-frozen. The sweep is flat across the range examined, and the value
  matches what the radius-free fallback already produced, which bounds but does
  not excuse the optimism.
- The stored depth aggregates predate a change to the evidence rule and were
  produced under the previous one. That rule stays selectable and reproduces
  its records exactly, but those figures **need regenerating** before they
  describe the shipped stage.

## Optional stages

These run after grouping is fixed. None of them can change it.

### Depth (2.5-D)

The depth stage finds projected crossings, reads one grayscale channel to
decide which instance lies on top at each, resolves those local calls into one
globally consistent order, infers the fewest layers that order needs, solves a
height per instance under a compact-stack prior, and can sweep circular tubes
to PLY or OBJ.

The cue is **occlusion**: at a projected crossing the shared core shows the
upper rod's appearance uninterrupted, while the lower rod's ridge is
interrupted there. The question asked of the image is how far the core's
appearance sits from each rod's own flanks, normalised so that a crossing whose
rods differ by less than the flank noise cannot report certainty. The shipped
rule reads a **constant-size core**, so no radius enters the evidence path, and
builds no gradient image.

Every rule abstains rather than guesses — too few clean flank samples, or a
score inside the abstention band, and the crossing is reported undecided. An
abstained crossing still means the two instances overlap in projection and must
clear each other in z, so `undecided_order` chooses how that tie is broken:
either by graph colouring with precedence, which minimises the level count, or
by the historical id-rank order. Both arrangements are solved and the thinner
film is kept.

Radii no longer take part in the evidence, but they still decide which pairs
graze, set the clearance every stacked pair must satisfy, and become the
exported tube's radius.

`DepthParams` is the complete list of what the stage may use. The fields, their
defaults, and the measurements behind each choice — including the alternative
scoring rules kept for reproducibility, the grazing-evidence option, and a
known limitation of the global ordering pass — are documented in the field
comments of [`plecta/parameters.yaml`](plecta/parameters.yaml) and in
`plecta/depth.py`. `solver_report` records the evidence
rule, sampling geometry and abstention threshold a run actually used, so a
stored record describes itself rather than depending on the `parameters.yaml`
of the day.

### SEM characterization

[`plecta/image/`](plecta/image/) measures each instance against the SEM image —
width, brightness, and how much to trust either number — and can re-render
instances as smooth ribbons. SEM intensity never affects grouping. See
[`plecta/image/README.md`](plecta/image/README.md).

### Joint mode

`joint_greyscale` configures a mechanism the pipeline does not run: letting a
registered greyscale image influence the 2-D junction matching at the pairs
geometry finds ambiguous. It was built, measured, and **rejected**, and is kept
only as the record of that finding. Nothing imports it, there is no console
script, and the only way to run it is `python -m plecta.joint`. **No 2-D
improvement is claimed for it anywhere.**

## Where the rest is

| If you want | Read |
|:--|:--|
| Machine-readable aggregates and provenance | [`evidence/`](evidence/) |
| How the evaluation split was drawn | [`evidence/README.md`](evidence/README.md#evaluation-split) |
| Every tunable of every stage | [`plecta/parameters.yaml`](plecta/parameters.yaml) |
| Source modules and architecture | [`plecta/README.md`](plecta/README.md) |
