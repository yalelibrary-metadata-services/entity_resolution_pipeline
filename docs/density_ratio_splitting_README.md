# Density-Ratio Cluster Splitting (Anti-Overmerging Second Pass)

`ClusterValidator.prevent_overmerging_density` in `src/cluster_validation.py`
detects and undoes **over-merged clusters**: clusters where transitive closure,
amplifying a handful of false-positive matches, has fused records belonging to
different identities. It finds each cluster's weakest line with a global
minimum-cut algorithm (Stoer–Wagner) and decides whether that line is a seam
between two people or the interior of one person using a conductance-style
**density-ratio test**.

Validated on labeled data ("Schubert, Franz": 89 records, 4 true identities,
all 3,916 pairs scored): the fused 89-record cluster splits into exactly the
4 identities, while the two hardest legitimate structures — a heterogeneous
single-person cluster (the composer's scores vs. streaming videos) and a
thinly-attached true member (Cid Franco's *Bloqueio*) — pass through untouched.

---

## Why a second pass exists

Transitive closure is asymmetric in how it treats classifier errors:

- A **false negative** (missed match) is a broken thread in a net: in a dense
  cluster, dozens of alternative paths keep the records together. The Schubert
  composer absorbed 276 false negatives without losing a single record.
- A **false positive** (wrong match) is a stitch sewing two garments together:
  one edge fuses two entire clusters. 16 false positives out of 3,916 pairs —
  99.6% pairwise accuracy — produced total fusion: one cluster containing all
  four Schubert identities.

A correction therefore does not need to repair missed matches; it needs to find
and cut the few false stitches. That is a graph problem, and this module is the
graph solution.

## The mental model

For each cluster, ask: *if this cluster had to be torn into two pieces along
its weakest line, would the tear look like a seam between two different people,
or like cutting through the middle of one person?*

- **Finding the weakest line** is the minimum-cut computation (Stoer–Wagner).
- **Judging what the tear means** is the density-ratio test.
- If the tear is a seam: cut, and ask the same question of each piece
  (recursion). If not: the cluster is finished.

## Usage

Enable via configuration — `prevent_overmerging` dispatches to the new method:

```yaml
cluster_validation:
  enabled: true
  method: "density_ratio"
  density_ratio_threshold: 0.1     # split when ratio < this (default 0.1)
  density_min_cluster_size: 3      # don't attempt to split smaller clusters
```

Or call directly:

```python
validator = ClusterValidator(config, feature_engineering,
                             match_confidences, hash_lookup)
refined = validator.prevent_overmerging_density(clusters)

# optionally, with an evidence adjudicator consulted before each split:
refined = validator.prevent_overmerging_density(clusters, adjudicator=my_check)
```

`adjudicator(crossing_edges) -> bool` receives the accepted edges spanning a
candidate cut as `(id1, id2, confidence)` tuples and returns `True` to confirm
the split. This is the hook for a name-uniqueness prior, a per-record taxonomy
check, or date-form conflict detection — cheap judges applied only to the few
edges on the seam, never to the whole cluster.

Validation harness: `scripts/weaviate_recovery/validate_density_split.py`
(runs the three labeled cases below and asserts the expected outcomes).

## Data requirements — and the trick that keeps them minimal

The only input is `self.match_confidences`: the pairs the classifier
**accepted**, with probabilities. Rejected pairs are not stored anywhere and
are not needed, because for an *acceptance-rate* computation an absent pair
**is** the rejection — `density()` counts what fraction of all possible pairs
in a group appear in the dictionary. The second pass therefore audits decisions
the pipeline already made: no new vector queries, no re-scoring.

## Algorithm walkthrough

The method (about 60 lines of logic) has one helper and one recursive driver.

### `density(members)`

Fraction of all `C(n, 2)` possible pairs inside the group that were accepted.
By convention a **singleton has density 1.0** — a single record is trivially
coherent with itself. This is deliberately conservative: it makes the
denominator of the ratio large for singleton evictions, so peeling off a lone
intruder (the Schubert liturgist) is *easier* than splitting a group, matching
the observed failure modes.

### `split(members, depth)` — the recursion

1. **Guards.** Clusters below `density_min_cluster_size` return unchanged (a
   2-record cluster is a single accepted edge the classifier already voted
   on). A depth cap of 20 is a runaway backstop; real recursions go 3–4 deep.
2. **Build the accepted-edge graph**, edges weighted by classifier confidence.
3. **Disconnected safety valve.** Closure output is connected by construction,
   but if a caller feeds anything else, disconnected components are free splits
   — take them and recurse into each.
4. **Find the minimum weighted cut** with `networkx.stoer_wagner`, yielding a
   partition `(S, T)` and its weight.
5. **Compute the ratio:**

   ```
   cross_rate = accepted edges crossing the cut / (|S| * |T|)
   ratio      = cross_rate / min(density(S), density(T))
   ```

   The denominator uses the *smaller* side density — conservative again: the
   boundary must look thin relative even to the more fragile side.
6. **Decide.** `ratio < density_ratio_threshold` → consult the adjudicator if
   provided, split, and recurse into both halves. Otherwise stop: the weakest
   line in this cluster is still too strong to be a seam.

Every evaluated cut is logged with its weight, cross-rate, and ratio, so a
production run leaves an auditable trace of splits made and rejected.

## Inside Stoer–Wagner: finding the weakest line without candidates

The algorithm returns the **global minimum weighted cut** — the two-way
partition minimizing total crossing weight — without being given candidate
partitions and without enumerating the `2^(n-1)` possibilities.

It works in *phases*. Each phase runs a **maximum adjacency search**: start
from an arbitrary vertex and repeatedly add the vertex most strongly connected
(by total edge weight) to the set built so far, until all vertices are ordered.
The load-bearing theorem: for the last two vertices in that ordering, *s* and
*t*, **the cut isolating *t* alone is a minimum s–t cut**. Record its weight
(the "cut of the phase"), then **merge** *s* and *t* into one super-vertex
(summing parallel edges) and repeat on the smaller graph. After n−1 phases, the
cheapest cut-of-the-phase seen is provably the global minimum; un-merging the
super-vertex that produced it yields the partition.

Correctness intuition: any global minimum cut either separates some phase's
final *s* from *t* — in which case that phase found it, since it found the
*best* s–t cut — or it does not, in which case merging *s* and *t* was harmless
because they sat on the same side anyway.

Complexity: `O(V·E + V² log V)`, which behaves like `n³` on dense graphs.
Measured with the pure-Python networkx implementation on ~90%-dense synthetic
clusters: 0.28 s at n=89, 2.8 s at n=200, 25 s at n=400, 217 s at n=800.

## Why the cut locates but the ratio decides

Two distinct jobs, split on purpose:

- **The minimum cut is the locator.** A false merge is by construction the
  cheapest place to tear: a few shallow edges against thousands of strong
  ones. Recursive min-cut walks the seams in order of cheapness.
- **The cut *value* is not the judge.** On the full Schubert data the absolute
  margin collapsed: the composer|artist boundary weighs 3.941 while the
  artist's cheapest *legitimate* internal cut weighs 3.954 — a margin of
  0.013. What still separates cleanly is the normalized question: the boundary
  accepts 7 of 486 possible pairs (1.4%) against ~91–100% internal density
  (ratio 0.016), while isolating the weakest legitimate member shows ~19%
  acceptance against ~99% (ratio ~0.19).

### Measured calibration (labeled data)

| structure | ratio | correct action |
|---|---|---|
| artist \| composer boundary (Schubert) | 0.016 | split |
| genealogist \| rest | 0.058 | split |
| liturgist \| rest | 0.073 | split |
| weakest single composer record | ~0.19 | keep |
| thinly-attached true member (Cid's *Bloqueio*, 2 of 4 edges) | 0.5 | keep |
| heterogeneous oeuvre (composer scores vs. streaming videos, 76% cross-acceptance) | 0.8 | keep |

True boundaries sit at ≤ 0.073; the worst legitimate structure at ~0.19; the
typical legitimate ones at 0.5–0.8. The default threshold 0.1 sits in that gap
with margin on both sides. Production graphs are denser than the labeled test
split (all in-block pairs are scored), which widens the gap further.

### Reference trace (fused Schubert cluster, 89 records, 4 identities)

```
density split: 89 -> 1+88  (cut=2.917, cross_rate=0.057, ratio=0.073)   liturgist off
density split: 88 -> 1+87  (cut=2.575, cross_rate=0.046, ratio=0.058)   genealogist off
density split: 87 -> 6+81  (cut=3.941, cross_rate=0.014, ratio=0.016)   artist | composer
inside the composer: best candidate ratio ≈ 0.19  -> stop
```

Result: the four identities exactly. Controls: composer-only (81 records,
1 person) and Cid Franco (5 records, 1 person) pass through unsplit.

## A common question: don't the two intruders stay stuck to each other?

Looking at the cuts diagram (`schubert_cortes_pt.png`), a sharp-eyed reader
will notice a red false-positive edge running **directly between the
genealogist and the liturgist** (p = 0.600), and the dashed cut lines are drawn
as "liturgist vs. everyone" and "genealogist vs. everyone." So doesn't that
direct link survive, leaving the two intruders fused into a 2-record cluster?

**No — and understanding why is a nice way to internalize how the cut works.**

Think of each record as a bead, and each accepted match as a thread tying two
beads together. When the algorithm decides to snip the liturgist bead off the
tangled ball, it isn't choosing *which* of the liturgist's threads to cut — it
is lifting the bead out entirely. Every thread attached to it must be cut,
because every one of them, by definition, runs from the bead to something
outside it. There is no way to remove a bead while leaving one of its threads
intact; a thread with only one end attached holds nothing.

The liturgist bead has exactly five threads:

- four to the artist cluster (p = 0.578, 0.573, 0.573, 0.594), and
- **one to the genealogist (p = 0.600).**

The recorded weight of the first cut is 2.917 — which is precisely the sum of
those five threads (0.578 + 0.573 + 0.573 + 0.594 + 0.600 = 2.918, matching to
rounding). The genealogist link is one of the five. It does not slip through;
it is severed in step 1 along with the rest.

After that snip, the genealogist has only its four threads into the artist
cluster left (0.647 + 0.634 + 0.634 + 0.661 = 2.576 — again exactly the
recorded second cut, 2.575). Step 2 takes those, and both intruders end as
clean singletons. At no point do they exist as a pair of two.

One belt-and-suspenders detail: even in a hypothetical run where the two
intruders *had* been left together as a 2-record cluster, the recursion stops
at clusters below `density_min_cluster_size` (default 3) — a 2-record cluster
is a single accepted edge the classifier already voted on, and the second pass
does not re-litigate individual votes. So the only mechanism that separates
them is the singleton cut taking the whole adjacency — and it does.

## Rejected alternatives (measured failures)

- **Absolute cut-weight threshold** — margin collapsed to 0.013 on the full
  pair set (see above).
- **Coherence percentage** (the pre-existing rule: member must be similar to
  ≥ 40% of its cluster) — on the labeled graph it would evict 80 of the 81
  legitimate composer records while *passing* the liturgist intruder
  (coherence 0.50) in the small mixed cluster.
- **Off-the-shelf community detection** (greedy modularity) — fails in both
  directions at once: splits the composer along the scores/videos line (which
  the ratio says to keep) and leaves a mixed community of 10 records
  (artist + intruders + 2 composer records). Modularity optimizes partition
  balance, not identity boundaries.
- **Flat name-uniqueness prior on all pairs** — kills all 16 false positives
  but fragments the composer into 13 components. It is the right judge in the
  wrong courtroom: as an `adjudicator` on seam edges only, it removes the
  contamination with zero collateral.

## Limitations and scaling

- **Threshold calibrated on one labeled name.** Before enabling in production,
  measure the ratio distribution over additional labeled blocks. The 30×
  canyon suggests slack; slack should be verified, not presumed.
- **Singleton evictions are the tightest case** (worst legitimate ratio ~0.19
  vs. threshold 0.1). Consider requiring adjudicator confirmation whenever a
  candidate split isolates a single record.
- **The ratio is evaluated at the minimum cut only.** False seams are
  simultaneously cheap *and* sparse, so this suffices on the labeled data; a
  cluster with a cheap-but-dense line and an expensive-but-sparse seam could
  stop the recursion early. Evaluating the k cheapest cuts would close this
  theoretical gap.
- **Cubic scaling on dense graphs.** Fine below ~500 records per cluster
  (~1 min worst case). For mega-entity name blocks (thousands of records),
  either sparsify first — Nagamochi–Ibaraki preserves all cuts of weight ≤ k
  using only O(k·n) edges, and the seams of interest weigh < 5 — or switch to
  a candidate-generating heuristic (confidence-threshold sweep with
  union-find, spectral bisection) and let the ratio test judge the candidates:
  an inexact generator can only *miss* a seam, never cause a false split.

## Related material

- Implementation: `src/cluster_validation.py`, `prevent_overmerging_density`
  (dispatch from `prevent_overmerging` when `method: "density_ratio"`).
- Validation: `scripts/weaviate_recovery/validate_density_split.py`.
- Full pairwise scoring that produced the labeled graph:
  `scripts/weaviate_recovery/score_schubert_full.py` →
  `data/output/reports/franco_cid/schubert_full_pair_scores.csv`.
- Graphs and case-study assets: `scripts/weaviate_recovery/charts/` →
  `data/output/reports/franco_cid/` (`schubert_overmerge_full_pt.png`,
  `schubert_cortes_pt.png`, `teste_razao_pt.png`).
- Pedagogical case study (pt-BR, with executive summary):
  `data/output/reports/franco_cid/density_ratio_case_study.html`.
- Algorithm reference: M. Stoer & F. Wagner, "A simple min-cut algorithm,"
  *Journal of the ACM* 44(4), 1997.
