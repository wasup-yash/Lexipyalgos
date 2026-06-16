# convexipy

**A production-ready Python library for abstract convexity spaces — validated axioms, convex hull operators, N-ary classification, exhaustive enumeration on finite carriers, and binomial-transform identities between general and grounded structures.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 198 passing](https://img.shields.io/badge/tests-198%20passing-brightgreen.svg)]()
[![Zero dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

---

## What we are building — and for whom

**convexipy is a pure-Python engine for abstract convexity spaces on finite sets, built for developers and quantitative teams who need algebraically rigorous closure-system reasoning as a software component.**

In concrete terms: a *convexity space* is a pair (X, G) where X is a finite ground set and G is a family of subsets closed under arbitrary intersection and containing X. This structure appears — often implicitly — across formal concept analysis, access-control hierarchy modelling, customer-segment consistency checking, graph-based influence propagation, and combinatorial research on preorders and finite topologies. `convexipy` exposes every fundamental operation on these objects (hull computation, N-arity testing, grounding/de-grounding transforms, full enumeration) as a clean, well-typed, dependency-free Python API, underpinned by the mathematical results of Dulliev & Naumikhin (2026) and verified against their published Tables 1 & 2 for all N=0..4, n=0..5.

The intended users are:

- **Research engineers** building on abstract convexity, formal concept analysis, or closure-system theory who need a correct, tested reference implementation with OEIS cross-references baked in.
- **SaaS / data-platform teams** who need to validate that a family of "rules", "segments", or "configurations" actually composes consistently under intersection — with structured, CI-ready diagnostic reports.
- **Graph-analytics and ML teams** applying halfspace-separation or betweenness-convexity concepts to finite networks or feature spaces.

---

## Installation

```bash
pip install convexipy
```

Or from source:

```bash
git clone https://github.com/example-org/convexipy.git
cd convexipy
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9. No third-party runtime dependencies — stdlib only (`itertools`, `math`, `json`, `dataclasses`, `typing`).

---

## Quick start

```python
from convexipy import ConvexitySpace

X = {0, 1, 2}
G = [{0, 1, 2}, {0, 1}, {0}, set()]
space = ConvexitySpace(X, G)

space.convex_hull({1})     # frozenset({0, 1})
space.convex_hull({2})     # frozenset({0, 1, 2})
space.is_grounded()        # True
space.arity()              # 1  (this is a binary / 1-ary convexity)

# Grounding reflection (Lemmas 1-5, Dulliev & Naumikhin 2026)
non_grounded = ConvexitySpace({0,1,2}, [{0,1,2}, {0,1}, {0}])
H = non_grounded.get_grounded_reflection()
H.ground_set               # frozenset({1, 2})
H.is_grounded()            # True

# Reconstruct the original from H and the minimal set C = g(∅) = {0}
rebuilt = ConvexitySpace.from_grounded(H, {0})
rebuilt == non_grounded    # True
```

---

## Use cases

### 1 — Access-control hierarchy validation (preorder recovery)

A grounded 1-ary convexity on a set of roles is exactly the family of upward-closed sets (up-sets) of a preorder — the mathematical content behind OEIS **A000798** (quasi-orders, finite topologies). `convexipy` turns a proposed list of "reachable-state sets" into a validated hierarchy and extracts the underlying `≤` relation.

```python
from convexipy.applications import preorder_from_relation, upset_closure

result = preorder_from_relation(
    {"admin", "editor", "viewer"},
    [
        {"admin", "editor", "viewer"},
        {"editor", "viewer"},
        {"viewer"},
        set(),
    ],
)
result.is_consistent          # True
result.leq                    # {('admin','editor'), ('admin','viewer'), ('editor','viewer'), ...}

# What can "editor" reach?
upset_closure(result.space, {"editor"})   # frozenset({'editor', 'viewer'})
```

If the input sets are not intersection-closed, `preorder_from_relation` **repairs** them (computes the closure) and sets `is_consistent = False`, so you can tell the difference between a self-consistent rule table and one that silently had gaps.

**Research backing:** OEIS A000798 — [https://oeis.org/A000798](https://oeis.org/A000798)

---

### 2 — Customer-segment / feature-space convexity (formal concept analysis)

Formal concept analysis represents knowledge as intersection-closed families of "concept extents". `convexipy` validates and manipulates such families, providing the convex hull (smallest concept containing an arbitrary group), halfspace separability queries (the abstract analogue of linear separability), and N-arity checks (whether segment membership is determined by small sub-groups — the "finite induction" property).

```python
from convexipy.applications import feature_space_convexity, is_separable

users = {"alice", "bob", "carol", "dave"}
segments = [
    {"alice", "bob", "carol"},   # "premium" segment
    {"bob", "carol", "dave"},    # "active"  segment
]
space = feature_space_convexity(users, segments)  # auto-closes intersections

# Smallest official segment containing bob
space.convex_hull({"bob"})        # frozenset({'bob', 'carol'})

# Are alice and dave halfspace-separable in this convexity?
is_separable(space, {"alice"}, {"dave"})   # False (their hulls overlap)
```

**Research backing:** Seiffarth, Horváth & Wrobel (2021). *Maximal Closed Set and Half-Space Separations in Finite Closure Systems.* arXiv:2001.04417

---

### 3 — Graph / network convexity (monophonic halfspace learning)

Given an undirected graph, `convexipy` builds the convexity whose convex sets are vertex-induced intervals (the union of all simple-path vertex sets between each pair), enabling graph-betweenness hull queries and influence-propagation boundary analysis.

```python
from convexipy.applications import graph_convexity_from_paths

# Linear path: 0 — 1 — 2 — 3
g = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
space = graph_convexity_from_paths(g)

# Convex hull of {0, 3} is the entire path
space.convex_hull({0, 3})    # frozenset({0, 1, 2, 3})
# Convex hull of {0, 2}
space.convex_hull({0, 2})    # frozenset({0, 1, 2})
```

**Research backing:** Bressan, Chepoi, Esposito & Thiessen (2025). *Efficient Algorithms for Learning and Compressing Monophonic Halfspaces in Graphs.* arXiv:2506.23186

---

### 4 — Configuration-space / rule-table validation (CI integration)

Product rule tables (allowed feature-flag × plan-tier × region combinations) implicitly assume that intersecting any two valid configurations yields another valid configuration. `convexipy` tests this assumption and returns a structured `ConfigurationValidationReport` detailing every missing intersection — suitable for a CI check on rule-table pull requests.

```python
from convexipy.applications import validate_configuration_family

universe = {"basic", "pro", "enterprise", "addon_a", "addon_b"}
bundles = [
    {"basic", "addon_a"},
    {"pro", "addon_a", "addon_b"},
]
report = validate_configuration_family(universe, bundles)

report.is_valid              # False
print(report.summary())
# Configuration family is NOT closed under intersection:
#   - The universe set X was not present among the inputs.
#   - Intersection of {'basic','addon_a'} and {'pro','addon_a','addon_b'}
#     is {'addon_a'}, which is not among the declared configurations.

# Always returns the repaired (intersection-closed) space too
report.space.is_valid()      # True
```

---

### 5 — Combinatorial enumeration and OEIS research

Enumerate every N-ary convexity on a finite carrier, count grounded/non-grounded breakdowns, and verify the central binomial-transform identity from Dulliev & Naumikhin (2026) connecting total and grounded counts.

```python
from convexipy import (
    count_convexities,
    count_grounded_convexities,
    enumerate_and_classify,
    binomial_transform,
    is_binomial_transform_pair,
)
from convexipy.oeis import lookup_total, lookup_grounded, OEIS_REFERENCES, oeis_url

# Tables 1 & 2 of the paper verified for all N=0..4, n=0..4
count_convexities(4, n_ary=1)          # 500  (Table 1, N=1, n=4)
count_grounded_convexities(4, n_ary=1) # 355  (Table 2, N=1, n=4 — OEIS A000798)

# The central result: total = binomial_transform(grounded)
grounded = [count_grounded_convexities(k, n_ary=1) for k in range(5)]
total    = [count_convexities(k, n_ary=1) for k in range(5)]
is_binomial_transform_pair(grounded, total)  # True

# Look up OEIS cross-references
OEIS_REFERENCES[(1, True)]        # 'A000798'
oeis_url("A000798")               # 'https://oeis.org/A000798'
lookup_total(6, 1)                # 257151  (Table 1, N=1, n=6)

# Stream every binary (1-ary) grounded convexity on 3 points
result = enumerate_and_classify(3, n_ary=1)
result.total           # 45
result.grounded_total  # 29
result.verify_binomial_identity([1, 1, 4, 29])  # True
```

---

## Core API reference

### `ConvexitySpace`

| Method | Description |
|--------|-------------|
| `ConvexitySpace(X, G, validate=True)` | Construct and validate. Raises `InvalidConvexityError` on axiom violation. |
| `.convex_hull(A)` | `g(A)` — smallest convex set containing `A`. Memoised. |
| `.is_convex(A)` | `True` iff `A ∈ G`. |
| `.is_grounded()` | `True` iff `∅ ∈ G`. |
| `.minimal_convex_set()` | `g(∅)` — the unique minimum of `G`. |
| `.is_n_ary(n)` | Check the N-ary closure condition for integer `n ≥ 0`. |
| `.arity()` | Return the minimal `N` for which `is_n_ary(N)` is `True`. |
| `.get_grounded_reflection()` | Build `H` on `Y = X \ g(∅)` (Lemma 2, paper). |
| `ConvexitySpace.from_grounded(H, C)` | Inverse: build `G` on `Y ∪ C` from grounded `H` (Lemma 5). |
| `.restrict_to(subset)` | Induced subspace convexity on `subset ∩ X`. |
| `.to_json() / .from_json()` | Serialise/deserialise to JSON string. |
| `.to_dict() / .from_dict()` | Dict-based serialisation. |
| `ConvexitySpace.discrete(X)` | Full power-set `P(X)`. |
| `ConvexitySpace.trivial(X)` | Indiscrete convexity `{X}`. |
| `ConvexitySpace.from_closure_operator(X, g)` | Recover `G` from hull function `g`. |
| `.maximal_proper_convex_sets()` | Co-atoms of the convexity lattice. |
| `.minimal_nonempty_convex_sets()` | Atoms of the convexity lattice. |
| `len(space)` | `\|G\|` — number of convex sets. |
| `A in space` | Membership test (`frozenset(A) ∈ G`). |
| `iter(space)` | Iterate over `G` in deterministic sorted order. |

### `convexipy.enumeration`

| Function | Description |
|----------|-------------|
| `generate_convexities(n, *, grounded_only, n_ary, on_progress)` | Generator over all valid `ConvexitySpace` objects on `{0..n-1}`. |
| `count_convexities(n, *, grounded_only, n_ary)` | Count without materialising. |
| `count_grounded_convexities(n, *, n_ary)` | Shorthand for grounded-only count. |
| `enumerate_and_classify(n, *, n_ary, on_progress)` | Returns `EnumerationResult` with grounded/total split and identity verification. |

### `convexipy.transform`

| Function | Description |
|----------|-------------|
| `binomial_transform(seq)` | `bₙ = Σ C(n,k) aₖ` |
| `inverse_binomial_transform(seq)` | `aₙ = Σ (-1)^(n-k) C(n,k) bₖ` |
| `is_binomial_transform_pair(grounded, total)` | Check the paper's central identity. |
| `apply_binomial_identity(grounded, n)` | Compute `\|Γ_N(Xₙ)\|` from grounded counts. |
| `recover_grounded_sequence(total)` | Inverse: recover grounded counts from total. |

### `convexipy.oeis`

Pre-loaded Tables 1 & 2 from the paper, OEIS IDs, URL helpers, and `lookup_total(n, N)` / `lookup_grounded(n, N)` for O(1) reference lookups.

### `convexipy.applications`

| Function | Returns | Use case |
|----------|---------|----------|
| `preorder_from_relation(X, upsets)` | `PreorderResult` | Access-control hierarchy (use case 1) |
| `upset_closure(space, elements)` | `FrozenSet` | Reachability in a preorder |
| `feature_space_convexity(universe, extents)` | `ConvexitySpace` | Segment/FCA consistency (use case 2) |
| `is_separable(space, A, B)` | `bool` | Halfspace separability check |
| `graph_convexity_from_paths(adjacency)` | `ConvexitySpace` | Graph betweenness convexity (use case 3) |
| `validate_configuration_family(universe, configs)` | `ConfigurationValidationReport` | CI rule-table validation (use case 4) |
| `safe_enumerate(n, *, n_ary, grounded_only, max_n)` | `List[ConvexitySpace]` | Guarded enumeration for API endpoints |

---

## Complexity & scale guidance

| `n` | Filter | Count | Practical? |
|-----|--------|-------|------------|
| 0–4 | any | up to 2480 | Instantaneous (ms) |
| 5 | N=1 | 9053 | ~40s |
| 5 | N=1, grounded | 6942 | ~40s |
| 6 | N=1 | 257 151 | Minutes |
| 6 | N=2 | 1 556 743 050 | Long-running, multi-GB if materialised |
| ≥7 | unfiltered | unknown | Not tractable with current algorithm |

Use `count_convexities()` (streaming counter, no materialisation) when you only need the count. Use `generate_convexities()` with `on_progress=` for long-running enumerations with a progress indicator. Use `safe_enumerate(n, max_n=5)` in API handlers to guard against accidental large requests from user input.

---

## Testing

```bash
# All fast tests (198 tests, ~50s due to n=4 enumeration)
pytest -m "not slow"

# Including doctests in source
pytest --doctest-modules src/convexipy/ -m "not slow"

# All tests including slow n=5 verification (~3 minutes total)
pytest

# Run the quickstart example
python examples/quickstart.py
```

Tests are verified against published Tables 1 & 2 of Dulliev & Naumikhin (2026) for all arity values N=0..4 and carrier sizes n=0..5.

---

## References

1. **Dulliev, A. & Naumikhin, D. (2026).** "Binomial Transform of Sequences Counting N-ary Convexities." Kazan National Research Technical University. arXiv:2606.11252.

2. **Seiffarth, F., Horváth, T. & Wrobel, S. (2021).** "Maximal Closed Set and Half-Space Separations in Finite Closure Systems." arXiv:2001.04417.

3. **Bressan, M., Chepoi, V., Esposito, E. & Thiessen, M. (2025).** "Efficient Algorithms for Learning and Compressing Monophonic Halfspaces in Graphs." arXiv:2506.23186.

4. **Soltan, V. P. (1984).** *Introduction to the Axiomatic Theory of Convexity.* Kishinev: Shtiinca.

5. **van de Vel, M. L. J. (1993).** *Theory of Convex Structures.* North-Holland Mathematical Library, vol. 50.

6. **OEIS — A000798** (grounded 1-ary / quasi-orders / finite topologies): [https://oeis.org/A000798](https://oeis.org/A000798)

7. **OEIS — A326878** (total 1-ary / binary convexities): [https://oeis.org/A326878](https://oeis.org/A326878)

8. **OEIS — A364656** (grounded 2-ary convexities): [https://oeis.org/A364656](https://oeis.org/A364656)

9. **OEIS — A395658** (grounded 3-ary convexities): [https://oeis.org/A395658](https://oeis.org/A395658)

---

## License

MIT — see [LICENSE](LICENSE).
