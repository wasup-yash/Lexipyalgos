"""
examples/quickstart.py
========================

A self-contained walkthrough of convexipy's core features.

Run with:
    python examples/quickstart.py

No external dependencies are required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import convexipy
from convexipy import (
    ConvexitySpace,
    InvalidConvexityError,
    binomial_transform,
    count_convexities,
    count_grounded_convexities,
    enumerate_and_classify,
    generate_convexities,
    inverse_binomial_transform,
    is_binomial_transform_pair,
    recover_grounded_sequence,
)
from convexipy.applications import (
    feature_space_convexity,
    graph_convexity_from_paths,
    is_separable,
    preorder_from_relation,
    validate_configuration_family,
)
from convexipy.oeis import OEIS_REFERENCES, lookup_grounded, lookup_total, oeis_url

print("=" * 60)
print(f"  convexipy {convexipy.__version__} — quickstart walkthrough")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# 1. Build a convexity space and compute convex hulls
# ──────────────────────────────────────────────────────────────
print("\n── 1. ConvexitySpace basics ─────────────────────────────")

X = {0, 1, 2}
G = [{0, 1, 2}, {0, 1}, {0}, set()]
space = ConvexitySpace(X, G)

print(f"space = {space}")
print(f"Convex sets: {[set(s) for s in space]}")
print(f"g({{1}})  = {set(space.convex_hull({1}))}")   # {0,1}
print(f"g({{2}})  = {set(space.convex_hull({2}))}")   # {0,1,2}
print(f"g({{0}})  = {set(space.convex_hull({0}))}")   # {0}
print(f"g({{  }}) = {set(space.convex_hull(set()))}")  # {}
print(f"is_grounded: {space.is_grounded()}")
print(f"arity:       {space.arity()}")


# ──────────────────────────────────────────────────────────────
# 2. Factory constructors
# ──────────────────────────────────────────────────────────────
print("\n── 2. Factory constructors ──────────────────────────────")

discrete = ConvexitySpace.discrete({0, 1, 2})
print(f"Discrete |G|:        {len(discrete)}  (full power set = 2^3 = 8)")

trivial = ConvexitySpace.trivial({0, 1, 2})
print(f"Trivial  |G|:        {len(trivial)}   (only X itself)")

def my_hull(A):
    return frozenset(X) if len(A) >= 2 else frozenset(A)

from_op = ConvexitySpace.from_closure_operator(X, my_hull)
print(f"From closure op |G|: {len(from_op)}  ({[set(s) for s in from_op]})")


# ──────────────────────────────────────────────────────────────
# 3. Grounding / de-grounding (Lemmas 1-5, Dulliev & Naumikhin 2026)
# ──────────────────────────────────────────────────────────────
print("\n── 3. Grounding and de-grounding ────────────────────────")

non_grounded = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}])
C = non_grounded.minimal_convex_set()
print(f"G (non-grounded): {[set(s) for s in non_grounded]}")
print(f"g(∅) = C = {set(C)}")

reflection = non_grounded.get_grounded_reflection()
print(f"H (grounded reflection on Y = X\\C): {[set(s) for s in reflection]}")
print(f"H.ground_set = {set(reflection.ground_set)}")
print(f"H.is_grounded = {reflection.is_grounded()}")

rebuilt = ConvexitySpace.from_grounded(reflection, C)
print(f"from_grounded(H, C) == original: {rebuilt == non_grounded}")


# ──────────────────────────────────────────────────────────────
# 4. Enumeration and counting
# ──────────────────────────────────────────────────────────────
print("\n── 4. Enumeration and counting ──────────────────────────")

for n in range(5):
    total = count_convexities(n, n_ary=1)
    grounded = count_grounded_convexities(n, n_ary=1)
    print(f"  n={n}: |Γ₁(Xₙ)|={total:>6},  |Γ⁰₁(Xₙ)|={grounded:>6}")

result = enumerate_and_classify(3, n_ary=1)
print(f"\nenumerate_and_classify(n=3, N=1):")
print(f"  total spaces:    {result.total}")
print(f"  grounded spaces: {result.grounded_total}")
print(f"  first 3 spaces:  {[[set(s) for s in sp] for sp in result.spaces[:3]]}")


# ──────────────────────────────────────────────────────────────
# 5. Binomial transform identity (Proposition, §2 of the paper)
# ──────────────────────────────────────────────────────────────
print("\n── 5. Binomial transform identity ───────────────────────")

grounded_N1 = [count_grounded_convexities(k, n_ary=1) for k in range(5)]
total_N1    = [count_convexities(k, n_ary=1) for k in range(5)]

print(f"Grounded N=1 counts: {grounded_N1}")
print(f"Total    N=1 counts: {total_N1}")
print(f"binomial_transform(grounded) == total: "
      f"{binomial_transform(grounded_N1) == total_N1}")
print(f"is_binomial_transform_pair:           "
      f"{is_binomial_transform_pair(grounded_N1, total_N1)}")
print(f"recover_grounded_sequence(total):     "
      f"{recover_grounded_sequence(total_N1)}")

grounded_seq = [1, 1, 4, 29]   # |Γ⁰₁(Xₖ)| for k=0..3 (length = n+1 = 4)
print(f"\nVerify binomial_identity on result: "
      f"{result.verify_binomial_identity(grounded_seq)}")


# ──────────────────────────────────────────────────────────────
# 6. OEIS cross-reference
# ──────────────────────────────────────────────────────────────
print("\n── 6. OEIS reference data ───────────────────────────────")

for (N, grnd), oeis_id in OEIS_REFERENCES.items():
    kind = "grounded" if grnd else "total  "
    label = oeis_id if oeis_id else "(new — not yet in OEIS)"
    url = f" → {oeis_url(oeis_id)}" if oeis_id else ""
    print(f"  N={N} {kind}: {label}{url}")

print(f"\nlookup_total(n=5, N=1)    = {lookup_total(5, 1)}")
print(f"lookup_grounded(n=6, N=1) = {lookup_grounded(6, 1)}")


# ──────────────────────────────────────────────────────────────
# 7. Application: preorder / hierarchy recovery
# ──────────────────────────────────────────────────────────────
print("\n── 7. Application: preorder recovery ────────────────────")

result_po = preorder_from_relation(
    {"admin", "editor", "viewer"},
    [{"admin", "editor", "viewer"}, {"editor", "viewer"}, {"viewer"}, set()],
)
print(f"Consistent: {result_po.is_consistent}")
print(f"Preorder ≤ : {sorted(result_po.leq)}")


# ──────────────────────────────────────────────────────────────
# 8. Application: feature-space / segment convexity
# ──────────────────────────────────────────────────────────────
print("\n── 8. Application: segment convexity & separability ─────")

users = {"alice", "bob", "carol", "dave"}
segments = [{"alice", "bob", "carol"}, {"bob", "carol", "dave"}]
fs = feature_space_convexity(users, segments)

print(f"Segment space |G|: {len(fs)}")
print(f"Hull of {{bob}}:    {set(fs.convex_hull({'bob'}))}")
print(f"Hull of {{alice}}:  {set(fs.convex_hull({'alice'}))}")
print(f"is_separable({{alice}},{{dave}}): "
      f"{is_separable(fs, {'alice'}, {'dave'})}")
print(f"is_separable({{bob}},{{carol}}):  "
      f"{is_separable(fs, {'bob'}, {'carol'})}")


# ──────────────────────────────────────────────────────────────
# 9. Application: graph convexity
# ──────────────────────────────────────────────────────────────
print("\n── 9. Application: graph convexity ──────────────────────")

# Linear path: 0 — 1 — 2 — 3
path_graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
gc = graph_convexity_from_paths(path_graph)
print(f"Path graph 0-1-2-3:")
print(f"  g({{0,3}}) = {set(gc.convex_hull({0,3}))} (entire path)")
print(f"  g({{0,2}}) = {set(gc.convex_hull({0,2}))} (0 through 2)")
print(f"  Convex sets: {[set(s) for s in gc]}")


# ──────────────────────────────────────────────────────────────
# 10. Application: configuration validation
# ──────────────────────────────────────────────────────────────
print("\n── 10. Application: configuration validation ────────────")

universe = {"basic", "pro", "enterprise", "addon_a", "addon_b"}
# Valid bundles (each is a set of features a customer can hold simultaneously)
bundles = [
    {"basic", "addon_a"},
    {"pro", "addon_a", "addon_b"},
    # Missing: {"basic", "pro", "addon_a", "addon_b"} — their intersection
]
report = validate_configuration_family(universe, bundles)
print(f"is_valid: {report.is_valid}")
print(report.summary())


# ──────────────────────────────────────────────────────────────
# 11. Validation and error handling
# ──────────────────────────────────────────────────────────────
print("\n── 11. Validation & error handling ─────────────────────")

try:
    # {0} ∩ {1} = {} is missing → invalid
    ConvexitySpace({0, 1}, [{0, 1}, {0}, {1}])
except InvalidConvexityError as e:
    print(f"InvalidConvexityError caught: {e}")

from convexipy.exceptions import EnumerationLimitError
from convexipy.applications import safe_enumerate
try:
    safe_enumerate(100)
except EnumerationLimitError as e:
    print(f"EnumerationLimitError: n={e.n}, limit={e.limit}")

print("\n── Serialization round-trip ──────────────────────────────")
json_str = space.to_json(indent=2)
print(f"to_json():\n{json_str}")
restored = ConvexitySpace.from_json(json_str)
print(f"Restored == original: {restored == space}")

print("\n" + "=" * 60)
print("  All examples completed successfully.")
print("=" * 60)
