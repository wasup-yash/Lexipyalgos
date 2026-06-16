"""
convexipy.applications
=======================

Application-layer helpers that map :mod:`convexipy.core` /
:mod:`convexipy.enumeration` onto concrete, named use cases. Each function
in this module is a thin, documented wrapper around the core primitives --
the underlying math is identical, but the names, inputs, and outputs are
chosen to match the vocabulary of the target domain, so application
developers do not need to first translate their problem into "ground sets
and intersection-closed families."

Use cases covered
------------------

1. **Preorder / hierarchy analysis** (:func:`preorder_from_relation`,
   :func:`upset_closure`) -- a *grounded, 1-ary* convexity on a finite set
   ``X`` is exactly the family of up-sets (order filters) of a preorder on
   ``X`` (this is the combinatorial content of OEIS A000798, see
   :mod:`convexipy.oeis`). This lets you validate that a proposed set of
   "reachable states" or "permission sets" in an access-control / workflow
   model is consistent with *some* underlying partial order, and compute
   the minimal such order.

2. **Feature-space / segment convexity** (:func:`feature_space_convexity`,
   :func:`is_separable`) -- in recommendation, customer-segmentation, and
   formal-concept-analysis settings, a family of "concept extents" (e.g.
   customer segments defined by shared attributes) is naturally an
   intersection-closed family. Checking convexity-axiom compliance flags
   segment definitions with inconsistent overlap rules, and the convex hull
   operator computes the smallest "official segment" containing an
   arbitrary group of users -- directly applicable to half-space
   separability questions in ML (Seiffarth, Horváth & Wrobel, 2021,
   arXiv:2001.04417).

3. **Graph / network convexity** (:func:`graph_convexity_from_paths`) --
   builds the convexity space whose convex sets are vertex sets closed
   under a path-based betweenness relation (e.g. monophonic/geodesic
   convexity, Bressan et al., arXiv:2506.23186), enabling convex-hull and
   halfspace queries on network data (community detection, influence
   propagation boundaries).

4. **Configuration-space validation** (:func:`validate_configuration_family`)
   -- many product/business rule systems define "valid configuration sets"
   (e.g. allowed combinations of plan tiers, feature flags, regional
   restrictions) that are implicitly assumed to compose via intersection.
   This validates that assumption and reports the precise violation,
   suitable for CI checks on rule-table changes.

References
----------
Seiffarth, F., Horváth, T., Wrobel, S. (2021). "Maximal Closed Set and
Half-Space Separations in Finite Closure Systems." arXiv:2001.04417.

Bressan, M., Chepoi, V., Esposito, E., Thiessen, M. (2025). "Efficient
Algorithms for Learning and Compressing Monophonic Halfspaces in Graphs."
arXiv:2506.23186.

Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities."
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .core import ConvexitySpace
from .enumeration import generate_convexities
from .exceptions import EnumerationLimitError, InvalidConvexityError

__all__ = [
    "preorder_from_relation",
    "upset_closure",
    "feature_space_convexity",
    "is_separable",
    "graph_convexity_from_paths",
    "validate_configuration_family",
    "ConfigurationValidationReport",
    "safe_enumerate",
    "PreorderResult",
]

#: Default safety limit for safe_enumerate -- carrier sizes beyond this
#: trigger EnumerationLimitError unless explicitly overridden. n=5 with no
#: filter is ~tens of seconds; n=6 unfiltered is impractical for an
#: interactive call.
_DEFAULT_MAX_N = 5


# ----------------------------------------------------------------------
# 1. Preorder / hierarchy analysis
# ----------------------------------------------------------------------


@dataclass
class PreorderResult:
    """Result of :func:`preorder_from_relation`.

    Attributes
    ----------
    space:
        The grounded :class:`ConvexitySpace` whose convex sets are exactly
        the up-sets of the recovered preorder.
    leq:
        The recovered preorder relation, as a set of pairs ``(x, y)``
        meaning ``x ≤ y`` (i.e. ``x`` is "below or equal to" ``y`` --
        equivalently, every up-set containing ``x`` also contains ``y``).
        Always reflexive: ``(x, x) ∈ leq`` for all ``x`` in the ground set.
    is_consistent:
        ``True`` iff the input ``upsets`` family, together with ``X`` and
        ``∅``, formed a valid (intersection-closed) family *before* any
        repair -- i.e. iff ``space`` was constructible without raising.
        When ``False``, ``space`` is the **intersection-closure** of the
        input (the smallest valid family containing it), and ``leq`` is
        derived from that closure -- useful for showing "the closest
        consistent hierarchy" to an inconsistent input.
    """

    space: ConvexitySpace
    leq: Set[Tuple[Hashable, Hashable]]
    is_consistent: bool


def preorder_from_relation(
    ground_set: Iterable[Hashable], upsets: Iterable[Iterable[Hashable]]
) -> PreorderResult:
    """Recover a preorder from a proposed family of "up-sets".

    Many access-control, workflow, and org-chart systems implicitly define
    a hierarchy by listing, for various subsets of states/roles/permissions,
    "the set of things reachable from here" -- an *up-set* in preorder
    terms. This function takes such a family, closes it under intersection
    (always including ``X`` and ``∅``) to obtain a valid **grounded 1-ary
    convexity**, and reads off the corresponding preorder ``≤`` via

    .. math::
        x \\le y \\iff \\text{every up-set containing } x \\text{ also
        contains } y.

    This is the precise combinatorial content behind OEIS **A000798**
    (quasi-orders / preorders / finite topologies on ``n`` labeled points),
    which Table 2 of Dulliev & Naumikhin (2026) identifies with the
    *grounded 1-ary* convexity counts (see :mod:`convexipy.oeis`).

    Parameters
    ----------
    ground_set:
        The elements ``X`` (roles, states, users, ...).
    upsets:
        An iterable of iterables, each a proposed up-set (subset of
        ``ground_set``). ``X`` and ``∅`` are added automatically if absent.

    Returns
    -------
    PreorderResult

    Examples
    --------
    A simple 3-level hierarchy: ``admin`` can reach everything, ``editor``
    can reach ``editor`` and ``viewer``, ``viewer`` can reach only itself.

    >>> result = preorder_from_relation(
    ...     {"admin", "editor", "viewer"},
    ...     [{"admin", "editor", "viewer"}, {"editor", "viewer"}, {"viewer"}, set()],
    ... )
    >>> result.is_consistent
    True
    >>> ("admin", "viewer") in result.leq
    True
    >>> ("viewer", "admin") in result.leq
    False
    """
    X = frozenset(ground_set)
    families = {frozenset(s) & X for s in upsets}
    families.add(X)
    families.add(frozenset())

    is_consistent = True
    try:
        space = ConvexitySpace(X, families)
    except InvalidConvexityError:
        is_consistent = False
        # Repair: take the intersection-closure of the proposed family.
        closure = set(families)
        changed = True
        while changed:
            changed = False
            for A, B in itertools.combinations(list(closure), 2):
                inter = A & B
                if inter not in closure:
                    closure.add(inter)
                    changed = True
        space = ConvexitySpace.unchecked(X, closure)

    # Read off the preorder: x <= y iff every convex set (up-set)
    # containing x also contains y. Equivalently, y is in g({x}) for the
    # *down*-closure... here, since members of G are up-sets, x <= y means
    # "x is below y" i.e. being in an up-set containing x forces y in too,
    # which means y is in every up-set that contains x, i.e. y in g({x})
    # under the convention that g({x}) = smallest up-set containing x =
    # the up-set of x itself = {z : x <= z}.
    leq: Set[Tuple[Hashable, Hashable]] = set()
    for x in X:
        up_x = space.convex_hull({x})
        for y in up_x:
            leq.add((x, y))

    return PreorderResult(space=space, leq=leq, is_consistent=is_consistent)


def upset_closure(
    space: ConvexitySpace, elements: Iterable[Hashable]
) -> FrozenSet[Hashable]:
    """Return the smallest up-set (convex set) containing ``elements``.

    Thin, domain-named wrapper around :meth:`ConvexitySpace.convex_hull` for
    use with grounded 1-ary convexities produced by
    :func:`preorder_from_relation`: given a set of "starting states",
    returns every state reachable from any of them under the recovered
    preorder.

    Parameters
    ----------
    space:
        A (grounded, 1-ary) :class:`ConvexitySpace`, typically
        ``preorder_from_relation(...).space``.
    elements:
        The starting elements.

    Returns
    -------
    FrozenSet[Hashable]
        ``g(elements)``, the closure under the preorder.

    Examples
    --------
    >>> result = preorder_from_relation(
    ...     {"a", "b", "c"}, [{"a", "b", "c"}, {"b", "c"}, {"c"}, set()]
    ... )
    >>> sorted(upset_closure(result.space, {"a"}))
    ['a', 'b', 'c']
    >>> sorted(upset_closure(result.space, {"c"}))
    ['c']
    """
    return space.convex_hull(elements)


# ----------------------------------------------------------------------
# 2. Feature-space / segment convexity, separability
# ----------------------------------------------------------------------


def feature_space_convexity(
    universe: Iterable[Hashable],
    concept_extents: Iterable[Iterable[Hashable]],
    *,
    auto_close: bool = True,
) -> ConvexitySpace:
    """Build a :class:`ConvexitySpace` from a family of "concept extents".

    In formal concept analysis, recommendation, and customer-segmentation
    settings, each "concept" (segment, cluster, tag-defined cohort, ...) is
    identified with its *extent* -- the set of entities (users, items,
    records) belonging to it. The family of all such extents is naturally
    intersection-closed: "users satisfying both concept A and concept B" is
    itself a meaningful extent. This function packages a list of extents
    into a :class:`ConvexitySpace`, which then exposes :meth:`convex_hull`
    (the smallest *officially defined* segment containing an arbitrary
    group), :meth:`is_n_ary` (testing whether segment membership is
    determined by small "diagnostic" sub-groups), and lattice-structure
    queries (:meth:`~ConvexitySpace.maximal_proper_convex_sets`,
    :meth:`~ConvexitySpace.minimal_nonempty_convex_sets`).

    Parameters
    ----------
    universe:
        The full entity set ``X``.
    concept_extents:
        An iterable of iterables, each a proposed extent (subset of
        ``universe``).
    auto_close:
        If ``True`` (default), automatically add ``universe``, ``∅``, and
        all pairwise intersections (transitively) so the result is always a
        valid :class:`ConvexitySpace`, even if the input extents were not
        already intersection-closed. If ``False``, the raw
        ``concept_extents`` (plus ``universe``) are passed directly to
        :class:`ConvexitySpace`, which will raise
        :class:`~convexipy.exceptions.InvalidConvexityError` if they are not
        already closed -- use this to *detect* inconsistent extent
        definitions rather than silently repairing them (see also
        :func:`validate_configuration_family` for a reporting-oriented
        variant).

    Returns
    -------
    ConvexitySpace

    Examples
    --------
    >>> users = {"alice", "bob", "carol", "dave"}
    >>> segments = [
    ...     {"alice", "bob", "carol"},   # "premium" segment
    ...     {"bob", "carol", "dave"},    # "active" segment
    ... ]
    >>> space = feature_space_convexity(users, segments)
    >>> sorted(space.convex_hull({"bob"}))
    ['bob', 'carol']
    """
    X = frozenset(universe)
    families = {frozenset(s) & X for s in concept_extents}
    families.add(X)

    if not auto_close:
        return ConvexitySpace(X, families)

    families.add(frozenset())
    closure = set(families)
    changed = True
    while changed:
        changed = False
        for A, B in itertools.combinations(list(closure), 2):
            inter = A & B
            if inter not in closure:
                closure.add(inter)
                changed = True
    return ConvexitySpace.unchecked(X, closure)


def is_separable(
    space: ConvexitySpace,
    group_a: Iterable[Hashable],
    group_b: Iterable[Hashable],
) -> bool:
    """Check whether two groups are "halfspace-separable" in ``space``.

    Two sets ``A`` and ``B`` are separable in a convexity space iff their
    convex hulls are disjoint: :math:`g(A) \\cap g(B) = \\emptyset`. This is
    the abstract-convexity generalisation of linear separability (cf. the
    classical fact that two finite point sets in :math:`\\mathbb{R}^d` are
    linearly separable iff their convex hulls are disjoint, Kakutani 1937),
    applied here to arbitrary finite closure systems as studied for ML
    half-space-separation problems (Seiffarth, Horváth & Wrobel, 2021,
    arXiv:2001.04417).

    Parameters
    ----------
    space:
        The :class:`ConvexitySpace` (e.g. from
        :func:`feature_space_convexity`).
    group_a, group_b:
        Two subsets of ``space.ground_set``.

    Returns
    -------
    bool
        ``True`` iff ``g(group_a) ∩ g(group_b) == ∅``.

    Examples
    --------
    >>> X = {0, 1, 2, 3}
    >>> space = ConvexitySpace(X, [X, {0, 1}, {2, 3}, set()])
    >>> is_separable(space, {0}, {3})
    True
    >>> is_separable(space, {0}, {1})
    False
    """
    hull_a = space.convex_hull(group_a)
    hull_b = space.convex_hull(group_b)
    return hull_a.isdisjoint(hull_b)


# ----------------------------------------------------------------------
# 3. Graph / network convexity
# ----------------------------------------------------------------------


def _all_paths_vertices(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    u: Hashable,
    v: Hashable,
    *,
    induced_only: bool,
) -> Set[FrozenSet[Hashable]]:
    """Enumerate vertex-sets of all simple u-v paths (helper)."""
    nodes = set(adjacency.keys())
    results: Set[FrozenSet[Hashable]] = set()

    def dfs(current: Hashable, visited: List[Hashable]) -> None:
        if current == v:
            results.add(frozenset(visited))
            return
        for nxt in adjacency.get(current, ()):  # type: ignore[union-attr]
            if nxt not in visited:
                dfs(nxt, visited + [nxt])

    dfs(u, [u])
    return results


def graph_convexity_from_paths(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    *,
    max_pair_samples: Optional[int] = None,
) -> ConvexitySpace:
    """Build the *geodesic/monophonic-style* convexity induced by a graph.

    Given an (undirected) graph specified as an adjacency mapping, this
    constructs the convexity on the vertex set ``V`` whose convex sets are
    exactly the **vertex-induced intervals**: for each pair ``u, v``, the
    union of vertex sets of all simple ``u``-``v`` paths forms a candidate
    convex set ("interval"), and the convexity ``G`` is the
    intersection-closure of ``{V}`` together with all such intervals (plus
    ``∅`` and all singletons, which are always intervals of the trivial
    path). This realises, in finite form, the graph-convexity framework
    used for monophonic halfspace learning (Bressan, Chepoi, Esposito &
    Thiessen, 2025, arXiv:2506.23186) and for the independent-set-based
    convexity construction of intersection-closed families from graphs
    (arXiv:2403.17910, §2.2).

    Parameters
    ----------
    adjacency:
        Mapping from each vertex to an iterable of its neighbours. The
        graph is treated as undirected; ``adjacency`` need not be
        symmetric (both directions are considered when searching for
        paths), but for clarity callers should typically supply a symmetric
        mapping.
    max_pair_samples:
        If given, limits the number of ``(u, v)`` vertex pairs considered
        to this many (sampled in iteration order of ``adjacency``).
        Path enumeration between two vertices is itself exponential in
        graph density, so for dense or large graphs, restrict either the
        graph size or this parameter. ``None`` (default) considers all
        pairs -- suitable for small graphs (roughly ``|V| <= 8`` for dense
        graphs, larger for sparse/tree-like graphs).

    Returns
    -------
    ConvexitySpace
        The induced convexity on ``V``.

    Examples
    --------
    A path graph ``0 - 1 - 2``: the only interval containing ``{0, 2}`` is
    all of ``V``.

    >>> g = {0: [1], 1: [0, 2], 2: [1]}
    >>> space = graph_convexity_from_paths(g)
    >>> space.convex_hull({0, 2}) == frozenset({0, 1, 2})
    True
    """
    V = frozenset(adjacency.keys())
    families: Set[FrozenSet[Hashable]] = {V, frozenset()}
    families.update(frozenset({v}) for v in V)

    pairs: List[Tuple[Hashable, Hashable]] = []
    vertices = list(V)
    for i, u in enumerate(vertices):
        for v in vertices[i + 1 :]:
            pairs.append((u, v))
    if max_pair_samples is not None:
        pairs = pairs[:max_pair_samples]

    for u, v in pairs:
        path_vertex_sets = _all_paths_vertices(adjacency, u, v, induced_only=True)
        if path_vertex_sets:
            interval = frozenset().union(*path_vertex_sets)
            families.add(interval)

    closure = set(families)
    changed = True
    while changed:
        changed = False
        for A, B in itertools.combinations(list(closure), 2):
            inter = A & B
            if inter not in closure:
                closure.add(inter)
                changed = True

    return ConvexitySpace.unchecked(V, closure)


# ----------------------------------------------------------------------
# 4. Configuration-space validation
# ----------------------------------------------------------------------


@dataclass
class ConfigurationValidationReport:
    """Result of :func:`validate_configuration_family`.

    Attributes
    ----------
    is_valid:
        ``True`` iff the input family already satisfies the convexity
        axioms (contains ``X``, all members are subsets of ``X``, and is
        closed under pairwise intersection).
    missing_ground_set:
        ``True`` iff the universe ``X`` itself was not present in the input
        family (and was therefore added to form ``space``).
    out_of_universe_members:
        Tuple of (input-index, element) pairs for elements found in some
        input set but not in ``universe`` -- these were silently dropped
        when forming ``space`` (via intersection with ``X``); reported here
        so callers can flag a likely typo upstream.
    missing_intersections:
        Tuple of ``(set_a, set_b, intersection)`` triples (as ``frozenset``)
        for every pair of *input* sets whose pairwise intersection was not
        itself present in the input family. Each such triple represents one
        concrete "your rule table is missing this combination" finding.
    space:
        The repaired :class:`ConvexitySpace`: the intersection-closure of
        the input family together with ``universe`` and ``∅``. Always a
        valid convexity, regardless of ``is_valid``.
    """

    is_valid: bool
    missing_ground_set: bool
    out_of_universe_members: Tuple[Tuple[int, Hashable], ...]
    missing_intersections: Tuple[
        Tuple[FrozenSet[Hashable], FrozenSet[Hashable], FrozenSet[Hashable]], ...
    ]
    space: ConvexitySpace

    def summary(self) -> str:
        """Return a short human-readable summary of findings."""
        if self.is_valid:
            return "Configuration family is a valid convexity (no issues found)."
        lines: List[str] = ["Configuration family is NOT closed under intersection:"]
        if self.missing_ground_set:
            lines.append(
                "  - The universe set X was not present among the inputs."
            )
        for idx, elem in self.out_of_universe_members:
            lines.append(
                f"  - Input #{idx} contains element {elem!r} not in the "
                "declared universe; it was dropped."
            )
        for A, B, inter in self.missing_intersections:
            lines.append(
                f"  - Intersection of {set(A)} and {set(B)} is "
                f"{set(inter)}, which is not among the declared "
                "configurations."
            )
        return "\n".join(lines)


def validate_configuration_family(
    universe: Iterable[Hashable],
    configurations: Sequence[Iterable[Hashable]],
) -> ConfigurationValidationReport:
    """Validate that a family of "allowed configuration sets" is
    intersection-closed, with a detailed diagnostic report.

    Many business rule systems (allowed plan-tier × feature-flag × region
    combinations, valid permission bundles, compatible-options matrices)
    implicitly assume that *any combination of constraints* yields another
    valid configuration -- i.e. that the family of "configuration sets"
    (each set being "the configurations satisfying constraint ``i``") is
    closed under intersection. This function checks that assumption
    directly and returns a structured report pinpointing every missing
    combination, suitable for a CI check on rule-table pull requests.

    Parameters
    ----------
    universe:
        The full set of atomic configurations ``X`` (e.g. every
        ``(tier, flag, region)`` triple as a single hashable token).
    configurations:
        A sequence of subsets of ``universe``, each representing "the
        configurations satisfying constraint ``i``" for some ``i``.

    Returns
    -------
    ConfigurationValidationReport

    Examples
    --------
    >>> universe = {"A1", "A2", "B1", "B2"}
    >>> configs = [{"A1", "A2"}, {"A1", "B1"}]  # intersection {"A1"} missing
    >>> report = validate_configuration_family(universe, configs)
    >>> report.is_valid
    False
    >>> {"A1"} in [set(t[2]) for t in report.missing_intersections]
    True
    """
    X = frozenset(universe)
    input_sets: List[FrozenSet[Hashable]] = [frozenset(c) for c in configurations]

    out_of_universe: List[Tuple[int, Hashable]] = []
    for idx, s in enumerate(input_sets):
        for elem in s - X:
            out_of_universe.append((idx, elem))

    clipped_sets = [s & X for s in input_sets]
    family = set(clipped_sets)
    missing_ground_set = X not in family

    missing_intersections: List[
        Tuple[FrozenSet[Hashable], FrozenSet[Hashable], FrozenSet[Hashable]]
    ] = []
    for A, B in itertools.combinations(clipped_sets, 2):
        inter = A & B
        if inter not in family and (A, B, inter) not in [
            (a, b, i) for a, b, i in missing_intersections
        ]:
            missing_intersections.append((A, B, inter))

    is_valid = (
        not out_of_universe
        and not missing_ground_set
        and not missing_intersections
    )

    closure = set(family)
    closure.add(X)
    closure.add(frozenset())
    changed = True
    while changed:
        changed = False
        for A, B in itertools.combinations(list(closure), 2):
            inter = A & B
            if inter not in closure:
                closure.add(inter)
                changed = True
    space = ConvexitySpace.unchecked(X, closure)

    return ConfigurationValidationReport(
        is_valid=is_valid,
        missing_ground_set=missing_ground_set,
        out_of_universe_members=tuple(out_of_universe),
        missing_intersections=tuple(missing_intersections),
        space=space,
    )


# ----------------------------------------------------------------------
# Safe enumeration wrapper
# ----------------------------------------------------------------------


def safe_enumerate(
    n: int,
    *,
    n_ary: Optional[int] = None,
    grounded_only: bool = False,
    max_n: int = _DEFAULT_MAX_N,
) -> List[ConvexitySpace]:
    """:func:`~convexipy.enumeration.generate_convexities`, materialised,
    with a default safety guard against accidentally-huge requests.

    This is the recommended entry point for application code (e.g. an API
    endpoint that lets a user request "all convexities on my 4-element
    config set") where ``n`` is not a hardcoded research constant and might
    come from user input.

    Parameters
    ----------
    n:
        Carrier size.
    n_ary, grounded_only:
        See :func:`~convexipy.enumeration.generate_convexities`.
    max_n:
        Raise :class:`~convexipy.exceptions.EnumerationLimitError` if
        ``n > max_n``. Default ``5``, chosen because unfiltered ``n=5``
        enumeration completes in low tens of seconds on commodity hardware
        (see :mod:`convexipy.enumeration` complexity notes), making it a
        reasonable upper bound for a synchronous call. Pass a larger
        ``max_n`` explicitly (and consider an ``n_ary`` filter) for
        background/batch jobs.

    Returns
    -------
    list[ConvexitySpace]

    Raises
    ------
    EnumerationLimitError
        If ``n > max_n``.

    Examples
    --------
    >>> spaces = safe_enumerate(3, n_ary=1)
    >>> len(spaces)
    45
    >>> safe_enumerate(100)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    convexipy.exceptions.EnumerationLimitError: ...
    """
    if n > max_n:
        raise EnumerationLimitError(
            n,
            max_n,
            hint=(
                "Pass max_n explicitly to override, and/or supply an "
                "n_ary filter to reduce the result size."
            ),
        )
    return list(
        generate_convexities(n, grounded_only=grounded_only, n_ary=n_ary)
    )
