"""
convexipy.enumeration
=====================

Algorithms for **enumerating** abstract convexity spaces on finite carriers.

The core algorithm is a depth-first backtracking search over the power set
``P(X)`` of a carrier ``X = {0, 1, ..., n-1}``. Each subset is assigned a
ternary state -- ``UNDECIDED``, ``INCLUDED`` (in ``G``), or ``EXCLUDED``
(not in ``G``) -- and the search explores both ``INCLUDED`` and ``EXCLUDED``
branches for each undecided subset, in order of decreasing cardinality.
*Incremental closure propagation* ensures that whenever a subset is marked
``INCLUDED``, all pairwise intersections with already-included subsets are
immediately forced ``INCLUDED`` too (pruning the branch if any such forced
inclusion collides with an existing ``EXCLUDED`` mark).

This module exposes three layers of API:

* :func:`generate_convexities` -- low-level generator, yields
  :class:`~convexipy.core.ConvexitySpace` objects one at a time (memory-
  efficient, suitable for streaming to disk or a database).
* :func:`count_convexities` / :func:`count_grounded_convexities` -- fast
  counting without materialising every space (same algorithm, but only a
  running total is kept).
* :class:`EnumerationResult` -- a convenience container returned by
  :func:`enumerate_and_classify`, bundling counts by arity alongside the
  raw spaces, suitable for direct comparison against the published OEIS
  sequences (Tables 1 & 2 of Dulliev & Naumikhin, 2026).

Complexity & practical limits
------------------------------
The search space is over subsets of ``P(X)``, i.e. ``2^(2^n)`` raw binary
choices before pruning. Pruning via incremental closure is extremely
effective in practice (the paper reports complete enumeration up to
``n = 6``), but enumeration is fundamentally super-exponential in ``n``.
As a rule of thumb:

* ``n <= 4``: instantaneous (milliseconds).
* ``n = 5``: with an ``N=1`` filter, both
  :math:`|\\Gamma_1(X_5)| = 9{,}053` and
  :math:`|\\Gamma^0_1(X_5)| = 6{,}942` complete in roughly half a minute
  on commodity hardware. Unfiltered ``n=5`` (no ``n_ary`` argument) is
  slower still, since the search itself does not shrink even when results
  are filtered after the fact.
* ``n = 6``: the ``N=1`` row is tractable
  (:math:`|\\Gamma_1(X_6)| = 257{,}151`,
  :math:`|\\Gamma^0_1(X_6)| = 209{,}527`) but takes considerably longer than
  ``n=5``. Higher-arity or unfiltered ``n=6`` searches (e.g. the ``N=2``
  row, :math:`|\\Gamma_2(X_6)| = 1{,}556{,}743{,}050`) are multi-gigabyte,
  long-running computations not practical for interactive use.
* ``n >= 7``: not currently tractable for *full, unfiltered* enumeration
  with this algorithm; use ``grounded_only=True`` and/or an ``N``-ary
  filter to restrict the search space, or use :func:`count_convexities`
  if only counts (not the spaces themselves) are needed.

References
----------
Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities."
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

from .core import ConvexitySpace

__all__ = [
    "generate_convexities",
    "count_convexities",
    "count_grounded_convexities",
    "enumerate_and_classify",
    "EnumerationResult",
    "EnumerationStats",
]

# Ternary subset states used by the backtracking search.
_UNDECIDED = 0
_INCLUDED = 1
_EXCLUDED = -1


def _build_subset_universe(n: int) -> Tuple[List[FrozenSet[int]], Dict[FrozenSet[int], int]]:
    """Enumerate ``P({0,...,n-1})``, sorted by ``(|A|, A)`` descending.

    Descending-size ordering is a search heuristic: deciding large subsets
    first tends to trigger closure propagation (which forces many smaller
    subsets) earlier, shrinking the remaining branching factor faster.

    Returns
    -------
    (all_subsets, subset_to_idx)
        ``all_subsets[i]`` is the ``i``-th subset; ``subset_to_idx`` maps
        each subset back to its index.
    """
    ground_set = frozenset(range(n))
    elements = list(ground_set)

    all_subsets: List[FrozenSet[int]] = []
    for r in range(n + 1):
        for combo in itertools.combinations(elements, r):
            all_subsets.append(frozenset(combo))

    all_subsets.sort(key=lambda s: (len(s), sorted(s)), reverse=True)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    return all_subsets, subset_to_idx


def _compute_closure_incremental(
    current_closure: set,
    new_element: FrozenSet[int],
    states: List[int],
    subset_to_idx: Dict[FrozenSet[int], int],
) -> Tuple[bool, List[int]]:
    """Incrementally extend ``current_closure`` by intersection-closing
    ``new_element`` against it, checking consistency against ``states``.

    Parameters
    ----------
    current_closure:
        The set of subsets currently known to be ``INCLUDED`` (as a Python
        ``set`` of ``frozenset``).
    new_element:
        The subset being tentatively included.
    states:
        Ternary state array, indexed by ``subset_to_idx``. **Not** mutated
        by this function -- callers apply ``newly_added_indices`` themselves
        (this allows clean backtracking).
    subset_to_idx:
        Maps each subset to its index in ``states``.

    Returns
    -------
    (is_valid, newly_added_indices)
        ``is_valid`` is ``False`` iff intersection-closing ``new_element``
        would force some ``EXCLUDED`` subset to become ``INCLUDED``
        (an immediate contradiction -- prune this branch).
        ``newly_added_indices`` lists the indices of subsets that must
        additionally become ``INCLUDED`` as a consequence (only those
        previously ``UNDECIDED``; already-``INCLUDED`` subsets are not
        re-listed).
    """
    if new_element in current_closure:
        return True, []

    closure = set(current_closure)
    queue = [new_element]
    closure.add(new_element)

    newly_added_indices: List[int] = []
    new_idx = subset_to_idx[new_element]
    if states[new_idx] == _EXCLUDED:
        return False, []
    if states[new_idx] == _UNDECIDED:
        newly_added_indices.append(new_idx)

    head = 0
    while head < len(queue):
        item = queue[head]
        head += 1

        for other in list(closure):
            inter = item & other
            if inter not in closure:
                inter_idx = subset_to_idx[inter]
                if states[inter_idx] == _EXCLUDED:
                    return False, []
                closure.add(inter)
                queue.append(inter)
                if states[inter_idx] == _UNDECIDED:
                    newly_added_indices.append(inter_idx)

    return True, newly_added_indices


def generate_convexities(
    n: int,
    *,
    grounded_only: bool = False,
    n_ary: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Iterator[ConvexitySpace]:
    """Yield every valid :class:`ConvexitySpace` on the carrier
    ``{0, 1, ..., n-1}``.

    Parameters
    ----------
    n:
        Size of the carrier set ``X``. Must satisfy ``n >= 0``.
    grounded_only:
        If ``True``, additionally require ``∅ ∈ G`` (i.e. enumerate only
        :math:`\\Gamma^0(X_n)` instead of :math:`\\Gamma(X_n)`). This
        roughly squares the pruning effectiveness for the same ``n``,
        since the empty set is forced ``INCLUDED`` from the start.
    n_ary:
        If given, only yield spaces ``G`` that are ``N``-ary for this
        value of ``N`` (post-filtered via
        :meth:`ConvexitySpace.is_n_ary`). For small ``N`` (e.g. ``N=1``,
        the most commonly studied case) this dramatically reduces the
        number of yielded spaces relative to the unfiltered total, but
        the *search* itself still explores the full unfiltered space --
        filtering happens after each candidate is fully constructed.
        For ``n <= 6`` this remains practical (it is exactly how the
        paper's Tables 1-2 were produced).
    on_progress:
        Optional callback ``on_progress(yielded_count, decided_subsets)``
        invoked each time a complete convexity is found (before any
        ``n_ary`` filtering), where ``decided_subsets`` is always
        ``2^n`` (the total number of subsets, all of which are decided
        at a leaf of the search tree). Useful for driving a progress bar
        in long-running ``n=6`` enumerations; the second argument is a
        constant and exists primarily for forward API stability.

    Yields
    ------
    ConvexitySpace
        Each valid convexity, constructed via
        :meth:`ConvexitySpace.unchecked` (axioms are guaranteed by
        construction, so validation is skipped for speed).

    Raises
    ------
    ValueError
        If ``n < 0``.

    Notes
    -----
    See the module docstring for complexity guidance. Calling this with
    ``n_ary=None`` (the default) performs **no** arity filtering at all --
    every valid convexity on ``X`` is yielded, regardless of its arity. For
    ``n=6`` this is impractical: even the largest *filtered* row published
    for ``n=6`` (the ``N=2`` row, :math:`|\\Gamma_2(X_6)| = 1{,}556{,}743{,}050`,
    see :func:`convexipy.oeis.lookup_total`) is already a multi-gigabyte,
    long-running enumeration if materialised, and the fully unfiltered count
    (which by :meth:`ConvexitySpace.arity` corresponds to ``N=6``, and is
    *at least* as large as every smaller-``N`` row -- e.g. for ``n=4`` the
    ``N=2``, ``N=3``, and ``N=4`` rows are ``2271``, ``2480``, ``2480``
    respectively, so arity can exceed 2) is not published by the paper and
    would be larger still. Prefer ``n_ary=1`` (``|Γ_1(X_6)| = 257{,}151``,
    tractable) and/or ``grounded_only=True`` unless you specifically need an
    unfiltered or high-arity search at ``n=6``.

    Examples
    --------
    >>> spaces = list(generate_convexities(2))
    >>> len(spaces)
    7
    >>> grounded = list(generate_convexities(2, grounded_only=True))
    >>> len(grounded)
    4
    >>> binary = list(generate_convexities(3, n_ary=1))
    >>> len(binary)
    45
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")

    if n == 0:
        space = ConvexitySpace.unchecked(frozenset(), [frozenset()])
        if n_ary is None or space.is_n_ary(n_ary):
            if on_progress is not None:
                on_progress(1, 1)
            yield space
        return

    ground_set = frozenset(range(n))
    all_subsets, subset_to_idx = _build_subset_universe(n)
    m = len(all_subsets)

    states = [_UNDECIDED] * m
    states[subset_to_idx[ground_set]] = _INCLUDED

    initial_closure: set = {ground_set}
    if grounded_only:
        empty_set = frozenset()
        states[subset_to_idx[empty_set]] = _INCLUDED
        initial_closure.add(empty_set)

    yielded = 0

    def backtrack(idx: int, current_closure: set) -> Iterator[ConvexitySpace]:
        nonlocal yielded
        if idx == m:
            families = [all_subsets[i] for i in range(m) if states[i] == _INCLUDED]
            space = ConvexitySpace.unchecked(ground_set, families)
            yielded += 1
            if on_progress is not None:
                on_progress(yielded, m)
            if n_ary is None or space.is_n_ary(n_ary):
                yield space
            return

        if states[idx] != _UNDECIDED:
            yield from backtrack(idx + 1, current_closure)
            return

        # Branch 1: exclude all_subsets[idx] from G.
        states[idx] = _EXCLUDED
        yield from backtrack(idx + 1, current_closure)
        states[idx] = _UNDECIDED

        # Branch 2: include all_subsets[idx] in G (with forced closure).
        success, newly_added = _compute_closure_incremental(
            current_closure, all_subsets[idx], states, subset_to_idx
        )
        if success:
            for i in newly_added:
                states[i] = _INCLUDED
            next_closure = current_closure | {all_subsets[i] for i in newly_added}

            yield from backtrack(idx + 1, next_closure)

            for i in newly_added:
                states[i] = _UNDECIDED

    yield from backtrack(0, initial_closure)


def count_convexities(
    n: int,
    *,
    grounded_only: bool = False,
    n_ary: Optional[int] = None,
) -> int:
    """Count valid convexities on ``{0, ..., n-1}`` without retaining them.

    Equivalent to ``sum(1 for _ in generate_convexities(n, ...))`` but does
    not hold all spaces in memory simultaneously -- only a running counter.
    Note that when ``n_ary`` is given, the *search* still explores the full
    unfiltered tree (the same as :func:`generate_convexities`); only the
    final yield/count is filtered.

    Parameters
    ----------
    n, grounded_only, n_ary:
        See :func:`generate_convexities`.

    Returns
    -------
    int
        The count.

    Examples
    --------
    >>> count_convexities(2)
    7
    >>> count_convexities(2, grounded_only=True)
    4
    >>> count_convexities(3, n_ary=1)
    45
    """
    total = 0
    for _ in generate_convexities(n, grounded_only=grounded_only, n_ary=n_ary):
        total += 1
    return total


def count_grounded_convexities(n: int, *, n_ary: Optional[int] = None) -> int:
    """Shorthand for ``count_convexities(n, grounded_only=True, n_ary=n_ary)``.

    Corresponds to :math:`|\\Gamma^0_N(X_n)|` (Table 2 of the paper) when
    ``n_ary=N`` is given, or :math:`|\\Gamma^0(X_n)|` (unfiltered grounded
    count) when ``n_ary=None``.
    """
    return count_convexities(n, grounded_only=True, n_ary=n_ary)


@dataclass
class EnumerationStats:
    """Lightweight summary statistics for an :class:`EnumerationResult`."""

    n: int
    n_ary: Optional[int]
    grounded_only: bool
    total: int
    grounded_total: int


@dataclass
class EnumerationResult:
    """Bundle of enumerated spaces plus derived statistics.

    Attributes
    ----------
    n:
        Carrier size used for this enumeration.
    n_ary:
        The ``N``-arity filter applied (``None`` if unfiltered).
    spaces:
        All :class:`~convexipy.core.ConvexitySpace` instances satisfying the
        filter, on carrier ``{0, ..., n-1}``.
    grounded_spaces:
        The subset of ``spaces`` that are grounded (``∅ ∈ G``). Always a
        subsequence of ``spaces`` -- if ``spaces`` was produced with
        ``grounded_only=True``, this equals ``spaces``.
    """

    n: int
    n_ary: Optional[int]
    spaces: List[ConvexitySpace] = field(default_factory=list)
    grounded_spaces: List[ConvexitySpace] = field(default_factory=list)

    @property
    def total(self) -> int:
        """``|Γ_N(X_n)|`` (or ``|Γ(X_n)|`` if ``n_ary is None``)."""
        return len(self.spaces)

    @property
    def grounded_total(self) -> int:
        """``|Γ^0_N(X_n)|`` (or ``|Γ^0(X_n)|`` if ``n_ary is None``)."""
        return len(self.grounded_spaces)

    def stats(self) -> EnumerationStats:
        """Return an :class:`EnumerationStats` summary of this result."""
        return EnumerationStats(
            n=self.n,
            n_ary=self.n_ary,
            grounded_only=False,
            total=self.total,
            grounded_total=self.grounded_total,
        )

    def verify_binomial_identity(self, grounded_counts: Sequence[int]) -> bool:
        """Check the binomial-transform identity (Proposition, §2 of the
        paper) for this carrier size.

        The identity states::

            |Γ_N(X_n)| = Σ_{k=0}^{n} C(n, k) · |Γ⁰_N(X_k)|

        i.e. the total ``N``-ary count on ``n`` points is the binomial
        transform of the *grounded* ``N``-ary counts for ``k = 0..n``.

        Parameters
        ----------
        grounded_counts:
            A sequence ``[|Γ⁰_N(X_0)|, |Γ⁰_N(X_1)|, ..., |Γ⁰_N(X_n)|]`` of
            length ``n + 1`` -- typically obtained by running
            :func:`enumerate_and_classify` (or
            :func:`count_grounded_convexities`) for each ``k = 0..n`` with
            the same ``n_ary`` value as this result.

        Returns
        -------
        bool
            ``True`` iff ``self.total`` equals the binomial transform of
            ``grounded_counts`` evaluated at ``self.n``.

        Examples
        --------
        >>> from convexipy.transform import binomial_transform
        >>> grounded = [count_grounded_convexities(k, n_ary=1) for k in range(4)]
        >>> result = enumerate_and_classify(3, n_ary=1)
        >>> result.verify_binomial_identity(grounded)
        True
        """
        from math import comb

        if len(grounded_counts) != self.n + 1:
            raise ValueError(
                f"grounded_counts must have length n+1={self.n + 1}, "
                f"got {len(grounded_counts)}."
            )
        expected = sum(
            comb(self.n, k) * grounded_counts[k] for k in range(self.n + 1)
        )
        return expected == self.total


def enumerate_and_classify(
    n: int,
    *,
    n_ary: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> EnumerationResult:
    """Enumerate all (optionally ``N``-ary) convexities on ``{0,...,n-1}``
    and partition them into grounded / non-grounded.

    This is the convenience entry point most callers want: a single pass
    that produces both :math:`|\\Gamma_N(X_n)|` and
    :math:`|\\Gamma^0_N(X_n)|` together (the two quantities related by the
    binomial-transform Proposition), plus the materialised spaces for
    further inspection.

    Parameters
    ----------
    n:
        Carrier size.
    n_ary:
        Optional ``N``-arity filter (see :func:`generate_convexities`).
    on_progress:
        Optional progress callback (see :func:`generate_convexities`).

    Returns
    -------
    EnumerationResult

    Warnings
    --------
    This materialises **all** matching spaces in memory. For ``n=6`` with
    no ``n_ary`` filter this is impractical (over 1.5 billion spaces); use
    :func:`count_convexities` instead if only the counts are needed, or
    apply an ``n_ary`` filter (e.g. ``n_ary=1`` yields only ``257{,}151``
    spaces for ``n=6``, which is large but tractable).

    Examples
    --------
    >>> result = enumerate_and_classify(2)
    >>> result.total, result.grounded_total
    (7, 4)
    >>> result = enumerate_and_classify(3, n_ary=1)
    >>> result.total, result.grounded_total
    (45, 29)
    """
    result = EnumerationResult(n=n, n_ary=n_ary)
    for space in generate_convexities(n, n_ary=n_ary, on_progress=on_progress):
        result.spaces.append(space)
        if space.is_grounded():
            result.grounded_spaces.append(space)
    return result
