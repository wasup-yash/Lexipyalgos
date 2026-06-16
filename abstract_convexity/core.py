"""
convexipy.core
==============

Core domain model for **abstract convexity spaces**.

A *convexity space* is a pair ``(X, G)`` where ``X`` is a finite ground set and
``G`` (the *convexity*) is a family of subsets of ``X`` that:

1. contains ``X`` itself, and
2. is closed under arbitrary intersection (the intersection of any
   sub-collection of members of ``G`` is again a member of ``G``).

Members of ``G`` are called *convex sets*. Every convexity space induces a
*convex hull operator* ``g`` satisfying the classical Kuratowski-style closure
axioms:

* ``A ⊆ g(A)``               (extensivity)
* ``A ⊆ B ⇒ g(A) ⊆ g(B)``     (monotonicity)
* ``g(g(A)) = g(A)``          (idempotence)

This module provides :class:`ConvexitySpace`, a fully validated, hashable,
immutable representation of such a structure, along with hull operators,
``N``-ary closure testing, and the *grounding* / *de-grounding* transforms
(Lemmas 1-5 of Dulliev & Naumikhin, 2026) that relate a convexity to its
*grounded reflection*.

References
----------
Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities." Department of Applied Mathematics and Informatics, Kazan
National Research Technical University.

Soltan, V.P. (1984). *Introduction to the Axiomatic Theory of Convexity*.
Kishinev: Shtiinca.

van de Vel, M.L.J. (1993). *Theory of Convex Structures*. North-Holland
Mathematical Library, vol. 50.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .exceptions import GroundingError, InvalidConvexityError

__all__ = [
    "ConvexitySpace",
    "InvalidConvexityError",
    "GroundingError",
    "Family",
    "GroundSetLike",
]

#: A "family-like" input -- any iterable of iterables of hashable elements.
Family = Iterable[Iterable[Any]]

#: A "ground-set-like" input -- any iterable of hashable elements.
GroundSetLike = Iterable[Any]


# `InvalidConvexityError` and `GroundingError` are defined in
# `convexipy.exceptions` (the single canonical location for the package's
# exception hierarchy) and re-exported here under their historical names
# so that `convexipy.core.InvalidConvexityError` and
# `convexipy.core.GroundingError` continue to work and refer to the *same*
# classes as `convexipy.exceptions.InvalidConvexityError` /
# `convexipy.exceptions.GroundingError` and `convexipy.InvalidConvexityError`
# / `convexipy.GroundingError`. See `convexipy.exceptions` for the full
# docstrings.


@dataclass(frozen=True, slots=True)
class _ValidationConfig:
    """Internal knobs controlling how strictly :class:`ConvexitySpace`
    validates its inputs. Exposed for advanced/perf-sensitive callers via
    :meth:`ConvexitySpace.unchecked`.
    """

    validate: bool = True


class ConvexitySpace:
    """An immutable, hashable abstract convexity space ``(X, G)``.

    Parameters
    ----------
    ground_set:
        Any iterable of hashable elements; coerced to a :class:`frozenset`.
    families:
        Any iterable of iterables of hashable elements; each member is
        coerced to a :class:`frozenset` and the whole collection to a
        :class:`frozenset` of frozensets (i.e. duplicates are removed).
    validate:
        If ``True`` (default), the convexity axioms are checked eagerly at
        construction time and :class:`InvalidConvexityError` is raised on
        failure. Set to ``False`` only when you can *guarantee* the input is
        already a valid convexity (e.g. it was produced by
        :func:`convexipy.enumeration.generate_convexities`), to skip the
        ``O(|G|^2)`` pairwise-intersection check.

    Raises
    ------
    InvalidConvexityError
        If ``validate=True`` and the axioms are violated.

    Examples
    --------
    >>> X = {0, 1, 2}
    >>> G = [{0, 1, 2}, {0, 1}, {0}, set()]
    >>> space = ConvexitySpace(X, G)
    >>> space.convex_hull({1})
    frozenset({0, 1})
    >>> space.is_grounded()
    True
    """

    __slots__ = ("ground_set", "G", "_hull_cache")

    ground_set: FrozenSet[Any]
    G: FrozenSet[FrozenSet[Any]]

    def __init__(
        self,
        ground_set: GroundSetLike,
        families: Family,
        *,
        validate: bool = True,
    ) -> None:
        gs = frozenset(ground_set)
        fam = frozenset(frozenset(s) for s in families)

        object.__setattr__(self, "ground_set", gs)
        object.__setattr__(self, "G", fam)
        # Per-instance hull memoisation. Safe because the instance is
        # immutable -- g(A) never changes for this (X, G).
        object.__setattr__(self, "_hull_cache", {})

        if validate:
            self._validate()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def unchecked(
        cls, ground_set: GroundSetLike, families: Family
    ) -> "ConvexitySpace":
        """Construct a :class:`ConvexitySpace` *without* axiom validation.

        Use this only when ``families`` is already known to satisfy the
        convexity axioms (e.g. output of :mod:`convexipy.enumeration`).
        Skipping validation avoids the ``O(|G|^2)`` pairwise-intersection
        scan, which matters when iterating over thousands of candidate
        spaces.

        .. warning::
           No checks are performed. Passing an invalid family will not raise
           immediately, but downstream methods (hulls, N-arity, grounding)
           may silently return incorrect results.
        """
        return cls(ground_set, families, validate=False)

    @classmethod
    def discrete(cls, ground_set: GroundSetLike) -> "ConvexitySpace":
        """The *discrete* convexity: every subset of ``X`` is convex.

        ``G = P(X)``, the full power set. This is the largest possible
        convexity on ``X`` (and is both grounded and ``N``-ary for every
        ``N``).
        """
        gs = frozenset(ground_set)
        elements = list(gs)
        families = (
            frozenset(c)
            for r in range(len(elements) + 1)
            for c in itertools.combinations(elements, r)
        )
        return cls.unchecked(gs, families)

    @classmethod
    def trivial(cls, ground_set: GroundSetLike) -> "ConvexitySpace":
        """The *trivial* (indiscrete) convexity: ``G = {X}`` only.

        This is the smallest possible convexity on ``X``. It is grounded
        only when ``X = ∅``.
        """
        gs = frozenset(ground_set)
        return cls.unchecked(gs, [gs])

    @classmethod
    def from_closure_operator(
        cls,
        ground_set: GroundSetLike,
        hull_fn: "Callable[[FrozenSet[Any]], AbstractSet[Any]]",
        *,
        validate: bool = True,
    ) -> "ConvexitySpace":
        """Build a :class:`ConvexitySpace` from a closure operator ``g``.

        ``G`` is recovered as ``{A ⊆ X : g(A) = A}``, i.e. the fixed points
        of ``hull_fn``. This is the inverse direction of
        :meth:`convex_hull`: given any monotone, extensive, idempotent
        operator on subsets of ``X``, this reconstructs the corresponding
        convexity family.

        Parameters
        ----------
        ground_set:
            The carrier set ``X``.
        hull_fn:
            A callable ``frozenset -> set-like`` implementing ``g``. It is
            evaluated on **every** subset of ``X``, so this constructor is
            only practical for small ``|X|`` (roughly ``|X| <= 18`` on
            commodity hardware due to the ``2^|X|`` enumeration).
        validate:
            Whether to validate the recovered family (default ``True``).
            If ``hull_fn`` does not actually satisfy the closure axioms, the
            recovered fixed-point family may fail to be a valid convexity,
            and this will surface that as :class:`InvalidConvexityError`.

        Examples
        --------
        >>> X = {0, 1, 2}
        >>> def g(A):
        ...     return frozenset(X) if len(A) >= 2 else frozenset(A)
        >>> space = ConvexitySpace.from_closure_operator(X, g)
        >>> sorted(sorted(s) for s in space.G)
        [[], [0], [0, 1, 2], [1], [2]]
        """
        gs = frozenset(ground_set)
        elements = list(gs)
        fixed_points = []
        for r in range(len(elements) + 1):
            for combo in itertools.combinations(elements, r):
                A = frozenset(combo)
                if frozenset(hull_fn(A)) == A:
                    fixed_points.append(A)
        return cls(gs, fixed_points, validate=validate)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate the convexity axioms.

        Checks (in order, failing fast):

        1. ``X ∈ G``
        2. every ``A ∈ G`` is a subset of ``X``
        3. for all ``A, B ∈ G``, ``A ∩ B ∈ G``

        Raises
        ------
        InvalidConvexityError
            On the first violated axiom, with a message identifying the
            offending set(s).
        """
        if self.ground_set not in self.G:
            raise InvalidConvexityError(
                "The ground set X must be an element of the convexity "
                "family G."
            )
        g_list = list(self.G)
        for A in g_list:
            if not A.issubset(self.ground_set):
                raise InvalidConvexityError(
                    f"Element {set(A)} in G is not a subset of the ground "
                    f"set X = {set(self.ground_set)}."
                )
        for A in g_list:
            for B in g_list:
                inter = A & B
                if inter not in self.G:
                    raise InvalidConvexityError(
                        "G is not closed under intersection: "
                        f"{set(A)} ∩ {set(B)} = {set(inter)}, which is "
                        "missing from G."
                    )

    def is_valid(self) -> bool:
        """Return ``True`` iff this instance currently satisfies the
        convexity axioms.

        Unlike :meth:`_validate`, this never raises -- it returns a boolean.
        Primarily useful after using :meth:`unchecked` /
        :meth:`from_closure_operator(validate=False)` to confirm validity
        out-of-band, e.g. in a test suite or sanity-check pass.
        """
        try:
            self._validate()
        except InvalidConvexityError:
            return False
        return True

    # ------------------------------------------------------------------
    # Hull operator
    # ------------------------------------------------------------------

    def convex_hull(self, A: Union[AbstractSet[Any], Sequence[Any]]) -> FrozenSet[Any]:
        """Compute the convex hull ``g(A)`` of a subset ``A ⊆ X``.

        ``g(A)`` is the intersection of every convex set containing ``A``::

            g(A) = ⋂ { B ∈ G : A ⊆ B }

        If no member of ``G`` is empty-intersection-compatible (impossible
        for a valid convexity, since ``X ∈ G`` always qualifies), the result
        defaults to ``X``.

        Results are memoised per-instance, so repeated calls with the same
        (hashable) ``A`` are O(1) after the first.

        Parameters
        ----------
        A:
            Any set-like or sequence of elements drawn from ``X``.

        Returns
        -------
        FrozenSet[Any]
            ``g(A)``, the smallest convex set containing ``A``.

        Raises
        ------
        ValueError
            If ``A`` is not a subset of the ground set ``X``.

        Examples
        --------
        >>> X = {0, 1, 2}
        >>> G = [{0, 1, 2}, {0, 1}, {0}, set()]
        >>> space = ConvexitySpace(X, G)
        >>> space.convex_hull([])
        frozenset()
        >>> space.convex_hull({1})
        frozenset({0, 1})
        >>> space.convex_hull({2})
        frozenset({0, 1, 2})
        """
        A_fs = frozenset(A)
        if not A_fs.issubset(self.ground_set):
            raise ValueError(
                f"The subset {set(A_fs)} is not a subset of the ground "
                f"set X = {set(self.ground_set)}."
            )

        cache: Dict[FrozenSet[Any], FrozenSet[Any]] = self._hull_cache
        cached = cache.get(A_fs)
        if cached is not None:
            return cached

        result: Optional[FrozenSet[Any]] = None
        for B in self.G:
            if A_fs.issubset(B):
                result = B if result is None else (result & B)

        if result is None:
            # Unreachable for a *valid* convexity (X always qualifies), but
            # kept as a safe fallback for `unchecked` instances.
            result = self.ground_set

        cache[A_fs] = result
        return result

    def is_convex(self, A: Union[AbstractSet[Any], Sequence[Any]]) -> bool:
        """Return ``True`` iff ``A`` is itself a convex set, i.e. ``A ∈ G``
        (equivalently, ``g(A) = A``).
        """
        A_fs = frozenset(A)
        return A_fs in self.G

    # ------------------------------------------------------------------
    # Grounding-related queries
    # ------------------------------------------------------------------

    def is_grounded(self) -> bool:
        """Return ``True`` iff ``G`` satisfies the *grounding axiom*
        ``∅ ∈ G``.

        Equivalently, ``True`` iff the minimal convex set
        ``g(∅) = ∅``.
        """
        return frozenset() in self.G

    def minimal_convex_set(self) -> FrozenSet[Any]:
        """Return ``C = g(∅)``, the (unique) minimal convex set.

        ``C`` is contained in every convex set (it is the intersection of
        all of ``G``) and equals ``∅`` iff the space is grounded.
        """
        return self.convex_hull(frozenset())

    def get_grounded_reflection(self) -> "ConvexitySpace":
        """Construct the *grounded reflection* ``H`` on ``Y = X \\ C``.

        Per **Lemma 1** of Dulliev & Naumikhin (2026), with
        ``C = g(∅)`` and ``H = { A \\ C : A ∈ G }``, the map
        ``A ↦ A \\ C`` is a bijection ``G → H``, and (per **Lemma 2**)
        ``H`` is a *grounded* convexity on ``Y = X \\ C``. Per **Lemma 4**,
        if ``G`` is ``N``-ary then so is ``H``.

        Returns
        -------
        ConvexitySpace
            The grounded convexity ``(Y, H)``.

        See Also
        --------
        from_grounded : the inverse construction (Lemma 5).

        Examples
        --------
        >>> X = {0, 1, 2}
        >>> G = [{0, 1, 2}, {0, 1}, {0}]   # g(empty) = {0}
        >>> space = ConvexitySpace(X, G)
        >>> refl = space.get_grounded_reflection()
        >>> refl.ground_set
        frozenset({1, 2})
        >>> refl.is_grounded()
        True
        """
        C = self.minimal_convex_set()
        Y = self.ground_set - C
        H_families = (A - C for A in self.G)
        return ConvexitySpace.unchecked(Y, H_families)

    @classmethod
    def from_grounded(
        cls, H: "ConvexitySpace", C: GroundSetLike, *, validate: bool = True
    ) -> "ConvexitySpace":
        """Construct ``G`` on ``X = Y ∪ C`` from a grounded convexity
        ``H`` on ``Y``, per **Lemma 5**.

        Given a grounded convexity ``H`` on ``Y`` (``∅ ∈ H``) and a set
        ``C`` disjoint from ``Y``, the family
        ``G = { C ∪ D : D ∈ H }`` is a convexity on ``X = Y ∪ C`` with
        ``g_G(∅) = C``, and if ``H`` is ``N``-ary then so is ``G``. This is
        the bijective inverse of :meth:`get_grounded_reflection`.

        Parameters
        ----------
        H:
            A grounded :class:`ConvexitySpace` (``H.is_grounded()`` must be
            ``True``).
        C:
            The minimal-set "prefix" to attach. Must be disjoint from
            ``H.ground_set``.
        validate:
            Whether to validate the resulting convexity (default ``True``).

        Returns
        -------
        ConvexitySpace
            The convexity ``(X, G)`` with ``X = Y ∪ C`` and
            ``g_G(∅) = C``.

        Raises
        ------
        GroundingError
            If ``H`` is not grounded, or if ``C`` overlaps ``H.ground_set``.

        Examples
        --------
        >>> Y = {1, 2}
        >>> H = ConvexitySpace(Y, [{1, 2}, set()])
        >>> space = ConvexitySpace.from_grounded(H, {0})
        >>> space.minimal_convex_set()
        frozenset({0})
        >>> space.ground_set == {0, 1, 2}
        True
        """
        C_fs = frozenset(C)
        if not H.is_grounded():
            raise GroundingError(
                "The provided convexity H must be grounded (∅ ∈ H) to "
                "apply the de-grounding construction of Lemma 5."
            )
        if H.ground_set & C_fs:
            raise GroundingError(
                "The minimal convex set C must be disjoint from the "
                f"ground set of H. Overlap: {set(H.ground_set & C_fs)}."
            )

        X = H.ground_set | C_fs
        G_families = (C_fs | D for D in H.G)
        return cls(X, G_families, validate=validate)

    # ------------------------------------------------------------------
    # N-arity
    # ------------------------------------------------------------------

    def is_n_ary(self, n: int) -> bool:
        """Check whether this convexity is ``N``-ary.

        A convexity ``G`` is *N-ary* iff for every ``A ⊆ X``::

            A ∈ G  ⟺  ∀ B ⊆ A : |B| ≤ N ⇒ g(B) ⊆ A

        Intuitively: convexity of a set is fully determined by the hulls of
        its size-``≤ N`` subsets. ``N=0`` corresponds to the empty subset
        only (so ``G`` is 0-ary iff ``G = P(X)``, the discrete convexity, or
        ``g(∅) = X``... in general very restrictive). ``N=1`` corresponds to
        the classical notion where hulls of singletons (and the empty set)
        determine convexity. Larger ``N`` is a weaker (more permissive)
        condition -- every convexity on a finite ``X`` is ``N``-ary for
        ``N ≥ |X|``.

        Parameters
        ----------
        n:
            The arity to test. Must be ``>= 0``.

        Returns
        -------
        bool
            ``True`` iff the ``N``-ary closure condition holds for every
            subset of ``X``.

        Raises
        ------
        ValueError
            If ``n < 0``.

        Notes
        -----
        Complexity is ``O(2^|X|)`` in the worst case (it inspects every
        subset of ``X``), with hull computations memoised via
        :meth:`convex_hull`. This is appropriate for the small carriers
        (``|X| <= ~10``) typical of enumeration studies; for larger ``X``,
        prefer :func:`convexipy.enumeration.generate_convexities`, which
        builds ``N``-ary spaces by construction rather than checking after
        the fact.

        Examples
        --------
        >>> X = {0, 1}
        >>> space = ConvexitySpace(X, [{0, 1}, {0}, set()])
        >>> space.is_n_ary(1)
        True
        """
        if n < 0:
            raise ValueError("Arity N must be a non-negative integer.")

        elements = list(self.ground_set)
        n_eff = min(n, len(elements))

        # Precompute g(B) for every B with |B| <= n.
        small_hulls: List[Tuple[FrozenSet[Any], FrozenSet[Any]]] = []
        for k in range(n_eff + 1):
            for B_tuple in itertools.combinations(elements, k):
                B = frozenset(B_tuple)
                small_hulls.append((B, self.convex_hull(B)))

        # For every A subset of X, check the N-ary biconditional.
        for r in range(len(elements) + 1):
            for A_tuple in itertools.combinations(elements, r):
                A = frozenset(A_tuple)
                small_hulls_contained = all(
                    g_B.issubset(A) for B, g_B in small_hulls if B.issubset(A)
                )
                if (A in self.G) != small_hulls_contained:
                    return False
        return True

    def arity(self, *, max_n: Optional[int] = None) -> int:
        """Return the **minimal** ``N`` for which this convexity is
        ``N``-ary.

        Every convexity on a finite carrier ``X`` is ``N``-ary for
        ``N = |X|`` (trivially, since the only ``B ⊆ A`` with
        ``|B| ≤ |X|`` and ``B ⊆ A`` ranges over all subsets of ``A``, so the
        condition reduces to ``A ∈ G ⟺ g(A) ⊆ A``, which always holds).
        This method searches ``N = 0, 1, ..., max_n`` (default
        ``max_n = |X|``) and returns the first ``N`` that passes
        :meth:`is_n_ary`.

        Parameters
        ----------
        max_n:
            Upper bound on the search (default ``len(self.ground_set)``,
            which is guaranteed sufficient).

        Returns
        -------
        int
            The minimal ``N`` such that ``is_n_ary(N)`` is ``True``.

        Examples
        --------
        >>> X = {0, 1, 2}
        >>> space = ConvexitySpace(X, [{0, 1, 2}, {0, 1}, {0}, set()])
        >>> space.arity()
        1
        """
        upper = len(self.ground_set) if max_n is None else max_n
        for n in range(upper + 1):
            if self.is_n_ary(n):
                return n
        return upper

    # ------------------------------------------------------------------
    # Set-theoretic structure of G
    # ------------------------------------------------------------------

    def maximal_proper_convex_sets(self) -> FrozenSet[FrozenSet[Any]]:
        """Return the convex sets that are maximal among ``G \\ {X}``.

        These are the "co-atoms" of the convexity lattice ``G`` ordered by
        inclusion, i.e. the convex sets covered only by ``X`` itself.
        """
        proper = [A for A in self.G if A != self.ground_set]
        return frozenset(
            A for A in proper if not any(A < B for B in proper)
        )

    def minimal_nonempty_convex_sets(self) -> FrozenSet[FrozenSet[Any]]:
        """Return the convex sets that are minimal among ``G \\ {∅}``.

        These are the "atoms" of the convexity lattice ``G``. If the space
        is not grounded, this returns the minimal elements of ``G`` itself
        (which all equal the minimal convex set ``C = g(∅)`` when ``G`` has
        a unique minimum, or the minimal elements above it otherwise).
        """
        nonempty = [A for A in self.G if A != frozenset()]
        return frozenset(
            A for A in nonempty if not any(B < A for B in nonempty)
        )

    def convex_sets_sorted(self) -> List[FrozenSet[Any]]:
        """Return ``G`` as a list sorted by ``(|A|, sorted(A))`` ascending.

        Convenient for deterministic display/serialisation.
        """
        return sorted(self.G, key=lambda s: (len(s), sorted(map(repr, s))))

    # ------------------------------------------------------------------
    # Sub/quotient spaces
    # ------------------------------------------------------------------

    def restrict_to(self, subset: GroundSetLike) -> "ConvexitySpace":
        """Return the *induced subspace convexity* on ``subset ∩ X``.

        The induced family is ``{ A ∩ subset : A ∈ G }``. This is always a
        valid convexity on ``subset ∩ X`` (intersection with a fixed set
        preserves closure under intersection and contains the new ground
        set).

        Parameters
        ----------
        subset:
            Any iterable of elements; intersected with ``X`` to form the new
            ground set.

        Examples
        --------
        >>> X = {0, 1, 2}
        >>> space = ConvexitySpace(X, [{0, 1, 2}, {0, 1}, {0}, set()])
        >>> sub = space.restrict_to({0, 1})
        >>> sub.ground_set
        frozenset({0, 1})
        """
        S = frozenset(subset) & self.ground_set
        induced = (A & S for A in self.G)
        return ConvexitySpace.unchecked(S, induced)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly ``dict``.

        Ground-set elements and family members are converted to sorted
        lists. Note this requires elements to be JSON-serialisable and
        (for the sort) mutually comparable; for non-orderable element
        types, sort by ``repr`` first or use :meth:`to_dict` only for
        ``int``/``str`` carriers (the common case for enumeration studies).

        Returns
        -------
        dict
            ``{"ground_set": [...], "families": [[...], ...]}``.

        See Also
        --------
        from_dict : the inverse operation.
        to_json, from_json : string-based convenience wrappers.
        """
        try:
            gs = sorted(self.ground_set)
        except TypeError:
            gs = sorted(self.ground_set, key=repr)

        families: List[List[Any]] = []
        for A in self.convex_sets_sorted():
            try:
                families.append(sorted(A))
            except TypeError:
                families.append(sorted(A, key=repr))

        return {"ground_set": gs, "families": families}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, validate: bool = True) -> "ConvexitySpace":
        """Construct a :class:`ConvexitySpace` from the output of
        :meth:`to_dict`.

        Parameters
        ----------
        data:
            A mapping with keys ``"ground_set"`` (iterable) and
            ``"families"`` (iterable of iterables).
        validate:
            Whether to validate the convexity axioms (default ``True``).
        """
        return cls(data["ground_set"], data["families"], validate=validate)

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string via :meth:`to_dict`.

        Any keyword arguments are forwarded to :func:`json.dumps`.
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str, *, validate: bool = True) -> "ConvexitySpace":
        """Construct a :class:`ConvexitySpace` from a JSON string produced
        by :meth:`to_json`.
        """
        return cls.from_dict(json.loads(s), validate=validate)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        gs = set(self.ground_set)
        fam = [set(s) for s in self.convex_sets_sorted()]
        return f"ConvexitySpace(ground_set={gs!r}, families={fam!r})"

    def __str__(self) -> str:
        n = len(self.ground_set)
        m = len(self.G)
        grounded = "grounded" if self.is_grounded() else "non-grounded"
        return f"<ConvexitySpace |X|={n}, |G|={m}, {grounded}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConvexitySpace):
            return NotImplemented
        return self.ground_set == other.ground_set and self.G == other.G

    def __hash__(self) -> int:
        return hash((self.ground_set, self.G))

    def __len__(self) -> int:
        """Return ``|G|``, the number of convex sets."""
        return len(self.G)

    def __iter__(self) -> Iterator[FrozenSet[Any]]:
        """Iterate over the convex sets in ``G``, sorted deterministically."""
        return iter(self.convex_sets_sorted())

    def __contains__(self, item: object) -> bool:
        """``A in space`` <=> ``A`` (coerced to ``frozenset``) is in ``G``."""
        try:
            return frozenset(item) in self.G  # type: ignore[arg-type]
        except TypeError:
            return False


# Avoid a hard import-time dependency on `Callable` typing import order
# issues with `from __future__ import annotations`.
from typing import Callable  # noqa: E402  (kept at bottom intentionally)
