"""
convexipy.exceptions
=====================

Central exception hierarchy for :mod:`convexipy`.

All exceptions raised by this package derive from :class:`ConvexipyError`,
so callers can do::

    try:
        ...
    except ConvexipyError:
        ...

to catch any domain-specific failure while letting unrelated exceptions
(``TypeError``, ``KeyError`` from genuinely malformed input, etc.) propagate
normally.
"""

from __future__ import annotations

__all__ = [
    "ConvexipyError",
    "InvalidConvexityError",
    "GroundingError",
    "EnumerationLimitError",
]


class ConvexipyError(Exception):
    """Base class for all exceptions raised by :mod:`convexipy`."""


class InvalidConvexityError(ConvexipyError, ValueError):
    """Raised when a candidate family ``G`` violates the convexity axioms.

    A family ``G`` over ground set ``X`` is a valid convexity iff:

    * ``X ∈ G``
    * every ``A ∈ G`` satisfies ``A ⊆ X``
    * for all ``A, B ∈ G``, ``A ∩ B ∈ G``  (closure under pairwise, hence
      arbitrary, intersection)

    This is the canonical definition, raised by
    :meth:`convexipy.core.ConvexitySpace._validate` and re-exported as
    ``convexipy.core.InvalidConvexityError`` and
    ``convexipy.InvalidConvexityError`` for convenience -- all three names
    refer to the *same* class, so ``except`` clauses written against any of
    them are interchangeable.
    """


class GroundingError(ConvexipyError, ValueError):
    """Raised by grounding/de-grounding operations when preconditions fail.

    Examples of preconditions that raise this error:

    * calling :meth:`convexipy.core.ConvexitySpace.from_grounded` with an
      ``H`` that is not itself grounded (``∅ ∉ H``);
    * supplying a minimal-set ``C`` that overlaps the ground set of ``H``.

    This is the canonical definition, also re-exported as
    ``convexipy.core.GroundingError`` and ``convexipy.GroundingError``.
    """


class EnumerationLimitError(ConvexipyError):
    """Raised when an enumeration request exceeds a configured safety limit.

    :mod:`convexipy.enumeration` itself never raises this (it has no
    built-in limits, by design, so that advanced users can run arbitrarily
    large searches). It is raised by higher-level convenience wrappers --
    e.g. :func:`convexipy.applications.safe_enumerate` -- that impose a
    default ``max_n`` guard to prevent accidental multi-hour/-gigabyte
    computations from a casual API call.
    """

    def __init__(self, n: int, limit: int, *, hint: str = "") -> None:
        msg = (
            f"Requested carrier size n={n} exceeds the configured safety "
            f"limit of {limit}. Full unfiltered enumeration grows "
            f"super-exponentially (see convexipy.enumeration module docs)."
        )
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.n = n
        self.limit = limit
