"""
convexipy
=========

A production-ready Python library for **abstract convexity spaces**:
validation, convex hull computation, ``N``-ary classification, exhaustive
enumeration on finite carriers, and binomial-transform identities relating
general and grounded structures.

Quick start
-----------

.. code-block:: python

    from convexipy import ConvexitySpace

    X = {0, 1, 2}
    G = [{0, 1, 2}, {0, 1}, {0}, set()]
    space = ConvexitySpace(X, G)

    space.convex_hull({1})   # frozenset({0, 1})
    space.is_grounded()      # True
    space.is_n_ary(1)        # True

Module map
----------

================================  =================================================
Module                            Purpose
================================  =================================================
:mod:`convexipy.core`             :class:`ConvexitySpace`: validation, hulls,
                                   N-arity, grounding/de-grounding, serialization.
:mod:`convexipy.enumeration`      Backtracking search over P(X) to enumerate all
                                   (optionally N-ary / grounded) convexities on a
                                   finite carrier.
:mod:`convexipy.transform`        Binomial transform / inverse, and the
                                   total-vs-grounded identity from the paper.
:mod:`convexipy.oeis`             Published reference tables (Tables 1-2) and
                                   OEIS cross-references for validation/lookup.
:mod:`convexipy.applications`     Named, documented wrappers for common use cases
                                   (preorder recovery, segment/feature-space
                                   convexity, graph convexity, config validation).
:mod:`convexipy.exceptions`       Exception hierarchy.
================================  =================================================

References
----------
Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities." Kazan National Research Technical University.

Soltan, V.P. (1984). *Introduction to the Axiomatic Theory of Convexity*.

van de Vel, M.L.J. (1993). *Theory of Convex Structures*. North-Holland
Mathematical Library, vol. 50.
"""

from __future__ import annotations

from .core import ConvexitySpace, GroundingError, InvalidConvexityError
from .enumeration import (
    EnumerationResult,
    EnumerationStats,
    count_convexities,
    count_grounded_convexities,
    enumerate_and_classify,
    generate_convexities,
)
from .exceptions import ConvexipyError, EnumerationLimitError
from .transform import (
    apply_binomial_identity,
    binomial_transform,
    inverse_binomial_transform,
    is_binomial_transform_pair,
    recover_grounded_sequence,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # core
    "ConvexitySpace",
    "InvalidConvexityError",
    "GroundingError",
    # enumeration
    "generate_convexities",
    "count_convexities",
    "count_grounded_convexities",
    "enumerate_and_classify",
    "EnumerationResult",
    "EnumerationStats",
    # transform
    "binomial_transform",
    "inverse_binomial_transform",
    "apply_binomial_identity",
    "recover_grounded_sequence",
    "is_binomial_transform_pair",
    # exceptions
    "ConvexipyError",
    "EnumerationLimitError",
]
