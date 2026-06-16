"""
convexipy.transform
====================

**Binomial transform** utilities for integer sequences.

The binomial transform of a sequence :math:`(a_0, a_1, a_2, \\dots)` is the
sequence :math:`(b_0, b_1, b_2, \\dots)` defined by

.. math::

    b_n = \\sum_{k=0}^{n} \\binom{n}{k} a_k

It is an involution up to sign: the **inverse binomial transform** recovers
``a`` from ``b`` via

.. math::

    a_n = \\sum_{k=0}^{n} (-1)^{n-k} \\binom{n}{k} b_k

In the context of :mod:`convexipy.enumeration`, the central result of
Dulliev & Naumikhin (2026) is that the sequence of *total* ``N``-ary
convexity counts is exactly the binomial transform of the sequence of
*grounded* ``N``-ary convexity counts::

    |Γ_N(X_n)| = Σ_{k=0}^{n} C(n,k) · |Γ⁰_N(X_k)|

This module provides the transform pair plus convenience helpers --
:func:`apply_binomial_identity`, :func:`recover_grounded_sequence`, and
:func:`is_binomial_transform_pair` -- for working with such sequences
directly, independent of the enumeration machinery (e.g. if you only have
the published OEIS counts and want to check/recover the relationship without
re-running the combinatorial search).

References
----------
Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities."
"""

from __future__ import annotations

import math
from typing import List, Sequence

__all__ = [
    "binomial_transform",
    "inverse_binomial_transform",
    "apply_binomial_identity",
    "recover_grounded_sequence",
    "is_binomial_transform_pair",
    "binomial_transform_term",
    "inverse_binomial_transform_term",
]


def binomial_transform(sequence: Sequence[int]) -> List[int]:
    """Compute the binomial transform of ``sequence``.

    .. math::
        b_n = \\sum_{k=0}^{n} \\binom{n}{k} a_k, \\quad n = 0, \\dots, |a|-1

    Parameters
    ----------
    sequence:
        The input sequence :math:`(a_0, a_1, \\dots, a_{N-1})`.

    Returns
    -------
    list[int]
        The transformed sequence :math:`(b_0, \\dots, b_{N-1})`, the same
        length as ``sequence``.

    Examples
    --------
    >>> binomial_transform([1, 1, 4, 29, 355])
    [1, 2, 7, 45, 500]

    The above is exactly the ``N=1`` row relationship from Tables 1 & 2 of
    Dulliev & Naumikhin (2026): the grounded 1-ary counts
    :math:`|\\Gamma^0_1(X_n)|` for ``n=0..4`` are ``[1, 1, 4, 29, 355]``
    (Table 2, OEIS A000798), and their binomial transform
    ``[1, 2, 7, 45, 500]`` is exactly the total 1-ary counts
    :math:`|\\Gamma_1(X_n)|` from Table 1 (OEIS A326878). See
    :func:`apply_binomial_identity` for a thin wrapper aligned to this
    paper's naming.
    """
    transform: List[int] = []
    for n in range(len(sequence)):
        val = 0
        for k in range(n + 1):
            val += math.comb(n, k) * sequence[k]
        transform.append(val)
    return transform


def inverse_binomial_transform(sequence: Sequence[int]) -> List[int]:
    """Compute the inverse binomial transform of ``sequence``.

    .. math::
        a_n = \\sum_{k=0}^{n} (-1)^{n-k} \\binom{n}{k} b_k,
        \\quad n = 0, \\dots, |b|-1

    This is the exact inverse of :func:`binomial_transform`:
    ``inverse_binomial_transform(binomial_transform(a)) == a`` for any
    finite integer sequence ``a`` (and vice versa).

    Parameters
    ----------
    sequence:
        The transformed sequence :math:`(b_0, \\dots, b_{N-1})`.

    Returns
    -------
    list[int]
        The original sequence :math:`(a_0, \\dots, a_{N-1})`.

    Examples
    --------
    >>> inverse_binomial_transform([1, 2, 7, 45, 500])
    [1, 1, 4, 29, 355]

    This recovers the ``N=1`` *grounded* row (Table 2) from the ``N=1``
    *total* row (Table 1) of Dulliev & Naumikhin (2026) -- see
    :func:`recover_grounded_sequence` for a thin convenience wrapper with
    paper-aligned naming.
    """
    inverse: List[int] = []
    for n in range(len(sequence)):
        val = 0
        for k in range(n + 1):
            val += ((-1) ** (n - k)) * math.comb(n, k) * sequence[k]
        inverse.append(val)
    return inverse


def binomial_transform_term(sequence: Sequence[int], n: int) -> int:
    """Compute a single term :math:`b_n` of the binomial transform without
    materialising the whole transformed sequence.

    Parameters
    ----------
    sequence:
        Must have length ``>= n + 1`` (only ``a_0, ..., a_n`` are used).
    n:
        The (zero-based) term index to compute.

    Returns
    -------
    int
        :math:`b_n = \\sum_{k=0}^{n} \\binom{n}{k} a_k`.

    Raises
    ------
    ValueError
        If ``n < 0`` or ``len(sequence) < n + 1``.

    Examples
    --------
    >>> binomial_transform_term([1, 1, 4, 29, 355], 4)
    500
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if len(sequence) < n + 1:
        raise ValueError(
            f"sequence must have at least {n + 1} elements to compute "
            f"term n={n}, got {len(sequence)}."
        )
    return sum(math.comb(n, k) * sequence[k] for k in range(n + 1))


def inverse_binomial_transform_term(sequence: Sequence[int], n: int) -> int:
    """Compute a single term :math:`a_n` of the inverse binomial transform.

    Parameters
    ----------
    sequence:
        Must have length ``>= n + 1`` (only ``b_0, ..., b_n`` are used).
    n:
        The (zero-based) term index to compute.

    Returns
    -------
    int
        :math:`a_n = \\sum_{k=0}^{n} (-1)^{n-k} \\binom{n}{k} b_k`.

    Raises
    ------
    ValueError
        If ``n < 0`` or ``len(sequence) < n + 1``.

    Examples
    --------
    >>> inverse_binomial_transform_term([1, 2, 7, 45, 500], 4)
    355
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if len(sequence) < n + 1:
        raise ValueError(
            f"sequence must have at least {n + 1} elements to compute "
            f"term n={n}, got {len(sequence)}."
        )
    return sum(
        ((-1) ** (n - k)) * math.comb(n, k) * sequence[k] for k in range(n + 1)
    )


def is_binomial_transform_pair(
    grounded: Sequence[int], total: Sequence[int]
) -> bool:
    """Check whether ``total`` is the binomial transform of ``grounded``.

    This is a direct executable check of the Proposition (§2) of
    Dulliev & Naumikhin (2026):

    .. math::
        |\\Gamma_N(X_n)| = \\sum_{k=0}^{n} \\binom{n}{k} |\\Gamma^0_N(X_k)|

    Parameters
    ----------
    grounded:
        The grounded-count sequence :math:`|\\Gamma^0_N(X_0)|, \\dots`.
    total:
        The total-count sequence :math:`|\\Gamma_N(X_0)|, \\dots`, expected
        to have the **same length** as ``grounded``.

    Returns
    -------
    bool
        ``True`` iff ``total == binomial_transform(grounded)`` (after
        truncating/comparing only over the common length -- both sequences
        must have equal length, or ``False`` is returned).

    Examples
    --------
    >>> grounded_N1 = [1, 1, 4, 29, 355, 6942]
    >>> total_N1    = [1, 2, 7, 45, 500, 9053]
    >>> is_binomial_transform_pair(grounded_N1, total_N1)
    True
    """
    if len(grounded) != len(total):
        return False
    return binomial_transform(grounded) == list(total)


def apply_binomial_identity(grounded_sequence: Sequence[int], n: int) -> int:
    """Compute :math:`|\\Gamma_N(X_n)|` from a sequence of grounded counts.

    Thin, paper-aligned wrapper around :func:`binomial_transform_term`:
    given :math:`|\\Gamma^0_N(X_0)|, \\dots, |\\Gamma^0_N(X_n)|`, returns
    :math:`|\\Gamma_N(X_n)|` via the Proposition of §2.

    Parameters
    ----------
    grounded_sequence:
        :math:`(|\\Gamma^0_N(X_0)|, \\dots, |\\Gamma^0_N(X_n)|)`, length
        ``n + 1``.
    n:
        The index at which to evaluate the total count.

    Returns
    -------
    int
        :math:`|\\Gamma_N(X_n)|`.

    Examples
    --------
    >>> grounded_N1 = [1, 1, 4, 29, 355]  # |Gamma^0_1(X_k)| for k=0..4
    >>> apply_binomial_identity(grounded_N1, 4)
    500
    """
    return binomial_transform_term(grounded_sequence, n)


def recover_grounded_sequence(total_sequence: Sequence[int]) -> List[int]:
    """Recover the grounded-count sequence from the total-count sequence.

    Paper-aligned wrapper around :func:`inverse_binomial_transform`: given
    :math:`(|\\Gamma_N(X_0)|, \\dots, |\\Gamma_N(X_n)|)`, returns
    :math:`(|\\Gamma^0_N(X_0)|, \\dots, |\\Gamma^0_N(X_n)|)`.

    Useful when only the *total* counts are known (e.g. read off an OEIS
    entry) and the *grounded* counts are desired without re-running
    enumeration.

    Parameters
    ----------
    total_sequence:
        :math:`(|\\Gamma_N(X_0)|, \\dots, |\\Gamma_N(X_n)|)`.

    Returns
    -------
    list[int]
        :math:`(|\\Gamma^0_N(X_0)|, \\dots, |\\Gamma^0_N(X_n)|)`.

    Examples
    --------
    >>> total_N1 = [1, 2, 7, 45, 500, 9053]
    >>> recover_grounded_sequence(total_N1)
    [1, 1, 4, 29, 355, 6942]
    """
    return inverse_binomial_transform(total_sequence)
