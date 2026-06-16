"""
convexipy.oeis
==============

Reference integer sequences from Dulliev & Naumikhin (2026), Tables 1 & 2,
cross-referenced against the `OEIS <https://oeis.org>`_ (Online Encyclopedia
of Integer Sequences).

These tables are the **ground truth** this library's enumeration algorithms
are validated against (see ``tests/test_paper_tables.py``), and are exposed
here as plain Python data so that:

* downstream code can sanity-check enumeration results without re-running
  the (super-exponential) search,
* applications can look up "how many ``N``-ary convexities exist on
  ``n`` points" for small ``n`` as an ``O(1)`` table lookup,
* the OEIS cross-references are discoverable programmatically.

Data
----
:data:`TOTAL_NARY_COUNTS` and :data:`GROUNDED_NARY_COUNTS` mirror Tables 1
and 2 of the paper: outer key is the arity ``N`` (``0`` through ``4``),
inner value is a tuple ``(|Γ_N(X_0)|, ..., |Γ_N(X_n)|)`` (resp. the grounded
analogue), for ``n = 0..4``.

:data:`OEIS_REFERENCES` maps ``(N, grounded: bool)`` to the OEIS sequence ID
quoted in the paper, where known (``"-"`` in the paper means "not yet in
OEIS" and is represented here as ``None``).

The paper additionally reports, for ``N=1`` (binary convexities) and ``N=2``,
partial data up to ``n=6``; these extra terms are recorded in
:data:`EXTENDED_TERMS` and are the values used by the slower ``n=5,6``
entries in the test suite.

References
----------
Dulliev, A., Naumikhin, D. (2026). "Binomial Transform of Sequences Counting
N-ary Convexities," Tables 1-2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = [
    "TOTAL_NARY_COUNTS",
    "GROUNDED_NARY_COUNTS",
    "OEIS_REFERENCES",
    "EXTENDED_TERMS",
    "oeis_url",
    "lookup_total",
    "lookup_grounded",
]


#: Table 1: |Γ_N(X_n)| for N=0..4, n=0..4 (rows exactly as printed in the paper).
TOTAL_NARY_COUNTS: Dict[int, Tuple[int, ...]] = {
    0: (1, 2, 4, 8, 16),
    1: (1, 2, 7, 45, 500),
    2: (1, 2, 7, 61, 2271),
    3: (1, 2, 7, 61, 2480),
    4: (1, 2, 7, 61, 2480),
}

#: Table 2: |Γ⁰_N(X_n)| for N=0..4, n=0..4 (rows exactly as printed in the paper).
GROUNDED_NARY_COUNTS: Dict[int, Tuple[int, ...]] = {
    0: (1, 1, 1, 1, 1),
    1: (1, 1, 4, 29, 355),
    2: (1, 1, 4, 45, 2062),
    3: (1, 1, 4, 45, 2271),
    4: (1, 1, 4, 45, 2271),
}

#: Extended terms beyond n=4, where the paper reports them (n=5, n=6 for
#: select rows of Tables 1-2). Keyed identically to TOTAL_NARY_COUNTS /
#: GROUNDED_NARY_COUNTS but only for the (N, n) pairs explicitly given.
#: Structure: {(N, grounded): {n: value}}.
EXTENDED_TERMS: Dict[Tuple[int, bool], Dict[int, int]] = {
    (1, False): {5: 9053, 6: 257151},
    (1, True): {5: 6942, 6: 209527},
    (2, False): {5: 600408, 6: 1556743050},
    (2, True): {5: 589602, 6: 1553173541},
    (3, False): {5: 1373701},
    (3, True): {5: 1361850},
    (4, False): {5: 1385552},
    (4, True): {5: 1373701},
}

#: OEIS sequence IDs as cross-referenced in Tables 1-2. ``None`` indicates
#: the paper marks this row as not (yet) present in OEIS ("-").
#: Keyed by (N, grounded).
OEIS_REFERENCES: Dict[Tuple[int, bool], Optional[str]] = {
    (0, False): "A000079",
    (0, True): "A000012",
    (1, False): "A326878",
    (1, True): "A000798",
    (2, False): None,
    (2, True): "A364656",
    (3, False): None,
    (3, True): "A395658",
    (4, False): None,
    (4, True): None,
}

#: Human-readable description of what the well-known OEIS rows count, for
#: convenience when presenting results to end users.
OEIS_DESCRIPTIONS: Dict[str, str] = {
    "A000079": "Powers of 2 (|G| = 2^n for the discrete convexity P(X_n)).",
    "A000012": "The all-ones sequence (the trivial convexity {X} is the "
    "unique grounded 0-ary structure for every n).",
    "A326878": "Total number of 1-ary (binary) convexities on n points.",
    "A000798": "Number of quasi-orders / preorders / finite topologies on "
    "n labeled points -- equals the number of grounded 1-ary "
    "convexities, since a grounded binary convexity is exactly "
    "the family of up-sets of a preorder.",
    "A364656": "Number of grounded 2-ary convexities on n points.",
    "A395658": "Number of grounded 3-ary convexities on n points.",
}


def oeis_url(sequence_id: str) -> str:
    """Return the canonical OEIS URL for a sequence ID.

    Parameters
    ----------
    sequence_id:
        An OEIS identifier such as ``"A000798"``.

    Returns
    -------
    str
        ``f"https://oeis.org/{sequence_id}"``.

    Examples
    --------
    >>> oeis_url("A000798")
    'https://oeis.org/A000798'
    """
    return f"https://oeis.org/{sequence_id}"


def lookup_total(n: int, n_ary: int) -> Optional[int]:
    """Look up :math:`|\\Gamma_N(X_n)|` from the paper's published tables.

    Checks :data:`TOTAL_NARY_COUNTS` first (covers ``n=0..4`` for
    ``N=0..4``), then :data:`EXTENDED_TERMS` for the additional ``n=5,6``
    rows the paper provides for ``N=1,2`` (and partially ``N=3,4``).

    Parameters
    ----------
    n:
        Carrier size.
    n_ary:
        Arity ``N``.

    Returns
    -------
    Optional[int]
        The published value, or ``None`` if this ``(N, n)`` combination is
        not covered by the paper's tables.

    Examples
    --------
    >>> lookup_total(3, 1)
    45
    >>> lookup_total(6, 1)
    257151
    >>> lookup_total(6, 0) is None
    True
    """
    row = TOTAL_NARY_COUNTS.get(n_ary)
    if row is not None and 0 <= n < len(row):
        return row[n]
    extended = EXTENDED_TERMS.get((n_ary, False))
    if extended is not None and n in extended:
        return extended[n]
    return None


def lookup_grounded(n: int, n_ary: int) -> Optional[int]:
    """Look up :math:`|\\Gamma^0_N(X_n)|` from the paper's published tables.

    See :func:`lookup_total` for the lookup strategy; this is the grounded
    analogue.

    Examples
    --------
    >>> lookup_grounded(4, 1)
    355
    >>> lookup_grounded(6, 1)
    209527
    """
    row = GROUNDED_NARY_COUNTS.get(n_ary)
    if row is not None and 0 <= n < len(row):
        return row[n]
    extended = EXTENDED_TERMS.get((n_ary, True))
    if extended is not None and n in extended:
        return extended[n]
    return None
