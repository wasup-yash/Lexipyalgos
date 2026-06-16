"""Tests for convexipy.oeis."""

from __future__ import annotations

from convexipy.oeis import (
    EXTENDED_TERMS,
    GROUNDED_NARY_COUNTS,
    OEIS_REFERENCES,
    TOTAL_NARY_COUNTS,
    lookup_grounded,
    lookup_total,
    oeis_url,
)


class TestTables:
    def test_total_table_shapes(self) -> None:
        for n_ary, row in TOTAL_NARY_COUNTS.items():
            assert len(row) == 5  # n=0..4

    def test_grounded_table_shapes(self) -> None:
        for n_ary, row in GROUNDED_NARY_COUNTS.items():
            assert len(row) == 5

    def test_n0_total_is_powers_of_two(self) -> None:
        assert TOTAL_NARY_COUNTS[0] == (1, 2, 4, 8, 16)

    def test_n0_grounded_is_all_ones(self) -> None:
        assert GROUNDED_NARY_COUNTS[0] == (1, 1, 1, 1, 1)

    def test_n1_grounded_matches_oeis_A000798_prefix(self) -> None:
        # A000798: 1, 1, 4, 29, 355, 6942, 209527, ...
        assert GROUNDED_NARY_COUNTS[1] == (1, 1, 4, 29, 355)
        assert EXTENDED_TERMS[(1, True)][5] == 6942
        assert EXTENDED_TERMS[(1, True)][6] == 209527


class TestLookup:
    def test_lookup_total_within_table(self) -> None:
        assert lookup_total(3, 1) == 45
        assert lookup_total(0, 0) == 1
        assert lookup_total(4, 2) == 2271

    def test_lookup_total_extended(self) -> None:
        assert lookup_total(5, 1) == 9053
        assert lookup_total(6, 1) == 257151
        assert lookup_total(6, 2) == 1556743050

    def test_lookup_total_missing_returns_none(self) -> None:
        assert lookup_total(6, 0) is None
        assert lookup_total(100, 1) is None

    def test_lookup_grounded_within_table(self) -> None:
        assert lookup_grounded(4, 1) == 355
        assert lookup_grounded(2, 2) == 4

    def test_lookup_grounded_extended(self) -> None:
        assert lookup_grounded(5, 1) == 6942
        assert lookup_grounded(6, 1) == 209527

    def test_lookup_grounded_missing_returns_none(self) -> None:
        assert lookup_grounded(6, 0) is None


class TestOeisReferences:
    def test_known_references(self) -> None:
        assert OEIS_REFERENCES[(0, False)] == "A000079"
        assert OEIS_REFERENCES[(0, True)] == "A000012"
        assert OEIS_REFERENCES[(1, False)] == "A326878"
        assert OEIS_REFERENCES[(1, True)] == "A000798"
        assert OEIS_REFERENCES[(2, True)] == "A364656"
        assert OEIS_REFERENCES[(3, True)] == "A395658"

    def test_unknown_references_are_none(self) -> None:
        assert OEIS_REFERENCES[(2, False)] is None
        assert OEIS_REFERENCES[(3, False)] is None
        assert OEIS_REFERENCES[(4, False)] is None
        assert OEIS_REFERENCES[(4, True)] is None

    def test_oeis_url(self) -> None:
        assert oeis_url("A000798") == "https://oeis.org/A000798"
