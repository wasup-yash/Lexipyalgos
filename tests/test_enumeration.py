"""Tests for convexipy.enumeration."""

from __future__ import annotations

import pytest

from convexipy.core import ConvexitySpace
from convexipy.enumeration import (
    count_convexities,
    count_grounded_convexities,
    enumerate_and_classify,
    generate_convexities,
)


class TestGenerateConvexitiesBasics:
    def test_n_zero_unfiltered(self) -> None:
        spaces = list(generate_convexities(0))
        assert len(spaces) == 1
        assert spaces[0].ground_set == frozenset()
        assert spaces[0].G == frozenset({frozenset()})

    def test_n_zero_grounded(self) -> None:
        spaces = list(generate_convexities(0, grounded_only=True))
        assert len(spaces) == 1

    def test_n_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            list(generate_convexities(-1))

    def test_all_yielded_are_valid(self) -> None:
        for space in generate_convexities(3):
            assert space.is_valid()

    def test_all_yielded_are_unique(self) -> None:
        spaces = list(generate_convexities(3))
        assert len(spaces) == len(set(spaces))

    def test_grounded_only_yields_only_grounded(self) -> None:
        for space in generate_convexities(3, grounded_only=True):
            assert space.is_grounded()

    def test_n_ary_filter_yields_only_n_ary(self) -> None:
        for space in generate_convexities(3, n_ary=1):
            assert space.is_n_ary(1)

    def test_on_progress_called(self) -> None:
        calls = []

        def cb(yielded: int, total: int) -> None:
            calls.append((yielded, total))

        list(generate_convexities(2, on_progress=cb))
        assert len(calls) == 7  # |Gamma(X_2)| = 7
        assert calls[-1][0] == 7


class TestPaperTable1TotalCounts:
    """Verify |Gamma_N(X_n)| against Table 1 of Dulliev & Naumikhin (2026)
    for n = 0..4 and N = 0..4.
    """

    TABLE_1 = {
        0: [1, 2, 4, 8, 16],
        1: [1, 2, 7, 45, 500],
        2: [1, 2, 7, 61, 2271],
        3: [1, 2, 7, 61, 2480],
        4: [1, 2, 7, 61, 2480],
    }

    @pytest.mark.parametrize("n_ary", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
    def test_table1_entry(self, n_ary: int, n: int) -> None:
        expected = self.TABLE_1[n_ary][n]
        actual = count_convexities(n, n_ary=n_ary)
        assert actual == expected, (
            f"|Gamma_{n_ary}(X_{n})| = {actual}, expected {expected}"
        )


class TestPaperTable2GroundedCounts:
    """Verify |Gamma^0_N(X_n)| against Table 2 of Dulliev & Naumikhin
    (2026) for n = 0..4 and N = 0..4.
    """

    TABLE_2 = {
        0: [1, 1, 1, 1, 1],
        1: [1, 1, 4, 29, 355],
        2: [1, 1, 4, 45, 2062],
        3: [1, 1, 4, 45, 2271],
        4: [1, 1, 4, 45, 2271],
    }

    @pytest.mark.parametrize("n_ary", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
    def test_table2_entry(self, n_ary: int, n: int) -> None:
        expected = self.TABLE_2[n_ary][n]
        actual = count_grounded_convexities(n, n_ary=n_ary)
        assert actual == expected, (
            f"|Gamma0_{n_ary}(X_{n})| = {actual}, expected {expected}"
        )


@pytest.mark.slow
class TestPaperN5ExtendedTerms:
    """Slower n=5 checks against the paper's extended (N=1) terms.

    These take on the order of tens of seconds each; run with
    `pytest -m slow` to include them.
    """

    def test_total_N1_n5(self) -> None:
        assert count_convexities(5, n_ary=1) == 9053

    def test_grounded_N1_n5(self) -> None:
        assert count_grounded_convexities(5, n_ary=1) == 6942


class TestPaperTable3BinaryConvexitiesN2:
    """Spot-check specific entries from Table 3 (n=2 binary convexities
    and their groundings) by reconstructing them directly.
    """

    def test_entry_1_trivial(self) -> None:
        # No 1 {{0,1}} | {{}}
        space = ConvexitySpace({0, 1}, [{0, 1}])
        refl = space.get_grounded_reflection()
        assert refl.G == frozenset({frozenset()})

    def test_entry_4_discrete_grounded(self) -> None:
        # No 4 {{0,1},{}} | {{0,1},{}}
        space = ConvexitySpace({0, 1}, [{0, 1}, set()])
        refl = space.get_grounded_reflection()
        assert refl == space  # already grounded -> reflection is itself

    def test_entry_5(self) -> None:
        # No 5 {{0,1},{1}} | {{0},{}}
        space = ConvexitySpace({0, 1}, [{0, 1}, {1}])
        refl = space.get_grounded_reflection()
        expected = frozenset({frozenset(), frozenset({0})})
        assert refl.G == expected

    def test_entry_7_full_lattice(self) -> None:
        # No 7 {{0,1},{},{0},{1}} | itself (discrete)
        space = ConvexitySpace({0, 1}, [{0, 1}, set(), {0}, {1}])
        refl = space.get_grounded_reflection()
        assert refl == space

    def test_table3_n2_count_is_7(self) -> None:
        assert count_convexities(2) == 7


class TestEnumerateAndClassify:
    def test_basic_partition(self) -> None:
        result = enumerate_and_classify(2)
        assert result.total == 7
        assert result.grounded_total == 4
        assert result.total == len(result.spaces)
        assert result.grounded_total == len(result.grounded_spaces)

    def test_grounded_spaces_subset_of_spaces(self) -> None:
        result = enumerate_and_classify(3, n_ary=1)
        space_set = set(result.spaces)
        for g in result.grounded_spaces:
            assert g in space_set
            assert g.is_grounded()

    def test_n_ary_filter_applied(self) -> None:
        # Table 1, N=1, n=3 = 45; Table 2, N=1, n=3 = 29.
        result = enumerate_and_classify(3, n_ary=1)
        assert result.total == 45
        assert result.grounded_total == 29

    def test_stats(self) -> None:
        result = enumerate_and_classify(2)
        stats = result.stats()
        assert stats.n == 2
        assert stats.total == 7
        assert stats.grounded_total == 4

    def test_verify_binomial_identity_true(self) -> None:
        grounded_seq = [count_grounded_convexities(k, n_ary=1) for k in range(4)]
        result = enumerate_and_classify(3, n_ary=1)
        assert result.verify_binomial_identity(grounded_seq)

    def test_verify_binomial_identity_wrong_length_raises(self) -> None:
        result = enumerate_and_classify(3, n_ary=1)
        with pytest.raises(ValueError, match="length n\\+1"):
            result.verify_binomial_identity([1, 2, 3])

    def test_verify_binomial_identity_false_for_wrong_data(self) -> None:
        result = enumerate_and_classify(3, n_ary=1)
        assert not result.verify_binomial_identity([1, 1, 1, 1])
