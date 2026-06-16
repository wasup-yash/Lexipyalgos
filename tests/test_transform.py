"""Tests for convexipy.transform."""

from __future__ import annotations

import pytest

from convexipy.transform import (
    apply_binomial_identity,
    binomial_transform,
    binomial_transform_term,
    inverse_binomial_transform,
    inverse_binomial_transform_term,
    is_binomial_transform_pair,
    recover_grounded_sequence,
)


class TestBinomialTransform:
    def test_known_pair_N1(self) -> None:
        grounded = [1, 1, 4, 29, 355]
        total = [1, 2, 7, 45, 500]
        assert binomial_transform(grounded) == total

    def test_known_pair_N2(self) -> None:
        grounded = [1, 1, 4, 45, 2062]
        total = [1, 2, 7, 61, 2271]
        assert binomial_transform(grounded) == total

    def test_empty_sequence(self) -> None:
        assert binomial_transform([]) == []

    def test_single_element(self) -> None:
        assert binomial_transform([5]) == [5]

    def test_all_zero(self) -> None:
        assert binomial_transform([0, 0, 0]) == [0, 0, 0]


class TestInverseBinomialTransform:
    def test_known_pair(self) -> None:
        total = [1, 2, 7, 45, 500]
        grounded = [1, 1, 4, 29, 355]
        assert inverse_binomial_transform(total) == grounded

    def test_inverse_is_exact_inverse(self) -> None:
        original = [1, 1, 4, 29, 355, 6942]
        transformed = binomial_transform(original)
        recovered = inverse_binomial_transform(transformed)
        assert recovered == original

    def test_double_application_is_identity(self) -> None:
        seq = [3, -2, 7, 0, 11, -5]
        assert inverse_binomial_transform(binomial_transform(seq)) == seq
        assert binomial_transform(inverse_binomial_transform(seq)) == seq

    def test_empty(self) -> None:
        assert inverse_binomial_transform([]) == []


class TestTermFunctions:
    def test_binomial_transform_term_matches_full(self) -> None:
        seq = [1, 1, 4, 29, 355]
        full = binomial_transform(seq)
        for n in range(len(seq)):
            assert binomial_transform_term(seq, n) == full[n]

    def test_binomial_transform_term_n4(self) -> None:
        # Table 2 row N=1: [1,1,4,29,355] -> Table 1 row N=1 at n=4 is 500.
        assert binomial_transform_term([1, 1, 4, 29, 355], 4) == 500

    def test_inverse_term_matches_full(self) -> None:
        seq = [1, 2, 7, 45, 500]
        full = inverse_binomial_transform(seq)
        for n in range(len(seq)):
            assert inverse_binomial_transform_term(seq, n) == full[n]

    def test_inverse_term_n4(self) -> None:
        assert inverse_binomial_transform_term([1, 2, 7, 45, 500], 4) == 355

    def test_term_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            binomial_transform_term([1, 2, 3], -1)
        with pytest.raises(ValueError, match="non-negative"):
            inverse_binomial_transform_term([1, 2, 3], -1)

    def test_term_insufficient_length_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            binomial_transform_term([1, 2], 5)
        with pytest.raises(ValueError, match="at least"):
            inverse_binomial_transform_term([1, 2], 5)


class TestIdentityHelpers:
    def test_is_binomial_transform_pair_true(self) -> None:
        grounded = [1, 1, 4, 29, 355, 6942]
        total = [1, 2, 7, 45, 500, 9053]
        assert is_binomial_transform_pair(grounded, total)

    def test_is_binomial_transform_pair_false(self) -> None:
        grounded = [1, 1, 4, 29, 355]
        wrong_total = [1, 2, 7, 45, 999]
        assert not is_binomial_transform_pair(grounded, wrong_total)

    def test_is_binomial_transform_pair_length_mismatch(self) -> None:
        assert not is_binomial_transform_pair([1, 2], [1, 2, 3])

    def test_apply_binomial_identity(self) -> None:
        grounded = [1, 1, 4, 29, 355]
        assert apply_binomial_identity(grounded, 4) == 500
        assert apply_binomial_identity(grounded, 0) == 1
        assert apply_binomial_identity(grounded, 1) == 2

    def test_recover_grounded_sequence(self) -> None:
        total = [1, 2, 7, 45, 500, 9053]
        expected_grounded = [1, 1, 4, 29, 355, 6942]
        assert recover_grounded_sequence(total) == expected_grounded
