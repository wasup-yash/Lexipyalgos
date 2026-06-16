"""Tests for convexipy.core."""

from __future__ import annotations

import json

import pytest

from convexipy.core import ConvexitySpace, GroundingError, InvalidConvexityError


class TestConstruction:
    def test_basic_construction(self) -> None:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}, set()]
        space = ConvexitySpace(X, G)
        assert space.ground_set == frozenset(X)
        assert space.G == frozenset(frozenset(s) for s in G)

    def test_duplicate_families_deduplicated(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0, 1}, set(), set()])
        assert len(space.G) == 2

    def test_missing_ground_set_raises(self) -> None:
        with pytest.raises(InvalidConvexityError, match="ground set X must be"):
            ConvexitySpace({0, 1}, [{0}, set()])

    def test_member_not_subset_raises(self) -> None:
        with pytest.raises(InvalidConvexityError, match="not a subset"):
            ConvexitySpace({0, 1}, [{0, 1}, {0, 1, 2}])

    def test_not_closed_under_intersection_raises(self) -> None:
        # {0} and {1} intersect to {} which is missing
        with pytest.raises(InvalidConvexityError, match="not closed under intersection"):
            ConvexitySpace({0, 1}, [{0, 1}, {0}, {1}])

    def test_unchecked_skips_validation(self) -> None:
        # This is NOT a valid convexity but unchecked construction succeeds.
        space = ConvexitySpace.unchecked({0, 1}, [{0, 1}, {0}, {1}])
        assert not space.is_valid()

    def test_is_valid_true_for_good_space(self) -> None:
        space = ConvexitySpace.unchecked({0, 1}, [{0, 1}, {0}, set()])
        assert space.is_valid()


class TestFactories:
    def test_discrete(self) -> None:
        d = ConvexitySpace.discrete({0, 1})
        assert len(d) == 4  # full power set of 2-element set
        assert d.G == frozenset(
            frozenset(s) for s in [set(), {0}, {1}, {0, 1}]
        )

    def test_discrete_empty(self) -> None:
        d = ConvexitySpace.discrete(set())
        assert len(d) == 1
        assert frozenset() in d.G

    def test_trivial(self) -> None:
        t = ConvexitySpace.trivial({0, 1, 2})
        assert len(t) == 1
        assert t.G == frozenset({frozenset({0, 1, 2})})

    def test_trivial_grounded_only_when_empty(self) -> None:
        assert ConvexitySpace.trivial(set()).is_grounded()
        assert not ConvexitySpace.trivial({0}).is_grounded()

    def test_from_closure_operator_roundtrip(self) -> None:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}, set()]
        space = ConvexitySpace(X, G)

        rebuilt = ConvexitySpace.from_closure_operator(X, space.convex_hull)
        assert rebuilt == space

    def test_from_closure_operator_threshold_example(self) -> None:
        X = {0, 1, 2}

        def g(A: frozenset) -> frozenset:
            return frozenset(X) if len(A) >= 2 else frozenset(A)

        space = ConvexitySpace.from_closure_operator(X, g)
        expected = [set(), {0}, {1}, {2}, set(X)]
        assert space.G == frozenset(frozenset(s) for s in expected)


class TestConvexHull:
    @pytest.fixture
    def space(self) -> ConvexitySpace:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}, set()]
        return ConvexitySpace(X, G)

    def test_hull_of_empty(self, space: ConvexitySpace) -> None:
        assert space.convex_hull(set()) == frozenset()

    def test_hull_of_member(self, space: ConvexitySpace) -> None:
        # {0} is itself convex
        assert space.convex_hull({0}) == frozenset({0})

    def test_hull_of_singleton_1(self, space: ConvexitySpace) -> None:
        # smallest convex set containing {1} is {0,1}
        assert space.convex_hull({1}) == frozenset({0, 1})

    def test_hull_of_singleton_2(self, space: ConvexitySpace) -> None:
        # smallest convex set containing {2} is X
        assert space.convex_hull({2}) == frozenset({0, 1, 2})

    def test_hull_of_full_set(self, space: ConvexitySpace) -> None:
        assert space.convex_hull({0, 1, 2}) == frozenset({0, 1, 2})

    def test_hull_caching_consistency(self, space: ConvexitySpace) -> None:
        # Repeated calls return equal (and cached) results.
        a = space.convex_hull({1})
        b = space.convex_hull({1})
        assert a == b == frozenset({0, 1})

    def test_hull_rejects_non_subset(self, space: ConvexitySpace) -> None:
        with pytest.raises(ValueError, match="not a subset"):
            space.convex_hull({99})

    def test_hull_idempotent(self, space: ConvexitySpace) -> None:
        for A in [set(), {0}, {1}, {2}, {0, 1}, {0, 1, 2}]:
            hull = space.convex_hull(A)
            assert space.convex_hull(hull) == hull

    def test_hull_extensive(self, space: ConvexitySpace) -> None:
        for A in [set(), {0}, {1}, {2}, {0, 1}]:
            assert frozenset(A).issubset(space.convex_hull(A))

    def test_hull_monotone(self, space: ConvexitySpace) -> None:
        assert space.convex_hull({1}).issubset(space.convex_hull({1, 2}))

    def test_is_convex(self, space: ConvexitySpace) -> None:
        assert space.is_convex({0})
        assert space.is_convex(set())
        assert not space.is_convex({1})
        assert not space.is_convex({2})


class TestGrounding:
    def test_is_grounded_true(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        assert space.is_grounded()
        assert space.minimal_convex_set() == frozenset()

    def test_is_grounded_false(self) -> None:
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}])
        assert not space.is_grounded()
        assert space.minimal_convex_set() == frozenset({0})

    def test_get_grounded_reflection_basic(self) -> None:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}]
        space = ConvexitySpace(X, G)
        refl = space.get_grounded_reflection()
        assert refl.ground_set == frozenset({1, 2})
        assert refl.is_grounded()

    def test_get_grounded_reflection_paper_example(self) -> None:
        # From Table 3, n=2, entry 5: G={{0,1},{1}} | H={{0},{}}
        X = {0, 1}
        space = ConvexitySpace(X, [{0, 1}, {1}])
        refl = space.get_grounded_reflection()
        expected_H = {frozenset(), frozenset({0})}
        assert refl.G == frozenset(expected_H)
        assert refl.ground_set == frozenset({0})

    def test_get_grounded_reflection_already_grounded(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        refl = space.get_grounded_reflection()
        assert refl == space

    def test_from_grounded_roundtrip(self) -> None:
        X = {0, 1, 2}
        original = ConvexitySpace(X, [{0, 1, 2}, {0, 1}, {0}])
        refl = original.get_grounded_reflection()
        C = original.minimal_convex_set()
        rebuilt = ConvexitySpace.from_grounded(refl, C)
        assert rebuilt == original

    def test_from_grounded_requires_grounded_input(self) -> None:
        non_grounded = ConvexitySpace({0, 1}, [{0, 1}, {0}])
        with pytest.raises(GroundingError, match="must be grounded"):
            ConvexitySpace.from_grounded(non_grounded, {2})

    def test_from_grounded_requires_disjoint_C(self) -> None:
        grounded = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        with pytest.raises(GroundingError, match="disjoint"):
            ConvexitySpace.from_grounded(grounded, {0})

    def test_from_grounded_result_minimal_set_is_C(self) -> None:
        Y = {1, 2}
        H = ConvexitySpace(Y, [{1, 2}, set()])
        space = ConvexitySpace.from_grounded(H, {0})
        assert space.minimal_convex_set() == frozenset({0})
        assert space.ground_set == frozenset({0, 1, 2})


class TestNArity:
    def test_n_ary_negative_raises(self) -> None:
        space = ConvexitySpace({0}, [{0}, set()])
        with pytest.raises(ValueError, match="non-negative"):
            space.is_n_ary(-1)

    def test_discrete_is_0_ary(self) -> None:
        # P(X) is 0-ary: every set is convex, and g(B)=B for all B,
        # so the condition holds trivially for all A.
        d = ConvexitySpace.discrete({0, 1, 2})
        assert d.is_n_ary(0)

    def test_grounded_non_discrete_is_not_0_ary(self) -> None:
        # For a grounded space (g(empty) = {}), the 0-ary condition
        # "A in G <=> g(empty) subset A" reduces to "A in G <=> True",
        # i.e. G = P(X). Any grounded, non-discrete G is therefore not
        # 0-ary, even though it may be 1-ary.
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}, set()])
        assert space.is_n_ary(1)
        assert not space.is_n_ary(0)
        assert space.arity() == 1

    def test_non_grounded_space_can_be_0_ary(self) -> None:
        # For a non-grounded space, 0-arity reduces to "A in G <=>
        # g(empty) subset A", which can hold without G being discrete.
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {1, 2}])
        assert not space.is_grounded()
        assert space.is_n_ary(0)
        assert space.arity() == 0

    def test_every_space_is_n_ary_for_n_ge_size(self) -> None:
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}])
        assert space.is_n_ary(3)

    def test_arity_minimal(self) -> None:
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}, set()])
        a = space.arity()
        assert space.is_n_ary(a)
        if a > 0:
            assert not space.is_n_ary(a - 1)

    def test_arity_of_discrete_is_zero(self) -> None:
        d = ConvexitySpace.discrete({0, 1})
        assert d.arity() == 0


class TestLatticeStructure:
    def test_maximal_proper_convex_sets(self) -> None:
        X = {0, 1, 2}
        space = ConvexitySpace(
            X,
            [
                {0, 1, 2},
                {0, 1},
                {0, 2},
                {1, 2},
                {0},
                {1},
                {2},
                set(),
            ],
        )
        maximal = space.maximal_proper_convex_sets()
        assert frozenset({0, 1}) in maximal
        assert frozenset({0, 2}) in maximal
        assert frozenset({1, 2}) in maximal
        assert frozenset(X) not in maximal
        assert frozenset({0}) not in maximal  # subset of {0,1}

    def test_minimal_nonempty_convex_sets(self) -> None:
        X = {0, 1, 2}
        space = ConvexitySpace(X, [{0, 1, 2}, {0, 1}, {0}, {1}, set()])
        minimal = space.minimal_nonempty_convex_sets()
        assert minimal == frozenset({frozenset({0}), frozenset({1})})

    def test_convex_sets_sorted_deterministic(self) -> None:
        space = ConvexitySpace({0, 1, 2}, [{0, 1, 2}, {0, 1}, {0}, set()])
        sorted1 = space.convex_sets_sorted()
        sorted2 = space.convex_sets_sorted()
        assert sorted1 == sorted2
        # ascending by size
        sizes = [len(s) for s in sorted1]
        assert sizes == sorted(sizes)


class TestRestriction:
    def test_restrict_to_subset(self) -> None:
        X = {0, 1, 2}
        space = ConvexitySpace(X, [{0, 1, 2}, {0, 1}, {0}, set()])
        sub = space.restrict_to({0, 1})
        assert sub.ground_set == frozenset({0, 1})
        assert sub.is_valid()

    def test_restrict_to_intersects_ground_set(self) -> None:
        X = {0, 1, 2}
        space = ConvexitySpace(X, [{0, 1, 2}, {0}, set()])
        sub = space.restrict_to({0, 1, 99})  # 99 not in X
        assert sub.ground_set == frozenset({0, 1})


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}, set()]
        space = ConvexitySpace(X, G)
        d = space.to_dict()
        rebuilt = ConvexitySpace.from_dict(d)
        assert rebuilt == space

    def test_to_dict_structure(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, set()])
        d = space.to_dict()
        assert d["ground_set"] == [0, 1]
        assert [] in d["families"]
        assert [0, 1] in d["families"]

    def test_to_json_from_json_roundtrip(self) -> None:
        X = {0, 1, 2}
        G = [{0, 1, 2}, {0, 1}, {0}, set()]
        space = ConvexitySpace(X, G)
        s = space.to_json()
        rebuilt = ConvexitySpace.from_json(s)
        assert rebuilt == space
        # also valid plain JSON
        json.loads(s)

    def test_to_dict_non_orderable_elements(self) -> None:
        # tuples are orderable, so this should work fine via repr fallback
        space = ConvexitySpace.unchecked(
            {(1, "a"), (2, "b")}, [{(1, "a"), (2, "b")}, set()]
        )
        d = space.to_dict()
        assert len(d["ground_set"]) == 2


class TestDunders:
    def test_repr_contains_class_name(self) -> None:
        space = ConvexitySpace({0}, [{0}, set()])
        assert "ConvexitySpace" in repr(space)

    def test_str_summary(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        s = str(space)
        assert "|X|=2" in s
        assert "grounded" in s

    def test_str_non_grounded(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}])
        s = str(space)
        assert "non-grounded" in s

    def test_equality_and_hash(self) -> None:
        s1 = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        s2 = ConvexitySpace({0, 1}, [set(), {0, 1}, {0}])  # different order
        assert s1 == s2
        assert hash(s1) == hash(s2)

    def test_inequality_with_other_types(self) -> None:
        space = ConvexitySpace({0}, [{0}, set()])
        assert space != "not a space"
        assert space != 42

    def test_len(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        assert len(space) == 3

    def test_iter_returns_all_members(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        members = list(space)
        assert len(members) == 3
        assert frozenset({0}) in members

    def test_contains(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        assert {0} in space
        assert {1} not in space
        assert set() in space

    def test_contains_non_iterable_returns_false(self) -> None:
        space = ConvexitySpace({0, 1}, [{0, 1}, {0}, set()])
        assert 42 not in space
