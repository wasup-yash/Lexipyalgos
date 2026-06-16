"""Tests for convexipy.applications."""

from __future__ import annotations

import pytest

from convexipy.applications import (
    feature_space_convexity,
    graph_convexity_from_paths,
    is_separable,
    preorder_from_relation,
    safe_enumerate,
    upset_closure,
    validate_configuration_family,
)
from convexipy.core import ConvexitySpace
from convexipy.exceptions import EnumerationLimitError, InvalidConvexityError


class TestPreorderFromRelation:
    def test_consistent_hierarchy(self) -> None:
        result = preorder_from_relation(
            {"admin", "editor", "viewer"},
            [
                {"admin", "editor", "viewer"},
                {"editor", "viewer"},
                {"viewer"},
                set(),
            ],
        )
        assert result.is_consistent
        assert result.space.is_valid()
        assert result.space.is_grounded()

    def test_leq_is_reflexive(self) -> None:
        result = preorder_from_relation(
            {"a", "b"}, [{"a", "b"}, {"b"}, set()]
        )
        assert ("a", "a") in result.leq
        assert ("b", "b") in result.leq

    def test_leq_transitive_chain(self) -> None:
        result = preorder_from_relation(
            {"admin", "editor", "viewer"},
            [
                {"admin", "editor", "viewer"},
                {"editor", "viewer"},
                {"viewer"},
                set(),
            ],
        )
        # admin <= viewer (transitively, via editor)
        assert ("admin", "viewer") in result.leq
        assert ("admin", "editor") in result.leq
        # but not the reverse
        assert ("viewer", "admin") not in result.leq
        assert ("editor", "admin") not in result.leq

    def test_total_order_chain(self) -> None:
        # Total order a <= b <= c: up-sets are {}, {c}, {b,c}, {a,b,c}
        result = preorder_from_relation(
            {"a", "b", "c"},
            [{"a", "b", "c"}, {"b", "c"}, {"c"}, set()],
        )
        assert result.is_consistent
        expected_leq = {
            ("a", "a"), ("a", "b"), ("a", "c"),
            ("b", "b"), ("b", "c"),
            ("c", "c"),
        }
        assert result.leq == expected_leq

    def test_inconsistent_input_is_repaired(self) -> None:
        # {a,b} and {b,c} intersect to {b}, which is not in the input.
        result = preorder_from_relation({"a", "b", "c"}, [{"a", "b"}, {"b", "c"}])
        assert not result.is_consistent
        # space is always a valid convexity, even when input wasn't
        assert result.space.is_valid()
        # the repaired family includes the missing intersection
        assert frozenset({"b"}) in result.space.G

    def test_X_and_empty_always_included(self) -> None:
        result = preorder_from_relation({"x", "y"}, [{"x"}])
        assert frozenset({"x", "y"}) in result.space.G
        assert frozenset() in result.space.G

    def test_upsets_outside_universe_are_clipped(self) -> None:
        result = preorder_from_relation({"x"}, [{"x", "outside"}])
        assert result.space.ground_set == frozenset({"x"})


class TestUpsetClosure:
    def test_closure_of_top_is_everything(self) -> None:
        result = preorder_from_relation(
            {"a", "b", "c"}, [{"a", "b", "c"}, {"b", "c"}, {"c"}, set()]
        )
        assert upset_closure(result.space, {"a"}) == frozenset({"a", "b", "c"})

    def test_closure_of_bottom_is_itself(self) -> None:
        result = preorder_from_relation(
            {"a", "b", "c"}, [{"a", "b", "c"}, {"b", "c"}, {"c"}, set()]
        )
        assert upset_closure(result.space, {"c"}) == frozenset({"c"})

    def test_matches_convex_hull(self) -> None:
        result = preorder_from_relation(
            {"a", "b", "c"}, [{"a", "b", "c"}, {"b", "c"}, {"c"}, set()]
        )
        for elem in ("a", "b", "c"):
            assert upset_closure(result.space, {elem}) == result.space.convex_hull(
                {elem}
            )


class TestFeatureSpaceConvexity:
    def test_basic_segments(self) -> None:
        users = {"alice", "bob", "carol", "dave"}
        segments = [
            {"alice", "bob", "carol"},
            {"bob", "carol", "dave"},
        ]
        space = feature_space_convexity(users, segments)
        assert space.is_valid()
        assert space.convex_hull({"bob"}) == frozenset({"bob", "carol"})

    def test_auto_close_adds_universe_and_empty(self) -> None:
        users = {"a", "b", "c"}
        space = feature_space_convexity(users, [{"a", "b"}])
        assert frozenset(users) in space.G
        assert frozenset() in space.G

    def test_auto_close_false_raises_on_non_closed_family(self) -> None:
        users = {"alice", "bob", "carol", "dave"}
        segments = [{"alice", "bob", "carol"}, {"bob", "carol", "dave"}]
        with pytest.raises(InvalidConvexityError):
            feature_space_convexity(users, segments, auto_close=False)

    def test_auto_close_false_succeeds_when_already_closed(self) -> None:
        users = {"a", "b", "c"}
        # {a,b,c}, {a,b}, {} -- intersection of {a,b} with itself is {a,b};
        # need full closure including empty set and universe present.
        space = feature_space_convexity(
            users, [{"a", "b", "c"}, {"a", "b"}, set()], auto_close=False
        )
        assert space.is_valid()

    def test_elements_outside_universe_clipped(self) -> None:
        users = {"a", "b"}
        space = feature_space_convexity(users, [{"a", "outside"}])
        assert space.ground_set == frozenset(users)
        for member in space.G:
            assert member.issubset(frozenset(users))


class TestIsSeparable:
    @pytest.fixture
    def space(self) -> ConvexitySpace:
        X = {0, 1, 2, 3}
        return ConvexitySpace(X, [X, {0, 1}, {2, 3}, set()])

    def test_disjoint_hulls_are_separable(self, space: ConvexitySpace) -> None:
        assert is_separable(space, {0}, {3})
        assert is_separable(space, {0, 1}, {2, 3})

    def test_overlapping_hulls_not_separable(self, space: ConvexitySpace) -> None:
        assert not is_separable(space, {0}, {1})

    def test_separable_is_symmetric(self, space: ConvexitySpace) -> None:
        assert is_separable(space, {0}, {3}) == is_separable(space, {3}, {0})

    def test_empty_group_is_separable_from_anything(
        self, space: ConvexitySpace
    ) -> None:
        assert is_separable(space, set(), {0, 1, 2, 3})


class TestGraphConvexityFromPaths:
    def test_path_graph(self) -> None:
        g = {0: [1], 1: [0, 2], 2: [1]}
        space = graph_convexity_from_paths(g)
        assert space.is_valid()
        assert space.ground_set == frozenset({0, 1, 2})
        # The only interval containing both endpoints of the path is V.
        assert space.convex_hull({0, 2}) == frozenset({0, 1, 2})

    def test_singletons_are_convex(self) -> None:
        g = {0: [1], 1: [0, 2], 2: [1]}
        space = graph_convexity_from_paths(g)
        for v in space.ground_set:
            assert space.is_convex({v})

    def test_triangle_graph_interval_includes_all_paths(self) -> None:
        # In a triangle 0-1-2-0, simple paths from 0 to 1 are:
        #   [0,1] (direct edge) and [0,2,1] (via 2).
        # The interval is the union: {0,1,2} = V, so {0,1} alone is NOT convex.
        g = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        space = graph_convexity_from_paths(g)
        assert space.is_valid()
        assert not space.is_convex({0, 1})
        assert space.convex_hull({0, 1}) == frozenset({0, 1, 2})

    def test_empty_graph(self) -> None:
        space = graph_convexity_from_paths({})
        assert space.ground_set == frozenset()
        assert space.is_valid()

    def test_max_pair_samples_limits_pairs_considered(self) -> None:
        g = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        # Should not error even with a restrictive sample limit.
        space = graph_convexity_from_paths(g, max_pair_samples=1)
        assert space.is_valid()
        assert space.ground_set == frozenset({0, 1, 2, 3})


class TestValidateConfigurationFamily:
    def test_valid_family(self) -> None:
        universe = {"A1", "A2", "B1", "B2"}
        configs = [universe, {"A1", "A2"}, {"A1"}, set()]
        report = validate_configuration_family(universe, configs)
        assert report.is_valid
        assert not report.missing_intersections
        assert not report.out_of_universe_members
        assert not report.missing_ground_set

    def test_missing_intersection_detected(self) -> None:
        universe = {"A1", "A2", "B1", "B2"}
        configs = [{"A1", "A2"}, {"A1", "B1"}]
        report = validate_configuration_family(universe, configs)
        assert not report.is_valid
        found = [set(c) for _, _, c in report.missing_intersections]
        assert {"A1"} in found

    def test_missing_ground_set_detected(self) -> None:
        universe = {"A1", "A2"}
        configs = [{"A1"}, set()]
        report = validate_configuration_family(universe, configs)
        assert report.missing_ground_set
        assert not report.is_valid

    def test_out_of_universe_elements_detected(self) -> None:
        universe = {"A1", "A2"}
        configs = [{"A1", "A2"}, {"A1", "ZZZ"}, set()]
        report = validate_configuration_family(universe, configs)
        assert (1, "ZZZ") in report.out_of_universe_members
        assert not report.is_valid

    def test_repaired_space_always_valid(self) -> None:
        universe = {"A1", "A2", "B1", "B2"}
        configs = [{"A1", "A2"}, {"A1", "B1"}]
        report = validate_configuration_family(universe, configs)
        assert report.space.is_valid()

    def test_summary_for_valid_family(self) -> None:
        universe = {"A1", "A2"}
        configs = [universe, {"A1"}, set()]
        report = validate_configuration_family(universe, configs)
        assert "no issues" in report.summary().lower()

    def test_summary_for_invalid_family_mentions_each_issue(self) -> None:
        universe = {"A1", "A2", "B1"}
        configs = [{"A1", "A2"}, {"A1", "B1"}]  # missing ground set + intersection
        report = validate_configuration_family(universe, configs)
        summary = report.summary()
        assert "NOT closed under intersection" in summary
        assert "universe set X was not present" in summary
        assert "A1" in summary


class TestSafeEnumerate:
    def test_within_limit(self) -> None:
        spaces = safe_enumerate(3, n_ary=1)
        assert len(spaces) == 45

    def test_exceeds_default_limit_raises(self) -> None:
        with pytest.raises(EnumerationLimitError):
            safe_enumerate(100)

    def test_error_contains_n_and_limit(self) -> None:
        with pytest.raises(EnumerationLimitError) as exc_info:
            safe_enumerate(10)
        err = exc_info.value
        assert err.n == 10
        assert err.limit == 5

    def test_explicit_max_n_override(self) -> None:
        # n=5 is within an explicitly raised max_n, and n_ary=1 keeps it fast.
        spaces = safe_enumerate(5, n_ary=1, max_n=5)
        assert len(spaces) == 9053

    def test_grounded_only_passthrough(self) -> None:
        spaces = safe_enumerate(3, n_ary=1, grounded_only=True)
        assert len(spaces) == 29
        for s in spaces:
            assert s.is_grounded()
