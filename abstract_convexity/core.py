
"""
Core domain logic for abstract convexity spaces.
Defines axioms, validation checks, and convex hull calculation operators.
"""

from typing import Set, FrozenSet, List, Union, Iterable
import itertools

from numpy import iterable

class InvalidConvexityError(ValueError):
    """Exception raised when a family of sets violates the axioms of a convexity space."""
    pass

class ConvexitySpace:
    """
    Represents a convexity space (X, G), where X is a finite ground set 
    and G is a family of subsets of X closed under arbitrary intersections and containing X.
    """
    def __init__(self, ground_set: Iterable, families: Iterable[FrozenSet]):
        self.ground_set = frozenset(ground_set)
        self.G = frozenset(frozenset(s) for s in families)
        self._validate()

    def _validate(self) -> None:
        """Validates the standard axiomatic criteria for abstract convexity spaces."""
        if self.ground_set not in self.G:
            raise InvalidConvexityError("The ground set X must be an element of the convexity family G.")
        for A in self.G:
            if not A.issubset(self.ground_set):
                raise InvalidConvexityError(f"Element {A} in G is not a subset of the ground set X.")
            for B in self.G:
                if (A & B) not in self.G:
                    raise InvalidConvexityError(
                        f"G is not closed under intersection: {set(A)} and {set(B)} "
                        f"intersect to {set(A & B)}, which is missing from G."
                    )

    def convex_hull(self, A: Union[Set, FrozenSet, List]) -> FrozenSet:
        """
        Computes the convex hull g(A) of a subset A of the ground set X.
        g(A) is defined as the intersection of all convex sets containing A.
        """
        A_fs = frozenset(A)
        if not A_fs.issubset(self.ground_set):
            raise ValueError("The subset A must be a subset of the ground set X.")
        
        containing_sets = [B for B in self.G if A_fs.issubset(B)]
        if not containing_sets:
            return self.ground_set
        
        result = containing_sets[0]
        for B in containing_sets[1:]:
            result = result & B
        return result

    def is_grounded(self) -> bool:
        """Checks if the convexity satisfies the grounding axiom (i.e., empty set is in G)."""
        return frozenset() in self.G

    def minimal_convex_set(self) -> FrozenSet:
        """Returns the minimal convex set, which is the convex hull of the empty set g(∅)."""
        return self.convex_hull(frozenset())

    def is_n_ary(self, n: int) -> bool:
        """
        Checks if the convexity is N-ary.
        A convexity is N-ary if for all A subset of X:
        A in G <=> (forall B subset of A, |B| <= N implies g(B) subset of A).
        """
        if n < 0:
            raise ValueError("Arity N must be a non-negative integer.")
            
        elements = list(self.ground_set)
        
        # Precompute g(B) for all B subset of X with |B| <= n to optimize lookups
        hull_cache = {}
        for k in range(min(n, len(elements)) + 1):
            for B_tuple in itertools.combinations(elements, k):
                B = frozenset(B_tuple)
                hull_cache[B] = self.convex_hull(B)
                
        # Evaluate the N-ary closure requirement for all subsets A of X
        for r in range(len(elements) + 1):
            for A_tuple in itertools.combinations(elements, r):
                A = frozenset(A_tuple)
                
                # Check if A contains the hulls of all its subsets of size <= N
                condition_satisfied = True
                for B, g_B in hull_cache.items():
                    if B.issubset(A):
                        if not g_B.issubset(A):
                            condition_satisfied = False
                            break
                            
                # G fails N-arity if a set satisfies the small-hull condition but is not convex
                if (A not in self.G) and condition_satisfied:
                    return False
        return True

    def get_grounded_reflection(self) -> 'ConvexitySpace':
        """
        Constructs the corresponding unique grounded convexity H on Y = X \\ C,
        where C is the minimal convex set g(∅), according to Lemma 1.
        """
        C = self.minimal_convex_set()
        Y = self.ground_set - C
        H_families = frozenset(A - C for A in self.G)
        return ConvexitySpace(Y, H_families)

    @classmethod
    def from_grounded(cls, H: 'ConvexitySpace', C: Union[Set, FrozenSet, List]) -> 'ConvexitySpace':
        """
        Constructs a convexity G on X = Y cup C from a grounded convexity H on Y,
        where C is the minimal convex set, according to Lemma 5.
        """
        C_fs = frozenset(C)
        if not H.is_grounded():
            raise ValueError("The provided convexity H must be grounded.")
        if not (H.ground_set & C_fs) == frozenset():
            raise ValueError("The minimal convex set C must be completely disjoint from the ground set of H.")
        
        X = H.ground_set | C_fs
        G_families = frozenset(C_fs | D for D in H.G)
        return cls(X, G_families)

    def __repr__(self) -> str:
        return f"ConvexitySpace(ground_set={set(self.ground_set)}, families={[set(s) for s in self.G]})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConvexitySpace):
            return False
        return self.ground_set == other.ground_set and self.G == other.G

    def __hash__(self) -> int:
        return hash((self.ground_set, self.G))