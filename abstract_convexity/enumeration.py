"""
High-performance algorithms for searching and generating valid abstract convexities
on finite carriers. Uses tree backtracking with incremental closure pruning.
"""

from typing import Iterator, Set, FrozenSet, List, Dict
import itertools
from .core import ConvexitySpace

def _compute_closure_incremental(
    current_closure: Set[FrozenSet], 
    new_element: FrozenSet, 
    states: List[int], 
    subset_to_idx: Dict[FrozenSet, int]
) -> tuple[bool, List[int]]:
    """
    Performs an incremental intersection closure check. 
    Returns (is_valid, newly_added_indices). Prunes early if an excluded state is forced.
    """
    if new_element in current_closure:
        return True, []
        
    closure = set(current_closure)
    queue = [new_element]
    closure.add(new_element)
    
    newly_added_indices = []
    new_element_idx = subset_to_idx[new_element]
    if states[new_element_idx] == -1:
        return False, []
    if states[new_element_idx] == 0:
        newly_added_indices.append(new_element_idx)
        
    idx_q = 0
    while idx_q < len(queue):
        item = queue[idx_q]
        idx_q += 1
        
        current_closure_list = list(closure)
        for other in current_closure_list:
            inter = item & other
            if inter not in closure:
                inter_idx = subset_to_idx[inter]
                if states[inter_idx] == -1:
                    return False, []
                closure.add(inter)
                queue.append(inter)
                if states[inter_idx] == 0:
                    newly_added_indices.append(inter_idx)
                    
    return True, newly_added_indices

def generate_convexities(n: int, grounded_only: bool = False) -> Iterator[ConvexitySpace]:
    """
    Generates all valid ConvexitySpace structures on a carrier set of size n.
    Utilizes binary state choices (INCLUDED/EXCLUDED) and prunes invalid branches.
    """
    ground_set = frozenset(range(n))
    elements = list(ground_set)
    
    # Generate all subsets of the carrier set
    all_subsets = []
    for r in range(n + 1):
        for combo in itertools.combinations(elements, r):
            all_subsets.append(frozenset(combo))
            
    # Sort descending by size to expedite intersection convergence
    all_subsets.sort(key=lambda s: (len(s), sorted(list(s))), reverse=True)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    m = len(all_subsets)
    
    # State tracking array: 0 = UNDECIDED, 1 = INCLUDED, -1 = EXCLUDED
    states = [0] * m
    states[0] = 1  # Ground set X must always be in G
    
    initial_closure = {ground_set}
    if grounded_only:
        empty_set = frozenset()
        states[subset_to_idx[empty_set]] = 1
        initial_closure.add(empty_set)
        
    def backtrack(idx: int, current_closure: Set[FrozenSet]) -> Iterator[ConvexitySpace]:
        if idx == m:
            families = [all_subsets[i] for i in range(m) if states[i] == 1]
            yield ConvexitySpace(ground_set, families)
            return
            
        if states[idx] != 0:
            yield from backtrack(idx + 1, current_closure)
            return
            
        # Decision Branch 1: Exclude the subset at all_subsets[idx]
        states[idx] = -1
        yield from backtrack(idx + 1, current_closure)
        states[idx] = 0
        
        # Decision Branch 2: Include the subset and evaluate incremental intersection closure
        success, newly_added = _compute_closure_incremental(
            current_closure, all_subsets[idx], states, subset_to_idx
        )
        if success:
            for i in newly_added:
                states[i] = 1
            next_closure = current_closure | {all_subsets[i] for i in newly_added}
            
            yield from backtrack(idx + 1, next_closure)
            
            for i in newly_added:
                states[i] = 0

    yield from backtrack(0, initial_closure)