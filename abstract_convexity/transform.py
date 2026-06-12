"""
Implements binomial transforms and their inverse equivalents for integer sequences.
 Used to establish identities between general and grounded structures.
"""

from typing import List
import math

def binomial_transform(sequence: List[int]) -> List[int]:
    """
    Computes the binomial transform of a sequence.
    b_n = sum_{k=0}^n C_n^k * a_k
    """
    transform = []
    for n in range(len(sequence)):
        val = 0
        for k in range(n + 1):
            val += math.comb(n, k) * sequence[k]
        transform.append(val)
    return transform

def inverse_binomial_transform(sequence: List[int]) -> List[int]:
    """
    Computes the inverse binomial transform of a sequence.
    a_n = sum_{k=0}^n (-1)^{n-k} * C_n^k * b_k
    """
    inverse = []
    for n in range(len(sequence)):
        val = 0
        for k in range(n + 1):
            val += ((-1) ** (n - k)) * math.comb(n, k) * sequence[k]
        inverse.append(val)
    return inverse