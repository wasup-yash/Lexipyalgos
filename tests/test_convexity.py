import unittest
from abstract_convexity.core import ConvexitySpace, InvalidConvexityError
from abstract_convexity.transform import binomial_transform, inverse_binomial_transform
from abstract_convexity.enumeration import generate_convexities

class TestAbstractConvexity(unittest.TestCase):
    
    def test_axiomatic_validation(self) -> None:
        """Tests that incorrect structures properly trigger custom validation exceptions."""
        # Missing the ground set completely
        with self.assertRaises(InvalidConvexityError):
            ConvexitySpace({0, 1}, [frozenset({0})])
            
        # Violating intersection closure property
        with self.assertRaises(InvalidConvexityError):
            ConvexitySpace({0, 1, 2}, [frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({1, 2})])

    def test_paper_table_counts(self) -> None:
        """
        Validates total N-ary convexity structures generated against the precise paper metrics 
        for n=0, 1, 2, 3 (Table 1 and Table 2).
        """
        # --- N = 0 (Total: 2^n, Grounded: 1) ---
        for n in range(4):
            all_convs = list(generate_convexities(n, grounded_only=False))
            grounded_convs = list(generate_convexities(n, grounded_only=True))
            
            n_0_ary_total = sum(1 for c in all_convs if c.is_n_ary(0))
            n_0_ary_grounded = sum(1 for c in grounded_convs if c.is_n_ary(0))
            
            self.assertEqual(n_0_ary_total, 2**n)
            self.assertEqual(n_0_ary_grounded, 1)

        # --- N = 1 (A326878 Total and A000798 Grounded) ---
        # n = 2
        convs_n2 = list(generate_convexities(2, grounded_only=False))
        grounded_n2 = list(generate_convexities(2, grounded_only=True))
        self.assertEqual(sum(1 for c in convs_n2 if c.is_n_ary(1)), 7)
        self.assertEqual(sum(1 for c in grounded_n2 if c.is_n_ary(1)), 4)

        # n = 3
        convs_n3 = list(generate_convexities(3, grounded_only=False))
        grounded_n3 = list(generate_convexities(3, grounded_only=True))
        self.assertEqual(sum(1 for c in convs_n3 if c.is_n_ary(1)), 45)
        self.assertEqual(sum(1 for c in grounded_n3 if c.is_n_ary(1)), 29)

    def test_binomial_transform_identities(self) -> None:
        """Tests the foundational theorem identity presented in Section 2."""
        # Sequence data derived directly from Row N=1 of Table 2 (Grounded 1-ary)
        grounded_sequence = [1, 1, 4, 29, 355]
        # Expected sequence from Row N=1 of Table 1 (Total 1-ary)
        expected_total_sequence = [1, 2, 7, 45, 500]
        
        computed_transform = binomial_transform(grounded_sequence)
        self.assertEqual(computed_transform, expected_total_sequence)
        
        computed_inverse = inverse_binomial_transform(expected_total_sequence)
        self.assertEqual(computed_inverse, grounded_sequence)

    def test_isomorphism_reflections(self) -> None:
        """Validates the bijection mapping established in Lemma 1 and Lemma 5."""
        # Generate arbitrary ungrounded convexity space
        X = {0, 1, 2}
        # A convexity where the minimal convex set is {0}
        G_families = [frozenset({0, 1, 2}), frozenset({0, 1}), frozenset({0})]
        G = ConvexitySpace(X, G_families)
        
        self.assertFalse(G.is_grounded())
        self.assertEqual(G.minimal_convex_set(), frozenset({0}))
        
        # Mirror reflecting to a grounded space
        H = G.get_grounded_reflection()
        self.assertTrue(H.is_grounded())
        self.assertEqual(H.ground_set, frozenset({1, 2}))
        
        # Reconstructing back to original space
        G_reconstructed = ConvexitySpace.from_grounded(H, C={0})
        self.assertEqual(G, G_reconstructed)

if __name__ == '__main__':
    unittest.main()