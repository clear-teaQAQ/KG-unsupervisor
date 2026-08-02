import unittest

import numpy as np

from semantic_tie import (
    cosine_similarity_matrix,
    matrix_mapping_score,
    optimize_equal_cost_mapping,
    shared_entity_alignment,
)


class SemanticTieTest(unittest.TestCase):
    def test_id_oracle_repairs_equal_cost_swap(self):
        source_ids = ["a", "b"]
        target_ids = ["a", "b"]
        result = optimize_equal_cost_mapping(
            [1, 0],
            2,
            dual_cost=lambda mapping: (0, 0),
            semantic_score=lambda mapping: shared_entity_alignment(
                mapping, source_ids, target_ids
            )[0],
        )
        self.assertEqual(result.mapping, (0, 1))
        self.assertEqual(result.initial_score, 0)
        self.assertEqual(result.final_score, 2)

    def test_rejects_semantic_move_that_changes_either_cost(self):
        result = optimize_equal_cost_mapping(
            [1, 0],
            2,
            dual_cost=lambda mapping: (0, 0) if tuple(mapping) == (1, 0) else (0, 1),
            semantic_score=lambda mapping: int(tuple(mapping) == (0, 1)),
        )
        self.assertEqual(result.mapping, (1, 0))
        self.assertEqual(result.final_score, 0)

    def test_replacement_can_use_unmatched_target(self):
        result = optimize_equal_cost_mapping(
            [0],
            2,
            dual_cost=lambda mapping: (1, 1),
            semantic_score=lambda mapping: int(mapping[0] == 1),
        )
        self.assertEqual(result.mapping, (1,))

    def test_cosine_score_prefers_matching_rows(self):
        similarity = cosine_similarity_matrix(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertAlmostEqual(matrix_mapping_score([0, 1], similarity), 2.0)
        self.assertAlmostEqual(matrix_mapping_score([1, 0], similarity), 0.0)

    def test_zero_vectors_have_finite_similarity(self):
        similarity = cosine_similarity_matrix([[0.0, 0.0]], [[0.0, 0.0]])
        self.assertTrue(np.isfinite(similarity).all())
        self.assertEqual(float(similarity[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
