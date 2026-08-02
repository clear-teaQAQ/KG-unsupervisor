import unittest

import numpy as np

from embedding_anchor import exact_embedding_matrix, mapping_anchor_count


class ExactEmbeddingAnchorTest(unittest.TestCase):
    def test_exact_matrix_uses_raw_equality(self):
        matrix = exact_embedding_matrix(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1e-12]],
        )
        np.testing.assert_array_equal(
            matrix,
            [[False, True, False], [True, False, False]],
        )

    def test_mapping_anchor_count(self):
        matrix = np.asarray([[False, True], [True, False]])
        self.assertEqual(mapping_anchor_count((1, 0), matrix), 2)
        self.assertEqual(mapping_anchor_count((0, 1), matrix), 0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_embedding_matrix([[1.0, 2.0]], [[1.0]])


if __name__ == "__main__":
    unittest.main()

