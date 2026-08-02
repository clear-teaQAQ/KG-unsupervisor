import unittest

import torch

from kg_objective import (
    exact_anchor_mask,
    lexicographic_update_masks,
    selected_anchor_counts,
)


class LexicographicObjectiveTest(unittest.TestCase):
    def test_lower_ged_always_wins_even_with_fewer_anchors(self):
        strict, semantic = lexicographic_update_masks(
            torch.tensor([4.0]),
            torch.tensor([5.0]),
            torch.tensor([0.0]),
            torch.tensor([3.0]),
        )
        self.assertTrue(strict.item())
        self.assertFalse(semantic.item())

    def test_equal_ged_with_more_anchors_wins(self):
        strict, semantic = lexicographic_update_masks(
            torch.tensor([5.0]),
            torch.tensor([5.0]),
            torch.tensor([3.0]),
            torch.tensor([2.0]),
        )
        self.assertFalse(strict.item())
        self.assertTrue(semantic.item())

    def test_equal_objectives_do_not_churn(self):
        strict, semantic = lexicographic_update_masks(
            torch.tensor([5.0]),
            torch.tensor([5.0]),
            torch.tensor([2.0]),
            torch.tensor([2.0]),
        )
        self.assertFalse(strict.item())
        self.assertFalse(semantic.item())

    def test_more_anchors_never_override_worse_ged(self):
        strict, semantic = lexicographic_update_masks(
            torch.tensor([6.0]),
            torch.tensor([5.0]),
            torch.tensor([4.0]),
            torch.tensor([0.0]),
        )
        self.assertFalse(strict.item())
        self.assertFalse(semantic.item())

    def test_exact_anchor_count_uses_selected_mapping_edges(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [0.0, 1.0],
            ]
        )
        mapping_edges = torch.tensor(
            [[0, 0, 0, 1, 1, 1], [2, 3, 4, 2, 3, 4]]
        )
        anchors = exact_anchor_mask(features, mapping_edges)
        solution = torch.tensor([[1], [0], [0], [0], [0], [1]])
        counts = selected_anchor_counts(
            solution,
            anchors,
            torch.zeros(6, dtype=torch.long),
            1,
        )
        self.assertEqual(anchors.squeeze(-1).tolist(), [True, False, False, False, False, True])
        self.assertEqual(counts.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
