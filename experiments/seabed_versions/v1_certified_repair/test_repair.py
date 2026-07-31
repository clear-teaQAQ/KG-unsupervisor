import itertools
import unittest

import torch

from repair import (
    deterministic_labeled_adjacency,
    permutation_unit_cost,
    permutation_unit_costs,
    repair_mapping,
    size_lower_bound,
    unit_cost,
)


def reference_cost(mapping, adj_1, adj_2):
    n1 = adj_1.shape[0]
    n2 = adj_2.shape[0]
    unmatched = [node for node in range(n2) if node not in mapping.tolist()]
    permutation = torch.tensor(mapping.tolist() + unmatched)
    padded_adj_1 = torch.zeros((n2, n2), dtype=adj_1.dtype)
    padded_adj_1[:n1, :n1] = adj_1
    permuted_adj_2 = adj_2[permutation[:, None], permutation[None, :]]
    upper = torch.triu_indices(n2, n2, offset=1)
    return int(n2 - n1 + torch.count_nonzero(
        padded_adj_1[upper[0], upper[1]] != permuted_adj_2[upper[0], upper[1]]
    ).item())


class RepairTest(unittest.TestCase):
    def test_adjacency_uses_deterministic_last_write_for_duplicate_cells(self):
        edge_index = torch.tensor(
            [[0, 0, 2, 2, 1, 3], [1, 1, 3, 3, 0, 2]],
            dtype=torch.long,
        )
        edge_labels = torch.tensor([4, 7, 5, 9, 0, 0], dtype=torch.long)
        adj_1, adj_2 = deterministic_labeled_adjacency(
            2,
            2,
            edge_index,
            edge_labels,
            torch.device("cpu"),
        )
        self.assertEqual(int(adj_1[0, 1]), 7)
        self.assertEqual(int(adj_2[0, 1]), 9)
        self.assertEqual(int(adj_1[1, 0]), 0)
        self.assertEqual(int(adj_2[1, 0]), 0)

    def test_unit_cost_matches_v0_reference(self):
        adj_1 = torch.tensor([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
        adj_2 = torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 2], [1, 0, 0, 2], [0, 2, 2, 0]]
        )
        for mapping_values in itertools.permutations(range(4), 3):
            mapping = torch.tensor(mapping_values)
            self.assertEqual(unit_cost(mapping, adj_1, adj_2), reference_cost(mapping, adj_1, adj_2))

    def test_unit_cost_matches_reference_with_asymmetric_relation_labels(self):
        adj_1 = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])
        adj_2 = torch.tensor(
            [[0, 1, 0, 5], [2, 0, 3, 0], [0, 4, 0, 6], [7, 0, 8, 0]]
        )
        for mapping_values in itertools.permutations(range(4), 3):
            mapping = torch.tensor(mapping_values)
            self.assertEqual(unit_cost(mapping, adj_1, adj_2), reference_cost(mapping, adj_1, adj_2))

    def test_complete_permutation_cost_matches_reference(self):
        adj_1 = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])
        adj_2 = torch.tensor(
            [[0, 1, 0, 5], [2, 0, 3, 0], [0, 4, 0, 6], [7, 0, 8, 0]]
        )
        for permutation_values in itertools.permutations(range(4)):
            permutation = torch.tensor(permutation_values)
            mapping = permutation[:3]
            self.assertEqual(
                permutation_unit_cost(permutation, adj_1, adj_2),
                reference_cost(mapping, adj_1, adj_2),
            )

        permutations = torch.tensor(list(itertools.permutations(range(4))))
        batch_costs = permutation_unit_costs(permutations, adj_1, adj_2)
        scalar_costs = torch.tensor(
            [permutation_unit_cost(permutation, adj_1, adj_2) for permutation in permutations]
        )
        self.assertTrue(torch.equal(batch_costs.cpu(), scalar_costs))

    def test_unit_cost_rejects_non_injective_mapping(self):
        adj = torch.zeros((3, 3), dtype=torch.long)
        with self.assertRaises(ValueError):
            unit_cost(torch.tensor([0, 0, 2]), adj, adj)

    def test_repair_can_swap_two_matched_targets(self):
        adj_1 = torch.tensor([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
        adj_2 = adj_1.clone()
        initial = torch.tensor([1, 0, 2])
        result = repair_mapping(initial, adj_1, adj_2)
        self.assertEqual(result.initial_cost, 2)
        self.assertEqual(result.final_cost, 0)
        self.assertTrue(result.certified)

    def test_repair_can_replace_with_unmatched_target(self):
        adj_1 = torch.tensor([[0, 1], [1, 0]])
        adj_2 = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        initial = torch.tensor([0, 2])
        result = repair_mapping(initial, adj_1, adj_2)
        self.assertEqual(size_lower_bound(adj_1, adj_2), 1)
        self.assertEqual(result.initial_cost, 3)
        self.assertEqual(result.final_cost, 1)
        self.assertTrue(result.certified)

    def test_repair_never_increases_cost(self):
        generator = torch.Generator().manual_seed(7)
        for _ in range(20):
            upper_1 = torch.randint(0, 4, (4, 4), generator=generator)
            upper_2 = torch.randint(0, 4, (6, 6), generator=generator)
            adj_1 = torch.triu(upper_1, diagonal=1)
            adj_1 = adj_1 + adj_1.t()
            adj_2 = torch.triu(upper_2, diagonal=1)
            adj_2 = adj_2 + adj_2.t()
            mapping = torch.randperm(6, generator=generator)[:4]
            result = repair_mapping(mapping, adj_1, adj_2, max_iterations=5)
            self.assertLessEqual(result.final_cost, result.initial_cost)
            self.assertGreaterEqual(result.final_cost, result.lower_bound)


if __name__ == "__main__":
    unittest.main()
