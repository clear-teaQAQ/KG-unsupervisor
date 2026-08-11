from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Data


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_trainer import (  # noqa: E402
    V15ProjectedRelationTrainer,
    project_last_write_edges,
)


class LastWriteProjectionTest(unittest.TestCase):
    def test_undirected_endpoint_keeps_last_raw_relation(self):
        projected = project_last_write_edges(
            [[4, 0], [0, 4], [1, 2], [4, 0]],
            ["r132", "r135", "r022", "r999"],
            [[1.0], [2.0], [3.0], [4.0]],
        )
        self.assertEqual(projected["graph"], [[1, 2], [4, 0]])
        self.assertEqual(projected["edge_ids"], ["r022", "r999"])
        self.assertEqual(projected["relation_features"], [[3.0], [4.0]])
        self.assertEqual(projected["selected_indices"], [2, 3])
        self.assertEqual(projected["dropped_edges"], 2)
        self.assertEqual(projected["parallel_endpoint_pairs"], 1)
        self.assertEqual(projected["max_multiplicity"], 3)

    def test_no_parallel_edges_is_an_exact_noop(self):
        projected = project_last_write_edges(
            [[2, 1], [0, 3]],
            ["a", "b"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(projected["graph"], [[2, 1], [0, 3]])
        self.assertEqual(projected["edge_ids"], ["a", "b"])
        self.assertEqual(projected["dropped_edges"], 0)

    def test_alignment_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            project_last_write_edges([[0, 1]], ["r"], [])


class UnitCostViewTest(unittest.TestCase):
    def test_single_ged_uses_preserved_v11_edges_not_model_edges(self):
        trainer = object.__new__(V15ProjectedRelationTrainer)
        trainer.args = SimpleNamespace(cost_mode="unit")
        data = Data(
            n=torch.tensor([[2, 2]]),
            x_indicator=torch.tensor([[0], [0], [1], [1]]),
            # Deliberately different model-input edges.
            edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            edge_labels=torch.tensor([1, 2], dtype=torch.long),
            # Identical unit-cost edge views for the two graphs.
            unit_cost_edge_index=torch.tensor(
                [[0, 1, 0, 1, 2, 3, 2, 3], [1, 0, 0, 1, 3, 2, 2, 3]],
                dtype=torch.long,
            ),
            unit_cost_edge_labels=torch.tensor(
                [1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.long
            ),
        )
        identity = torch.eye(2, dtype=torch.bool)
        self.assertEqual(
            trainer._compute_single_ged_from_dense_solution(identity, data),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
