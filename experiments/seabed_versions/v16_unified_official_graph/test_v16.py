from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Data


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
for path in (PROJECT_ROOT, CURRENT_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_evaluator import build_simple_edit_path, build_simple_graph  # noqa: E402
from relation_trainer import V16UnifiedOfficialGraphTrainer  # noqa: E402


class V16UnifiedCostTest(unittest.TestCase):
    def test_model_and_cost_views_must_match(self):
        data = Data(
            n=torch.tensor([[2, 2]]),
            x_indicator=torch.tensor([[0], [0], [1], [1]]),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            edge_labels=torch.tensor([1, 1, 1, 1]),
            unit_cost_edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            unit_cost_edge_labels=torch.tensor([1, 1, 1, 1]),
        )
        V16UnifiedOfficialGraphTrainer._assert_unified_pair(data)
        data.unit_cost_edge_labels[0] = 2
        with self.assertRaises(RuntimeError):
            V16UnifiedOfficialGraphTrainer._assert_unified_pair(data)

    def test_dense_cost_matches_official_last_write_path(self):
        trainer = object.__new__(V16UnifiedOfficialGraphTrainer)
        trainer.args = SimpleNamespace(cost_mode="unit")
        source = build_simple_graph(
            2,
            [[0, 1], [1, 0]],
            ["overwritten", "last"],
        )
        target = build_simple_graph(3, [[0, 1], [1, 2]], ["last", "insert"])
        # Pair-local relation IDs: last=1, insert=2. Symmetric projected edges.
        data = Data(
            n=torch.tensor([[2, 3]]),
            i_j=torch.tensor([[0, 1]]),
            ged=torch.tensor([1.0]),
            x_indicator=torch.tensor([[0], [0], [1], [1], [1]]),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]], dtype=torch.long
            ),
            edge_labels=torch.tensor([1, 1, 1, 1, 2, 2]),
        )
        data.unit_cost_edge_index = data.edge_index.clone()
        data.unit_cost_edge_labels = data.edge_labels.clone()
        solution = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
        dense_cost = trainer._compute_single_ged_from_dense_solution(solution, data)
        path_cost = build_simple_edit_path([0, 1], source, target)["total_cost"]
        self.assertEqual(dense_cost, path_cost)
        self.assertEqual(dense_cost, 2.0)

    def test_below_ground_truth_is_rejected(self):
        batch = Data(ged=torch.tensor([4.0, 5.0]))
        with self.assertRaises(RuntimeError):
            V16UnifiedOfficialGraphTrainer._assert_not_below_ground_truth(
                torch.tensor([4.0, 4.0]), batch, "test"
            )


if __name__ == "__main__":
    unittest.main()

