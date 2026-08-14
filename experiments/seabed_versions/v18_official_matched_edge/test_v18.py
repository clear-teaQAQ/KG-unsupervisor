from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Batch, Data


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_models import RelationAwareDiscriminator  # noqa: E402
from v18_models import (  # noqa: E402
    BatchedOfficialEdgeReasoner,
    OfficialMatchedEdgeDiscriminator,
    attach_edge_reasoning_cache,
)


def make_pair(target_relation=1):
    n1 = n2 = 3
    source_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    target_edges = torch.tensor(
        [[3, 4], [4, 3], [4, 5], [5, 4]], dtype=torch.long
    ).t()
    loops = torch.arange(6).repeat(2, 1)
    edge_index = torch.cat([source_edges, target_edges, loops], dim=1)
    edge_labels = torch.tensor(
        [1, 1, target_relation, target_relation, 2, 2] + [0] * 6,
        dtype=torch.long,
    )
    rows = torch.arange(3).repeat_interleave(3)
    cols = torch.arange(3).repeat(3) + 3
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] * 2),
        edge_index=edge_index,
        edge_attr=torch.zeros((edge_index.shape[1], 3)),
        edge_labels=edge_labels,
        x_indicator=torch.cat([torch.zeros((3, 1)), torch.ones((3, 1))]),
        edge_index_mapping=torch.stack([rows, cols]),
        edge_attr_mapping=torch.zeros((9, 1)),
        n=torch.tensor([[n1, n2]]),
        m=torch.tensor([[1, 2]]),
    )
    attach_edge_reasoning_cache(data)
    return data


IDENTITY = torch.eye(3).reshape(-1, 1)
BROKEN = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float
).reshape(-1, 1)


class BatchedEvidenceTest(unittest.TestCase):
    def test_exact_wrong_and_broken_events(self):
        reasoner = BatchedOfficialEdgeReasoner(8)
        exact = reasoner.event_features(Batch.from_data_list([make_pair()]), IDENTITY)
        torch.testing.assert_close(
            exact["source_events"], torch.tensor([[1.0, 0.0, 0.0]])
        )
        self.assertEqual(float(exact["target_events"][:, 2].sum()), 1.0)

        wrong = reasoner.event_features(
            Batch.from_data_list([make_pair(target_relation=3)]), IDENTITY
        )
        torch.testing.assert_close(
            wrong["source_events"], torch.tensor([[0.0, 1.0, 0.0]])
        )
        broken = reasoner.event_features(
            Batch.from_data_list([make_pair()]), BROKEN
        )
        self.assertGreater(
            float(broken["source_events"][:, 2].sum()),
            float(exact["source_events"][:, 2].sum()),
        )

    def test_batching_matches_pairwise_scores(self):
        reasoner = BatchedOfficialEdgeReasoner(8)
        pair_a = make_pair()
        pair_b = make_pair(target_relation=3)
        with torch.no_grad():
            separate = torch.cat(
                [
                    reasoner(Batch.from_data_list([pair_a]), IDENTITY),
                    reasoner(Batch.from_data_list([pair_b]), IDENTITY),
                ]
            )
            together = reasoner(
                Batch.from_data_list([pair_a, pair_b]),
                torch.cat([IDENTITY, IDENTITY]),
            )
        torch.testing.assert_close(together, separate)

    def test_soft_mapping_has_finite_gradient(self):
        reasoner = BatchedOfficialEdgeReasoner(8)
        soft = (IDENTITY * 0.7 + 0.1).requires_grad_(True)
        reasoner(Batch.from_data_list([make_pair()]), soft).sum().backward()
        self.assertIsNotNone(soft.grad)
        self.assertTrue(torch.isfinite(soft.grad).all())
        self.assertGreater(float(soft.grad.abs().sum()), 0.0)


class ZeroGateParityTest(unittest.TestCase):
    def test_zero_gate_exactly_matches_v11(self):
        args = SimpleNamespace(
            d_hidden_dim=[8, 4],
            v18_edge_hidden_dim=8,
            v18_gate_init=0.0,
        )
        torch.manual_seed(7)
        baseline = RelationAwareDiscriminator(args, 2, 3)
        full = OfficialMatchedEdgeDiscriminator(args, 2, 3)
        full.load_state_dict(baseline.state_dict(), strict=False)
        batch = Batch.from_data_list([make_pair()])
        with torch.no_grad():
            expected = baseline(batch, IDENTITY)
            actual = full(batch, IDENTITY)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

