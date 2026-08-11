from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch_geometric.data import Batch, Data


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v14_models import (  # noqa: E402
    MatchingConditionedEdgeDiscriminator,
    MatchingConditionedEdgeReasoner,
    RelationAwareDiscriminator,
)


def make_pair(target_first_relation=1):
    graph_1_edges = torch.tensor(
        [[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]], dtype=torch.long
    )
    graph_2_edges = torch.tensor(
        [
            [3, 4, 4, 5, 4, 3, 5, 4, 3, 4, 5],
            [4, 5, 3, 4, 3, 4, 4, 5, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    edge_index = torch.cat([graph_1_edges, graph_2_edges], dim=1)
    graph_1_labels = torch.tensor([1, 1, 0, 0, 0], dtype=torch.long)
    graph_2_labels = torch.tensor(
        [
            target_first_relation,
            2,
            target_first_relation,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=torch.long,
    )
    rows = torch.arange(3).repeat_interleave(3)
    cols = torch.arange(3).repeat(3) + 3
    data = Data(
        x=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] * 2,
            dtype=torch.float,
        ),
        edge_index=edge_index,
        edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.float),
        edge_labels=torch.cat([graph_1_labels, graph_2_labels]),
        x_indicator=torch.cat([torch.zeros((3, 1)), torch.ones((3, 1))]),
        edge_index_mapping=torch.stack([rows, cols]),
        edge_attr_mapping=torch.zeros((9, 1)),
        n=torch.tensor([[3, 3]]),
        m=torch.tensor([[1, 2]]),
    )
    return Batch.from_data_list([data])


def mapping(values, requires_grad=False):
    value = torch.tensor(values, dtype=torch.float).reshape(9, 1).clone().detach()
    return value.requires_grad_(requires_grad)


IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]
BROKEN = [1, 0, 0, 0, 0, 1, 0, 1, 0]


class EdgeEvidenceTest(unittest.TestCase):
    def test_exact_wrong_and_extra_events(self):
        reasoner = MatchingConditionedEdgeReasoner(hidden_dim=8)
        data = make_pair(target_first_relation=1)
        evidence = reasoner.pair_event_features(
            data, mapping(IDENTITY).reshape(3, 3), 0, 0
        )
        torch.testing.assert_close(
            evidence["source_events"], torch.tensor([[1.0, 0.0, 0.0]])
        )
        self.assertEqual(evidence["target_events"].shape, (2, 3))
        self.assertEqual(float(evidence["target_events"][:, 2].sum()), 1.0)

        wrong = reasoner.pair_event_features(
            make_pair(target_first_relation=3),
            mapping(IDENTITY).reshape(3, 3),
            0,
            0,
        )
        torch.testing.assert_close(
            wrong["source_events"], torch.tensor([[0.0, 1.0, 0.0]])
        )

    def test_hard_evidence_reconstructs_unit_edge_cost(self):
        reasoner = MatchingConditionedEdgeReasoner(hidden_dim=8)
        data = make_pair(target_first_relation=1)
        evidence = reasoner.pair_event_features(
            data, mapping(IDENTITY).reshape(3, 3), 0, 0
        )
        reconstructed = (
            evidence["source_edge_count"]
            + evidence["target_edge_count"]
            - evidence["topology_overlap"]
            - evidence["exact_overlap"]
        )
        self.assertEqual(float(reconstructed), 1.0)

        broken = reasoner.pair_event_features(
            data, mapping(BROKEN).reshape(3, 3), 0, 0
        )
        self.assertGreater(
            float(broken["source_events"][:, 2].sum()),
            float(evidence["source_events"][:, 2].sum()),
        )

    def test_soft_matching_reasoner_is_differentiable(self):
        reasoner = MatchingConditionedEdgeReasoner(hidden_dim=8)
        soft = mapping([0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8], True)
        score = reasoner(make_pair(), soft).sum()
        score.backward()
        self.assertIsNotNone(soft.grad)
        self.assertTrue(torch.isfinite(soft.grad).all())
        self.assertGreater(float(soft.grad.abs().sum()), 0.0)


class ZeroGateParityTest(unittest.TestCase):
    def test_zero_gate_is_exact_v11_score(self):
        args = SimpleNamespace(
            d_hidden_dim=[8, 4],
            v14_edge_hidden_dim=8,
            v14_gate_init=0.0,
        )
        torch.manual_seed(7)
        baseline = RelationAwareDiscriminator(args, number_of_labels=2, relation_dim=3)
        full = MatchingConditionedEdgeDiscriminator(
            args, number_of_labels=2, relation_dim=3
        )
        full.load_state_dict(baseline.state_dict(), strict=False)
        data = make_pair()
        candidate = mapping(IDENTITY)
        with torch.no_grad():
            expected = baseline(data, candidate)
            actual = full(data, candidate)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
