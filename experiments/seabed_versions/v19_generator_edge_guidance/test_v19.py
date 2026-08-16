from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_models import RelationAwareDiffMatch  # noqa: E402
from v19_models import (  # noqa: E402
    BatchedGeneratorEdgeEvidence,
    RelationGuidedDiffMatch,
    attach_edge_reasoning_cache,
)


def make_pair(target_relation=1, edge_free=False):
    n1 = n2 = 3
    loops = torch.arange(6).repeat(2, 1)
    if edge_free:
        edge_index = loops
        edge_labels = torch.zeros(6, dtype=torch.long)
    else:
        source_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        target_edges = torch.tensor(
            [[3, 4], [4, 3], [4, 5], [5, 4]], dtype=torch.long
        ).t()
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
        m=torch.tensor([[0, 0]] if edge_free else [[1, 2]]),
    )
    attach_edge_reasoning_cache(data)
    return data


IDENTITY = torch.eye(3).reshape(-1, 1)


def diffusion_state(mapping):
    return F.one_hot(mapping.reshape(-1).long(), num_classes=2).float()


def model_args():
    return SimpleNamespace(
        hidden_dim=[8, 4],
        v19_edge_hidden_dim=8,
        v19_gate_init=0.0,
        diffusion_steps=1000,
    )


class GeneratorEvidenceTest(unittest.TestCase):
    def test_exact_and_wrong_relation_are_separated(self):
        reasoner = BatchedGeneratorEdgeEvidence(8, 1000)
        exact = reasoner.evidence_features(
            Batch.from_data_list([make_pair()]),
            diffusion_state(IDENTITY),
            torch.tensor([500.0]),
        )
        wrong = reasoner.evidence_features(
            Batch.from_data_list([make_pair(target_relation=3)]),
            diffusion_state(IDENTITY),
            torch.tensor([500.0]),
        )
        self.assertGreater(float(exact[:, 1].sum()), float(wrong[:, 1].sum()))
        self.assertGreater(float(wrong[:, 2].sum()), 0.0)
        torch.testing.assert_close(exact[:, 4], torch.full((9,), 0.5))

    def test_edge_free_pair_is_finite_and_has_no_edge_signal(self):
        reasoner = BatchedGeneratorEdgeEvidence(8, 1000)
        features = reasoner.evidence_features(
            Batch.from_data_list([make_pair(edge_free=True)]),
            diffusion_state(IDENTITY),
            torch.tensor([1.0]),
        )
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[:, 1:4], torch.zeros((9, 3)))

    def test_batching_matches_pairwise_features(self):
        reasoner = BatchedGeneratorEdgeEvidence(8, 1000)
        pair_a = make_pair()
        pair_b = make_pair(target_relation=3)
        mapping = diffusion_state(IDENTITY)
        separate = torch.cat(
            [
                reasoner.evidence_features(
                    Batch.from_data_list([pair_a]), mapping, torch.tensor([100.0])
                ),
                reasoner.evidence_features(
                    Batch.from_data_list([pair_b]), mapping, torch.tensor([700.0])
                ),
            ]
        )
        together = reasoner.evidence_features(
            Batch.from_data_list([pair_a, pair_b]),
            torch.cat([mapping, mapping]),
            torch.tensor([100.0, 700.0]),
        )
        torch.testing.assert_close(together, separate)


class GeneratorResidualTest(unittest.TestCase):
    def test_zero_gate_exactly_matches_original_generator(self):
        args = model_args()
        torch.manual_seed(7)
        baseline = RelationAwareDiffMatch(args, 2, 3)
        torch.manual_seed(7)
        guided = RelationGuidedDiffMatch(args, 2, 3)
        batch = Batch.from_data_list([make_pair()])
        mapping = IDENTITY
        timestep = torch.tensor([500.0])
        baseline.eval()
        guided.eval()
        with torch.no_grad():
            expected = baseline(batch, mapping, timestep)
            actual = guided(batch, mapping, timestep)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gate_receives_finite_nonzero_gradient(self):
        torch.manual_seed(11)
        guided = RelationGuidedDiffMatch(model_args(), 2, 3)
        output = guided(
            Batch.from_data_list([make_pair()]),
            IDENTITY,
            torch.tensor([500.0]),
        )
        output.sum().backward()
        gradient = guided.generator_edge_gate.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(float(gradient.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
