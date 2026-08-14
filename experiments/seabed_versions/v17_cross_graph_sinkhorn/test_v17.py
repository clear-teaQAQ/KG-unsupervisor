import sys
from pathlib import Path

import torch
from types import SimpleNamespace
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v17_models import (
    CrossGraphSinkhornMatcher,
    direct_sinkhorn_candidates,
    direct_sinkhorn_rollout,
)
from src.GEDRanker.loss_fn import mapping_loss


def test_output_shape_and_gradient():
    args = SimpleNamespace(hidden_dim=[16, 8])
    model = CrossGraphSinkhornMatcher(args, 4, 3)
    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 1], [1, 0, 3, 2, 4, 4]]),
        edge_attr=torch.randn(6, 3),
        batch=torch.zeros(5, dtype=torch.long),
        x_indicator=torch.tensor([[0.], [0.], [1.], [1.], [1.]]),
        n=torch.tensor([[2, 3]]),
    )
    output = model(data)
    assert output.shape == (6, 1)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_direct_candidates_are_injective():
    torch.manual_seed(7)
    logits = torch.zeros(6, 1)
    candidates, soft = direct_sinkhorn_candidates(
        logits,
        source_nodes=2,
        target_nodes=3,
        sample_count=32,
        tau=1.0,
        iterations=20,
        stochastic=True,
        include_deterministic=True,
    )
    assert candidates.shape == (32, 2, 3)
    assert soft.shape == (32, 2, 3)
    assert torch.all(candidates.sum(dim=-1) == 1)
    assert torch.all(candidates.sum(dim=-2) <= 1)
    assert torch.unique(candidates.reshape(32, -1), dim=0).shape[0] > 1
    assert torch.allclose(soft.sum(dim=-1), torch.ones_like(soft.sum(dim=-1)), atol=2e-2)
    assert torch.all(soft.sum(dim=-2) <= 1.0 + 2e-2)


def test_direct_rollout_gradient_and_batch_order():
    logits = torch.randn(10, 1, requires_grad=True)
    batch = SimpleNamespace(
        n=torch.tensor([[2, 3], [2, 2]]),
        batch=torch.tensor([0] * 5 + [1] * 4),
        edge_index_mapping=torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 5, 5, 6, 6],
                [2, 3, 4, 2, 3, 4, 7, 8, 7, 8],
            ]
        ),
    )
    hard, soft = direct_sinkhorn_rollout(logits, batch, tau=1.0, iterations=20)
    assert hard.shape == logits.shape
    assert soft.shape == logits.shape
    assert hard.dtype == logits.dtype
    assert torch.equal(hard[:6].reshape(2, 3).sum(dim=1), torch.ones(2))
    assert torch.all(hard[:6].reshape(2, 3).sum(dim=0) <= 1)
    assert torch.equal(hard[6:].reshape(2, 2).sum(dim=1), torch.ones(2))
    updated_best_mapping = hard.bool()
    loss = mapping_loss(logits, batch, updated_best_mapping.float())
    (soft.sum() + loss).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_batched_rollout_matches_pairwise_reference():
    logits = torch.tensor(
        [[2.0], [0.3], [-0.2], [0.1], [1.7], [0.4], [1.0], [-0.5], [-0.1], [1.3]],
        requires_grad=True,
    )
    batch = SimpleNamespace(
        n=torch.tensor([[2, 3], [2, 2]]),
        batch=torch.tensor([0] * 5 + [1] * 4),
        edge_index_mapping=torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 5, 5, 6, 6],
                [2, 3, 4, 2, 3, 4, 7, 8, 7, 8],
            ]
        ),
    )
    hard, soft = direct_sinkhorn_rollout(
        logits, batch, tau=1.0, iterations=20, stochastic=False
    )
    reference_hard_1, reference_soft_1 = direct_sinkhorn_candidates(
        logits[:6], 2, 3, 1, 1.0, 20, stochastic=False
    )
    reference_hard_2, reference_soft_2 = direct_sinkhorn_candidates(
        logits[6:], 2, 2, 1, 1.0, 20, stochastic=False
    )
    assert torch.equal(hard[:6], reference_hard_1[0].reshape(-1, 1))
    assert torch.equal(hard[6:], reference_hard_2[0].reshape(-1, 1))
    assert torch.allclose(soft[:6], reference_soft_1[0].reshape(-1, 1), atol=1e-6)
    assert torch.allclose(soft[6:], reference_soft_2[0].reshape(-1, 1), atol=1e-6)


if __name__ == "__main__":
    test_output_shape_and_gradient()
    test_direct_candidates_are_injective()
    test_direct_rollout_gradient_and_batch_order()
    test_batched_rollout_matches_pairwise_reference()
    print("V17 tests passed")
