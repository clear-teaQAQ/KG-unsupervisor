"""Auxiliary modules for V12 semantic preference learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class SemanticPreferenceCritic(torch.nn.Module):
    """Score node-pair candidates from fixed KG semantic compatibility features."""

    def __init__(self, input_dim: int = 5, hidden_dims=(64, 32)):
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.append(torch.nn.Linear(left, right))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return torch.zeros((0, 1), device=features.device, dtype=features.dtype)
        return self.network(features)


class SparseCandidateMatcher:
    """Simple top-r + epsilon candidate selector for future sparse wiring."""

    def __init__(self, top_r: int = 32, random_epsilon: float = 0.05):
        self.top_r = int(top_r)
        self.random_epsilon = float(random_epsilon)

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("SparseCandidateMatcher expects a 2D score matrix.")
        if scores.numel() == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        top_r = min(self.top_r, scores.shape[-1])
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if top_r > 0:
            top_idx = torch.topk(scores, k=top_r, dim=-1).indices
            mask.scatter_(dim=-1, index=top_idx, value=True)
        if self.random_epsilon > 0:
            noise_mask = torch.rand_like(scores) < self.random_epsilon
            mask = mask | noise_mask
        return mask


@dataclass
class PairPreferenceFeatures:
    """Candidate-level features consumed by the semantic critic."""

    entity_similarity: torch.Tensor
    outgoing_relation_similarity: torch.Tensor
    incoming_relation_similarity: torch.Tensor
    degree_similarity: torch.Tensor
    exact_entity_anchor: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        parts = [
            self.entity_similarity,
            self.outgoing_relation_similarity,
            self.incoming_relation_similarity,
            self.degree_similarity,
            self.exact_entity_anchor,
        ]
        return torch.stack([p.reshape(-1) for p in parts], dim=-1)
