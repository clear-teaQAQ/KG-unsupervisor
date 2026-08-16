"""Cost-preserving relation evidence inside the diffusion generator."""

from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V18_DIR = CURRENT_DIR.parent / "v18_official_matched_edge"
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V18_DIR, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_models import RelationAwareDiffMatch  # noqa: E402
from v18_models import (  # noqa: E402
    OfficialMatchedEdgeDiscriminator,
    _offsets,
    _pair_ids,
    attach_edge_reasoning_cache,
)


class BatchedGeneratorEdgeEvidence(torch.nn.Module):
    """Compute per-node-pair edge support from the current diffusion state.

    For a candidate pair (u, a), every pair of incident source/target edges
    contributes the current assignment value of their opposite endpoints. The
    cache distinguishes exact relation labels from topology-only support.
    """

    def __init__(self, hidden_dim=32, diffusion_steps=1000):
        super().__init__()
        self.diffusion_steps = max(int(diffusion_steps), 1)
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(5, int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def evidence_features(self, data, mapping_attr, timestep):
        # Diffusion passes a two-class state [not-matched, matched], whereas
        # direct evidence tests may pass the matched probability alone.
        if mapping_attr.ndim > 1 and mapping_attr.shape[-1] > 1:
            mapping = mapping_attr[..., 1].reshape(-1).float()
        else:
            mapping = mapping_attr.reshape(-1).float()
        mapping_counts = data.v18_mapping_count.reshape(-1).long()
        mapping_pair = _pair_ids(mapping_counts)
        exact_support = mapping.new_zeros(mapping.shape)
        topology_support = mapping.new_zeros(mapping.shape)
        possible_support = mapping.new_zeros(mapping.shape)

        if data.v18_combo_mapping_positions.numel():
            combo_counts = data.v18_combo_count.reshape(-1).long()
            combo_pair = _pair_ids(combo_counts)
            positions = data.v18_combo_mapping_positions.long()
            positions = positions + _offsets(mapping_counts)[combo_pair, None]

            # For (u,a), the opposite endpoint assignment (v,b) is evidence;
            # the remaining three orientations follow the same rule.
            opposite = positions[:, [1, 0, 3, 2]]
            destinations = positions.reshape(-1)
            support = mapping[opposite].reshape(-1)
            exact_mask = data.v18_combo_exact.reshape(-1, 1).expand(-1, 4)
            exact_mask = exact_mask.reshape(-1).to(mapping.dtype)

            topology_support.index_add_(0, destinations, support)
            exact_support.index_add_(0, destinations, support * exact_mask)
            possible_support.index_add_(0, destinations, torch.ones_like(support))

        active = (possible_support > 0).to(mapping.dtype)
        denominator = possible_support.clamp_min(1.0)
        topology_ratio = (topology_support / denominator).clamp(0.0, 1.0)
        exact_ratio = (exact_support / denominator).clamp(0.0, 1.0)
        wrong_relation_ratio = (topology_ratio - exact_ratio).clamp_min(0.0)
        missing_ratio = (1.0 - topology_ratio) * active

        timestep = timestep.reshape(-1).to(mapping.device, mapping.dtype)
        time_feature = timestep[mapping_pair] / float(self.diffusion_steps)
        return torch.stack(
            [
                mapping.clamp(0.0, 1.0),
                exact_ratio,
                wrong_relation_ratio,
                missing_ratio,
                time_feature,
            ],
            dim=1,
        )

    def forward(self, data, mapping_attr, timestep):
        return self.residual(
            self.evidence_features(data, mapping_attr, timestep)
        )


class RelationGuidedDiffMatch(RelationAwareDiffMatch):
    """V18 diffusion generator plus a bounded relation-evidence residual."""

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim)

        # Do not shift discriminator or graph-pair initialization relative to
        # the V18 baseline merely because this optional branch exists.
        rng_state = torch.random.get_rng_state()
        self.generator_edge_evidence = BatchedGeneratorEdgeEvidence(
            hidden_dim=int(getattr(args, "v19_edge_hidden_dim", 32)),
            diffusion_steps=int(getattr(args, "diffusion_steps", 1000)),
        )
        torch.random.set_rng_state(rng_state)
        self.generator_edge_gate = torch.nn.Parameter(
            torch.tensor(float(getattr(args, "v19_gate_init", 0.0)))
        )

    @property
    def effective_generator_edge_gate(self):
        return torch.tanh(self.generator_edge_gate)

    def forward(self, data, noise_mapping_attr, timestep):
        base_logits = super().forward(data, noise_mapping_attr, timestep)
        edge_residual = self.generator_edge_evidence(
            data, noise_mapping_attr, timestep
        ).to(base_logits.dtype)
        return base_logits + self.effective_generator_edge_gate * edge_residual


__all__ = [
    "BatchedGeneratorEdgeEvidence",
    "OfficialMatchedEdgeDiscriminator",
    "RelationAwareDiffMatch",
    "RelationGuidedDiffMatch",
    "attach_edge_reasoning_cache",
]
