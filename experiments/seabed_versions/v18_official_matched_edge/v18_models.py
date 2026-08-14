"""Batched matching-conditioned edge reasoning for the V16 official graph."""

from pathlib import Path
import sys

import torch
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_models import (  # noqa: E402
    RelationAwareDiffMatch,
    RelationAwareDiscriminator,
)


def _empty_long(rows=0, columns=None):
    shape = (rows,) if columns is None else (rows, columns)
    return torch.empty(shape, dtype=torch.long)


def attach_edge_reasoning_cache(data):
    """Attach vectorized exact-edge correspondence indices to one graph pair.

    The cache is derived from the same projected edge tensors used by V16 for
    representation and unit-cost evaluation. Malformed or edge-free inputs
    receive an empty cache so training diagnostics never abort an epoch.
    """
    n1 = int(data.n[0, 0].item())
    n2 = int(data.n[0, 1].item())
    valid = (
        data.edge_index.ndim == 2
        and data.edge_index.shape[0] == 2
        and data.edge_labels.numel() == data.edge_index.shape[1]
    )

    if valid:
        source, target = data.edge_index
        labels = data.edge_labels
        graph_id = data.x_indicator[source].reshape(-1).long()
        source_mask = (graph_id == 0) & (labels > 0) & (source < target)
        target_mask = (graph_id == 1) & (labels > 0) & (source < target)
        source_edges = data.edge_index[:, source_mask].t().contiguous()
        target_edges = (data.edge_index[:, target_mask] - n1).t().contiguous()
        source_labels = labels[source_mask].long()
        target_labels = labels[target_mask].long()
        if source_edges.numel():
            valid = bool(
                (source_edges >= 0).all()
                and (source_edges < n1).all()
            )
        if valid and target_edges.numel():
            valid = bool(
                (target_edges >= 0).all()
                and (target_edges < n2).all()
            )
    else:
        source_edges = _empty_long(0, 2)
        target_edges = _empty_long(0, 2)
        source_labels = _empty_long()
        target_labels = _empty_long()

    if not valid:
        source_edges = _empty_long(0, 2)
        target_edges = _empty_long(0, 2)
        source_labels = _empty_long()
        target_labels = _empty_long()

    source_count = int(source_edges.shape[0])
    target_count = int(target_edges.shape[0])
    combo_count = source_count * target_count
    if combo_count:
        source_ids = torch.arange(source_count).repeat_interleave(target_count)
        target_ids = torch.arange(target_count).repeat(source_count)
        source_combo = source_edges[source_ids]
        target_combo = target_edges[target_ids]
        u, v = source_combo[:, 0], source_combo[:, 1]
        a, b = target_combo[:, 0], target_combo[:, 1]
        mapping_positions = torch.stack(
            [u * n2 + a, v * n2 + b, u * n2 + b, v * n2 + a], dim=1
        )
        exact = source_labels[source_ids] == target_labels[target_ids]
    else:
        source_ids = _empty_long()
        target_ids = _empty_long()
        mapping_positions = _empty_long(0, 4)
        exact = torch.empty(0, dtype=torch.bool)

    # Attribute names intentionally avoid "index" so PyG does not offset the
    # pair-local values while batching. The reasoner applies offsets in one
    # vectorized operation on the GPU.
    data.v18_mapping_count = torch.tensor([n1 * n2], dtype=torch.long)
    data.v18_source_edge_count = torch.tensor([source_count], dtype=torch.long)
    data.v18_target_edge_count = torch.tensor([target_count], dtype=torch.long)
    data.v18_combo_count = torch.tensor([combo_count], dtype=torch.long)
    data.v18_combo_mapping_positions = mapping_positions
    data.v18_combo_source_edge = source_ids
    data.v18_combo_target_edge = target_ids
    data.v18_combo_exact = exact
    return {
        "valid": valid,
        "source_edges": source_count,
        "target_edges": target_count,
        "edge_combinations": combo_count,
    }


def _offsets(counts):
    if counts.numel() == 0:
        return counts
    return torch.cat([counts.new_zeros(1), torch.cumsum(counts[:-1], dim=0)])


def _pair_ids(counts):
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )


class BatchedOfficialEdgeReasoner(torch.nn.Module):
    """Score exact/wrong/broken edge events without per-pair Python loops."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.event_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.graph_scorer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4 + 4, self.hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _events(exact, topology):
        wrong_relation = (topology - exact).clamp_min(0)
        broken_or_extra = (1.0 - topology).clamp_min(0)
        return torch.stack([exact, wrong_relation, broken_or_extra], dim=-1)

    def event_features(self, data, mapping_attr):
        mapping = mapping_attr.reshape(-1)
        batch_size = int(data.n.shape[0])
        mapping_counts = data.v18_mapping_count.reshape(-1).long()
        source_counts = data.v18_source_edge_count.reshape(-1).long()
        target_counts = data.v18_target_edge_count.reshape(-1).long()
        combo_counts = data.v18_combo_count.reshape(-1).long()
        total_source = int(source_counts.sum().item())
        total_target = int(target_counts.sum().item())

        source_topology = mapping.new_zeros(total_source)
        source_exact = mapping.new_zeros(total_source)
        target_topology = mapping.new_zeros(total_target)
        target_exact = mapping.new_zeros(total_target)

        if data.v18_combo_mapping_positions.numel():
            combo_pair = _pair_ids(combo_counts)
            mapping_positions = data.v18_combo_mapping_positions.long()
            mapping_positions = mapping_positions + _offsets(mapping_counts)[
                combo_pair, None
            ]
            support = (
                mapping[mapping_positions[:, 0]] * mapping[mapping_positions[:, 1]]
                + mapping[mapping_positions[:, 2]]
                * mapping[mapping_positions[:, 3]]
            )
            exact_support = support * data.v18_combo_exact.to(mapping.dtype)
            source_ids = data.v18_combo_source_edge.long() + _offsets(source_counts)[
                combo_pair
            ]
            target_ids = data.v18_combo_target_edge.long() + _offsets(target_counts)[
                combo_pair
            ]
            source_topology.index_add_(0, source_ids, support)
            source_exact.index_add_(0, source_ids, exact_support)
            target_topology.index_add_(0, target_ids, support)
            target_exact.index_add_(0, target_ids, exact_support)

        return {
            "source_events": self._events(source_exact, source_topology),
            "target_events": self._events(target_exact, target_topology),
            "source_pair": _pair_ids(source_counts),
            "target_pair": _pair_ids(target_counts),
            "mapping_pair": _pair_ids(mapping_counts),
            "batch_size": batch_size,
        }

    def forward(self, data, mapping_attr):
        evidence = self.event_features(data, mapping_attr)
        batch_size = evidence["batch_size"]
        source_encoded = self.event_encoder(evidence["source_events"])
        target_encoded = self.event_encoder(evidence["target_events"])
        source_mean = global_mean_pool(
            source_encoded, evidence["source_pair"], size=batch_size
        )
        source_max = global_max_pool(
            source_encoded, evidence["source_pair"], size=batch_size
        )
        target_mean = global_mean_pool(
            target_encoded, evidence["target_pair"], size=batch_size
        )
        target_max = global_max_pool(
            target_encoded, evidence["target_pair"], size=batch_size
        )

        n1 = data.n[:, 0].to(mapping_attr.dtype)
        n2 = data.n[:, 1].to(mapping_attr.dtype)
        source_counts = data.v18_source_edge_count.reshape(-1).to(mapping_attr.dtype)
        target_counts = data.v18_target_edge_count.reshape(-1).to(mapping_attr.dtype)
        possible_1 = (n1 * (n1 - 1) / 2).clamp_min(1)
        possible_2 = (n2 * (n2 - 1) / 2).clamp_min(1)
        mapping = mapping_attr.reshape(-1)
        assignment_confidence = global_add_pool(
            mapping.square(), evidence["mapping_pair"], size=batch_size
        ).reshape(-1) / n1.clamp_min(1)
        summary = torch.stack(
            [
                source_counts / possible_1,
                target_counts / possible_2,
                n1 / n2.clamp_min(1),
                assignment_confidence,
            ],
            dim=1,
        )
        representation = torch.cat(
            [source_mean, source_max, target_mean, target_max, summary], dim=1
        )
        return self.graph_scorer(representation).squeeze(-1)


class OfficialMatchedEdgeDiscriminator(RelationAwareDiscriminator):
    """V11 discriminator plus a bounded official-graph edge residual."""

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim)
        rng_state = torch.random.get_rng_state()
        hidden_dim = int(getattr(args, "v18_edge_hidden_dim", 32))
        self.edge_reasoner = BatchedOfficialEdgeReasoner(hidden_dim)
        torch.random.set_rng_state(rng_state)
        self.edge_reasoning_gate = torch.nn.Parameter(
            torch.tensor(float(getattr(args, "v18_gate_init", 0.0)))
        )

    @property
    def effective_edge_gate(self):
        return torch.tanh(self.edge_reasoning_gate)

    def forward(self, data, mapping_attr):
        base_score = super().forward(data, mapping_attr)
        edge_score = self.edge_reasoner(data, mapping_attr)
        return base_score + self.effective_edge_gate * edge_score

