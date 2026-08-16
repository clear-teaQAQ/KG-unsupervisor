"""V11 discriminator plus matching-conditioned exact-edge reasoning."""

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


def _last_write_upper_edges(data, offset, node_count):
    """Reproduce V14's dense-adjacency last-write rule once on the CPU."""
    last_labels = {}
    for endpoints, label in zip(data.edge_index.t().tolist(), data.edge_labels.tolist()):
        if int(label) <= 0:
            continue
        source = int(endpoints[0]) - offset
        target = int(endpoints[1]) - offset
        if 0 <= source < node_count and 0 <= target < node_count:
            last_labels[(source, target)] = int(label)

    upper = sorted(
        (source, target, label)
        for (source, target), label in last_labels.items()
        if source < target
    )
    if not upper:
        return _empty_long(0, 2), _empty_long()
    edges = torch.tensor([(source, target) for source, target, _ in upper])
    labels = torch.tensor([label for _, _, label in upper])
    return edges.long(), labels.long()


def attach_v14_edge_reasoning_cache(data):
    """Cache V14's exact raw-graph edge correspondences for one graph pair."""
    n1 = int(data.n[0, 0].item())
    n2 = int(data.n[0, 1].item())
    valid = (
        data.edge_index.ndim == 2
        and data.edge_index.shape[0] == 2
        and data.edge_labels.numel() == data.edge_index.shape[1]
    )
    if valid:
        source_edges, source_labels = _last_write_upper_edges(data, 0, n1)
        target_edges, target_labels = _last_write_upper_edges(data, n1, n2)
    else:
        source_edges, source_labels = _empty_long(0, 2), _empty_long()
        target_edges, target_labels = _empty_long(0, 2), _empty_long()

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

    # These are pair-local positions. Avoid "index" in attribute names so
    # torch-geometric does not offset them while forming a mini-batch.
    data.v14_mapping_count = torch.tensor([n1 * n2], dtype=torch.long)
    data.v14_source_edge_count = torch.tensor([source_count], dtype=torch.long)
    data.v14_target_edge_count = torch.tensor([target_count], dtype=torch.long)
    data.v14_combo_count = torch.tensor([combo_count], dtype=torch.long)
    data.v14_source_endpoints = source_edges
    data.v14_target_endpoints = target_edges
    data.v14_source_labels = source_labels
    data.v14_target_labels = target_labels
    data.v14_combo_mapping_positions = mapping_positions
    data.v14_combo_source_edge = source_ids
    data.v14_combo_target_edge = target_ids
    data.v14_combo_exact = exact
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


class BatchedMatchingConditionedEdgeReasoner(torch.nn.Module):
    """V14 edge reasoner with cached correspondence and batched pooling."""

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
        mapping_counts = data.v14_mapping_count.reshape(-1).long()
        source_counts = data.v14_source_edge_count.reshape(-1).long()
        target_counts = data.v14_target_edge_count.reshape(-1).long()
        combo_counts = data.v14_combo_count.reshape(-1).long()
        source_topology = mapping.new_zeros(int(source_counts.sum().item()))
        source_exact = mapping.new_zeros(source_topology.shape)
        target_topology = mapping.new_zeros(int(target_counts.sum().item()))
        target_exact = mapping.new_zeros(target_topology.shape)

        if data.v14_combo_mapping_positions.numel():
            combo_pair = _pair_ids(combo_counts)
            positions = data.v14_combo_mapping_positions.long()
            positions = positions + _offsets(mapping_counts)[combo_pair, None]
            support = (
                mapping[positions[:, 0]] * mapping[positions[:, 1]]
                + mapping[positions[:, 2]] * mapping[positions[:, 3]]
            )
            exact_support = support * data.v14_combo_exact.to(mapping.dtype)
            source_ids = data.v14_combo_source_edge.long() + _offsets(source_counts)[
                combo_pair
            ]
            target_ids = data.v14_combo_target_edge.long() + _offsets(target_counts)[
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
            "batch_size": int(data.n.shape[0]),
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
        source_counts = data.v14_source_edge_count.reshape(-1).to(mapping_attr.dtype)
        target_counts = data.v14_target_edge_count.reshape(-1).to(mapping_attr.dtype)
        possible_1 = (n1 * (n1 - 1) / 2).clamp_min(1)
        possible_2 = (n2 * (n2 - 1) / 2).clamp_min(1)
        confidence = global_add_pool(
            mapping_attr.reshape(-1).square(),
            evidence["mapping_pair"],
            size=batch_size,
        ).reshape(-1) / n1.clamp_min(1)
        summary = torch.stack(
            [
                source_counts / possible_1,
                target_counts / possible_2,
                n1 / n2.clamp_min(1),
                confidence,
            ],
            dim=1,
        )
        representation = torch.cat(
            [source_mean, source_max, target_mean, target_max, summary], dim=1
        )
        return self.graph_scorer(representation).squeeze(-1)


class CachedPairwiseMatchingConditionedEdgeReasoner(torch.nn.Module):
    """Cache graph topology while preserving V14's pairwise operation order."""

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
    def _pool(events, encoder, device, dtype, hidden_dim):
        if events.numel() == 0:
            zero = torch.zeros(hidden_dim, device=device, dtype=dtype)
            return zero, zero
        encoded = encoder(events)
        return encoded.mean(dim=0), encoded.amax(dim=0)

    @staticmethod
    def _events(exact, topology):
        wrong_relation = (topology - exact).clamp_min(0)
        broken_or_extra = (1.0 - topology).clamp_min(0)
        return torch.stack([exact, wrong_relation, broken_or_extra], dim=-1)

    def forward(self, data, mapping_attr):
        mapping = mapping_attr.reshape(-1)
        mapping_batch = data.batch[data.edge_index_mapping[0]]
        pair_sizes = [
            (int(n1), int(n2)) for n1, n2 in data.n.detach().cpu().tolist()
        ]
        source_counts = data.v14_source_edge_count.reshape(-1).tolist()
        target_counts = data.v14_target_edge_count.reshape(-1).tolist()
        combo_counts = data.v14_combo_count.reshape(-1).tolist()
        pair_representations = []
        source_offset = 0
        target_offset = 0

        for pair_index, (n1, n2) in enumerate(pair_sizes):
            source_count = int(source_counts[pair_index])
            target_count = int(target_counts[pair_index])
            combo_count = int(combo_counts[pair_index])
            # Keep the original V14 gather operation so backward accumulation is
            # identical; only the graph's immutable labeled edges are cached.
            pair_mapping = mapping[mapping_batch == pair_index].reshape(n1, n2)

            if combo_count:
                source_edges = data.v14_source_endpoints[
                    source_offset : source_offset + source_count
                ].long()
                target_edges = data.v14_target_endpoints[
                    target_offset : target_offset + target_count
                ].long()
                source_labels = data.v14_source_labels[
                    source_offset : source_offset + source_count
                ].long()
                target_labels = data.v14_target_labels[
                    target_offset : target_offset + target_count
                ].long()
                left = (
                    pair_mapping[source_edges[:, 0, None], target_edges[None, :, 0]]
                    * pair_mapping[source_edges[:, 1, None], target_edges[None, :, 1]]
                )
                crossed = (
                    pair_mapping[source_edges[:, 0, None], target_edges[None, :, 1]]
                    * pair_mapping[source_edges[:, 1, None], target_edges[None, :, 0]]
                )
                joint_support = left + crossed
                exact_mask = source_labels[:, None] == target_labels[None, :]
                exact_support = joint_support * exact_mask.to(joint_support.dtype)
                source_topology = joint_support.sum(dim=1)
                source_exact = exact_support.sum(dim=1)
                target_topology = joint_support.sum(dim=0)
                target_exact = exact_support.sum(dim=0)
            else:
                source_topology = mapping.new_zeros(source_count)
                source_exact = mapping.new_zeros(source_count)
                target_topology = mapping.new_zeros(target_count)
                target_exact = mapping.new_zeros(target_count)

            source_events = self._events(source_exact, source_topology)
            target_events = self._events(target_exact, target_topology)
            source_mean, source_max = self._pool(
                source_events,
                self.event_encoder,
                pair_mapping.device,
                pair_mapping.dtype,
                self.hidden_dim,
            )
            target_mean, target_max = self._pool(
                target_events,
                self.event_encoder,
                pair_mapping.device,
                pair_mapping.dtype,
                self.hidden_dim,
            )
            possible_1 = max(n1 * (n1 - 1) / 2, 1)
            possible_2 = max(n2 * (n2 - 1) / 2, 1)
            density = pair_mapping.new_tensor(
                [
                    source_count / possible_1,
                    target_count / possible_2,
                    n1 / max(n2, 1),
                ]
            )
            confidence = pair_mapping.square().sum() / max(n1, 1)
            pair_representations.append(
                torch.cat(
                    [
                        source_mean,
                        source_max,
                        target_mean,
                        target_max,
                        density,
                        confidence.reshape(1),
                    ]
                )
            )
            source_offset += source_count
            target_offset += target_count

        if not pair_representations:
            return mapping_attr.new_zeros(0)
        return self.graph_scorer(torch.stack(pair_representations)).squeeze(-1)


class MatchingConditionedEdgeReasoner(torch.nn.Module):
    """Encode edge-edit evidence induced by a candidate matching.

    For every real edge on both sides, the module measures the soft probability
    that the candidate matching maps it to an exact-label edge, a different-
    label edge, or no edge. These are model inputs, not replacement GED labels.
    """

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
    def _pool(events, encoder, device, dtype, hidden_dim):
        if events.numel() == 0:
            zero = torch.zeros(hidden_dim, device=device, dtype=dtype)
            return zero, zero
        encoded = encoder(events)
        return encoded.mean(dim=0), encoded.amax(dim=0)

    @staticmethod
    def _labeled_adjacencies(data, pair_index, n1, n2, node_offset):
        edge_batch = data.batch[data.edge_index[0]]
        pair_mask = edge_batch == pair_index
        pair_edges = data.edge_index[:, pair_mask]
        pair_labels = data.edge_labels[pair_mask]
        pair_indicator = data.x_indicator[pair_edges[0]].squeeze(1)

        adjacency_1 = torch.zeros(
            (n1, n1), dtype=torch.long, device=data.edge_index.device
        )
        adjacency_2 = torch.zeros(
            (n2, n2), dtype=torch.long, device=data.edge_index.device
        )

        mask_1 = (pair_indicator == 0) & (pair_labels > 0)
        edges_1 = pair_edges[:, mask_1] - node_offset
        adjacency_1[edges_1[0], edges_1[1]] = pair_labels[mask_1]

        mask_2 = (pair_indicator == 1) & (pair_labels > 0)
        edges_2 = pair_edges[:, mask_2] - (node_offset + n1)
        adjacency_2[edges_2[0], edges_2[1]] = pair_labels[mask_2]

        return adjacency_1, adjacency_2

    @staticmethod
    def _edge_list(adjacency):
        upper = torch.triu(adjacency, diagonal=1)
        endpoints = torch.nonzero(upper > 0, as_tuple=False)
        if endpoints.numel() == 0:
            return endpoints, torch.empty(
                0, dtype=torch.long, device=adjacency.device
            )
        return endpoints, upper[endpoints[:, 0], endpoints[:, 1]]

    def pair_event_features(
        self, data, mapping, pair_index, node_offset, pair_sizes=None
    ):
        """Return differentiable per-edge evidence for one graph pair."""
        if pair_sizes is None:
            n1, n2 = (int(value) for value in data.n[pair_index].detach().cpu())
        else:
            n1, n2 = pair_sizes
        adjacency_1, adjacency_2 = self._labeled_adjacencies(
            data, pair_index, n1, n2, node_offset
        )
        edges_1, labels_1 = self._edge_list(adjacency_1)
        edges_2, labels_2 = self._edge_list(adjacency_2)

        if edges_1.numel() and edges_2.numel():
            left = (
                mapping[edges_1[:, 0, None], edges_2[None, :, 0]]
                * mapping[edges_1[:, 1, None], edges_2[None, :, 1]]
            )
            crossed = (
                mapping[edges_1[:, 0, None], edges_2[None, :, 1]]
                * mapping[edges_1[:, 1, None], edges_2[None, :, 0]]
            )
            joint_support = left + crossed
            exact_mask = labels_1[:, None] == labels_2[None, :]
            exact_support = joint_support * exact_mask.to(joint_support.dtype)
            source_topology = joint_support.sum(dim=1)
            source_exact = exact_support.sum(dim=1)
            target_topology = joint_support.sum(dim=0)
            target_exact = exact_support.sum(dim=0)
        else:
            source_topology = mapping.new_zeros(labels_1.shape[0])
            source_exact = mapping.new_zeros(labels_1.shape[0])
            target_topology = mapping.new_zeros(labels_2.shape[0])
            target_exact = mapping.new_zeros(labels_2.shape[0])

        def events(exact, topology):
            wrong_relation = (topology - exact).clamp_min(0)
            broken_or_extra = (1.0 - topology).clamp_min(0)
            return torch.stack([exact, wrong_relation, broken_or_extra], dim=-1)

        source_events = events(source_exact, source_topology)
        target_events = events(target_exact, target_topology)
        return {
            "source_events": source_events,
            "target_events": target_events,
            "source_edge_count": int(labels_1.numel()),
            "target_edge_count": int(labels_2.numel()),
            "exact_overlap": source_exact.sum(),
            "topology_overlap": source_topology.sum(),
        }

    def forward(self, data, mapping_attr):
        mapping_values = mapping_attr.reshape(-1)
        mapping_batch = data.batch[data.edge_index_mapping[0]]
        pair_representations = []
        node_offset = 0
        pair_sizes = [
            (int(n1), int(n2)) for n1, n2 in data.n.detach().cpu().tolist()
        ]

        for pair_index, (n1, n2) in enumerate(pair_sizes):
            pair_mapping = mapping_values[mapping_batch == pair_index].reshape(n1, n2)
            evidence = self.pair_event_features(
                data,
                pair_mapping,
                pair_index,
                node_offset,
                pair_sizes=(n1, n2),
            )
            source_mean, source_max = self._pool(
                evidence["source_events"],
                self.event_encoder,
                pair_mapping.device,
                pair_mapping.dtype,
                self.hidden_dim,
            )
            target_mean, target_max = self._pool(
                evidence["target_events"],
                self.event_encoder,
                pair_mapping.device,
                pair_mapping.dtype,
                self.hidden_dim,
            )

            possible_1 = max(n1 * (n1 - 1) / 2, 1)
            possible_2 = max(n2 * (n2 - 1) / 2, 1)
            density = pair_mapping.new_tensor(
                [
                    evidence["source_edge_count"] / possible_1,
                    evidence["target_edge_count"] / possible_2,
                    n1 / max(n2, 1),
                ]
            )
            assignment_confidence = pair_mapping.square().sum() / max(n1, 1)
            density_and_assignment = torch.cat(
                [density, assignment_confidence.reshape(1)]
            )
            pair_representations.append(
                torch.cat(
                    [
                        source_mean,
                        source_max,
                        target_mean,
                        target_max,
                        density_and_assignment,
                    ]
                )
            )
            node_offset += n1 + n2

        if not pair_representations:
            return mapping_attr.new_zeros(0)
        return self.graph_scorer(torch.stack(pair_representations)).squeeze(-1)


class MatchingConditionedEdgeDiscriminator(RelationAwareDiscriminator):
    """V11 discriminator with a bounded, initially disabled edge residual."""

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim)

        # Preserve the V11 random stream so baseline and full modes start from
        # identical generator/discriminator/data initializations.
        rng_state = torch.random.get_rng_state()
        hidden_dim = int(getattr(args, "v14_edge_hidden_dim", 32))
        if not bool(getattr(args, "v14_edge_cache", False)):
            reasoner_cls = MatchingConditionedEdgeReasoner
        elif bool(getattr(args, "v14_vectorized_edge", False)):
            reasoner_cls = BatchedMatchingConditionedEdgeReasoner
        else:
            reasoner_cls = CachedPairwiseMatchingConditionedEdgeReasoner
        self.edge_reasoner = reasoner_cls(hidden_dim)
        torch.random.set_rng_state(rng_state)

        gate_init = float(getattr(args, "v14_gate_init", 0.0))
        self.edge_reasoning_gate = torch.nn.Parameter(
            torch.tensor(gate_init, dtype=torch.float)
        )

    @property
    def effective_edge_gate(self):
        return torch.tanh(self.edge_reasoning_gate)

    def forward(self, data, mapping_attr):
        v11_score = super().forward(data, mapping_attr)
        residual = self.edge_reasoner(data, mapping_attr)
        return v11_score + self.effective_edge_gate * residual
