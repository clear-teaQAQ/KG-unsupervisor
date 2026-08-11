"""V11 discriminator plus matching-conditioned exact-edge reasoning."""

from pathlib import Path
import sys

import torch


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
        self.edge_reasoner = MatchingConditionedEdgeReasoner(hidden_dim)
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
