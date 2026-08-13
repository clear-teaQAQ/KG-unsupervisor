"""V17: cross-graph relation-aware matching scores.

The existing unit-GED evaluator and BPR training loop consume the returned
pair logits.  This module changes how those logits are produced: node states
are encoded per graph and then compared across graphs with learned attention.
"""

from pathlib import Path
import sys

import torch
from torch_geometric.nn import GINEConv, GraphNorm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CrossGraphSinkhornMatcher(torch.nn.Module):
    """Generate dense source-target assignment logits for each graph pair."""

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__()
        hidden_dims = list(getattr(args, "hidden_dim", [128, 64, 32]))
        self.hidden_dims = hidden_dims
        self.relation_dim = int(relation_dim)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer, hidden_dim in enumerate(hidden_dims):
            input_dim = number_of_labels if layer == 0 else hidden_dims[layer - 1]
            network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GINEConv(network, train_eps=True, edge_dim=self.relation_dim)
            )
            self.norms.append(GraphNorm(hidden_dim))

        embed_dim = hidden_dims[-1]
        attention_dim = max(embed_dim // 2, 16)
        self.query = torch.nn.Linear(embed_dim, attention_dim, bias=False)
        self.key = torch.nn.Linear(embed_dim, attention_dim, bias=False)
        self.pair_scorer = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, 1),
        )
        self.scale = attention_dim ** -0.5

    def _encode(self, data):
        features = data.x
        for conv, norm in zip(self.convs, self.norms):
            features = torch.relu(norm(conv(features, data.edge_index, data.edge_attr), data.batch))
        return features

    def forward(self, data, noise_mapping_attr=None, timestep=None):
        del noise_mapping_attr, timestep
        features = self._encode(data)
        pair_logits = []
        graph_batch = data.batch
        indicators = data.x_indicator.squeeze(-1).bool()
        pair_count = int(data.n.shape[0])
        for pair_index in range(pair_count):
            nodes = graph_batch == pair_index
            source = features[nodes & ~indicators]
            target = features[nodes & indicators]
            q = self.query(source)
            k = self.key(target)
            attention = (q @ k.transpose(0, 1)) * self.scale
            source_expanded = source[:, None, :].expand(-1, target.shape[0], -1)
            target_expanded = target[None, :, :].expand(source.shape[0], -1, -1)
            pair_features = torch.cat(
                [
                    source_expanded,
                    target_expanded,
                    torch.abs(source_expanded - target_expanded),
                    source_expanded * target_expanded,
                    attention.unsqueeze(-1),
                ],
                dim=-1,
            )
            pair_logits.append(self.pair_scorer(pair_features).reshape(-1, 1))
        if not pair_logits:
            return features.new_empty((0, 1))
        return torch.cat(pair_logits, dim=0)
