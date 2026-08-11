"""Relation-aware GEDRanker modules owned by the isolated V12 experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import to_undirected

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
GEDRANKER_DIR = PROJECT_ROOT / "src" / "GEDRanker"
if str(GEDRANKER_DIR) not in sys.path:
    sys.path.insert(0, str(GEDRANKER_DIR))

from layers import AGNN, AGNN_D, ScalarEmbeddingSine, timestep_embedding


class _RelationNetwork(torch.nn.Module):
    def __init__(self, args, number_of_labels, relation_dim, discriminator=False):
        super().__init__()
        hidden = args.d_hidden_dim if discriminator else args.hidden_dim
        self.hidden = hidden
        self.relation_dim = relation_dim
        self.convs = torch.nn.ModuleList()
        self.agnn = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer, output_dim in enumerate(hidden):
            input_dim = number_of_labels if layer == 0 else hidden[layer - 1]
            net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim),
            )
            previous_dim = output_dim if layer == 0 else hidden[layer - 1]
            self.convs.append(GINEConv(net, train_eps=True, edge_dim=relation_dim))
            self.agnn.append(
                AGNN_D(output_dim, previous_dim)
                if discriminator
                else AGNN(output_dim, hidden[0] // 2, previous_dim)
            )
            self.norms.append(GraphNorm(output_dim))
        self.edge_pos_embed = ScalarEmbeddingSine(hidden[0], normalize=False)
        self.edge_embed = torch.nn.Linear(hidden[0], hidden[0])

    def _message_pass(self, x, edge_index, edge_attr, mapping_index, mapping_emb, batch, graph_2, time_emb=None):
        graph_batch = batch * 2
        graph_batch[graph_2] += 1
        for layer, conv in enumerate(self.convs):
            x = torch.relu(self.norms[layer](conv(x, edge_index, edge_attr), batch=graph_batch))
            if time_emb is None:
                x, mapping_emb = self.agnn[layer](x, mapping_index, mapping_emb, batch)
            else:
                x, mapping_emb = self.agnn[layer](x, mapping_index, mapping_emb, time_emb, batch)
        return x, mapping_emb

    def _mapping_embedding(self, mapping_index, mapping_attr):
        mapping_index, mapping_attr = to_undirected(mapping_index, mapping_attr)
        return mapping_index, self.edge_embed(self.edge_pos_embed(mapping_attr))


class RelationAwareDiffMatch(_RelationNetwork):
    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim, discriminator=False)
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden[0], self.hidden[0] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden[0] // 2, self.hidden[0] // 2),
        )
        self.map_matrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden[-1], self.hidden[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden[-1] * 2, self.hidden[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden[-1], 1),
        )

    def forward(self, data, mapping_attr, timestep):
        mapping_index, mapping_emb = self._mapping_embedding(data.edge_index_mapping, mapping_attr)
        graph_2 = (data.x_indicator == 1).squeeze(1)
        time_emb = self.time_embed(timestep_embedding(timestep, self.hidden[0]))
        _, mapping_emb = self._message_pass(
            data.x, data.edge_index, data.edge_attr, mapping_index, mapping_emb,
            data.batch, graph_2, time_emb,
        )
        result = self.map_matrix(mapping_emb)
        _, result = to_undirected(mapping_index, result)
        source_mask = (data.x_indicator[mapping_index[0]] == 0).squeeze(1)
        return result[source_mask]


class RelationAwareDiscriminator(_RelationNetwork):
    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim, discriminator=True)
        last = self.hidden[-1]
        self.cost_matrix = torch.nn.Sequential(
            torch.nn.Linear(last, last * 2), torch.nn.ReLU(),
            torch.nn.Linear(last * 2, last), torch.nn.ReLU(),
            torch.nn.Linear(last, 1),
        )

    def forward(self, data, mapping_attr):
        mapping_index, mapping_emb = self._mapping_embedding(data.edge_index_mapping, mapping_attr)
        graph_2 = (data.x_indicator == 1).squeeze(1)
        _, mapping_emb = self._message_pass(
            data.x, data.edge_index, data.edge_attr, mapping_index, mapping_emb,
            data.batch, graph_2,
        )
        result = self.cost_matrix(mapping_emb)
        _, result = to_undirected(mapping_index, result)
        source_mask = (data.x_indicator[mapping_index[0]] == 0).squeeze(1)
        return global_add_pool(result[source_mask], data.batch[data.edge_index_mapping[0]]).squeeze(-1)
