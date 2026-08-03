"""Relation-aware variants of the GEDRanker generator and discriminator."""

from pathlib import Path
import sys

import torch
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import to_undirected


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.GEDRanker.layers import (
    AGNN,
    AGNN_D,
    ScalarEmbeddingSine,
    timestep_embedding,
)


class RelationAwareDiscriminator(torch.nn.Module):
    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.relation_dim = relation_dim
        self.setup_layers()

    def setup_layers(self):
        self.hidden_dims = self.args.d_hidden_dim
        self.num_layers = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            input_dim = self.number_labels if layer == 0 else self.hidden_dims[layer - 1]
            network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.hidden_dims[layer]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dims[layer], self.hidden_dims[layer]),
            )
            noise_dim = self.hidden_dims[layer] if layer == 0 else self.hidden_dims[layer - 1]
            self.conv_layers.append(
                GINEConv(network, train_eps=True, edge_dim=self.relation_dim)
            )
            self.agnn_layers.append(AGNN_D(self.hidden_dims[layer], noise_dim))
            self.gns.append(GraphNorm(self.hidden_dims[layer]))

        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0], normalize=False)
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.cost_matrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def convolutional_pass(
        self,
        features,
        graph_edge_index,
        graph_edge_attr,
        edge_mapping_idx,
        noise_mapping_emb,
        batch,
        graph_2,
    ):
        graph_batch = batch * 2
        graph_batch[graph_2] += 1
        for layer in range(self.num_layers):
            features = torch.relu(
                self.gns[layer](
                    self.conv_layers[layer](
                        features,
                        graph_edge_index,
                        graph_edge_attr,
                    ),
                    batch=graph_batch,
                )
            )
            features, noise_mapping_emb = self.agnn_layers[layer](
                features,
                edge_mapping_idx,
                noise_mapping_emb,
                batch,
            )
        return features, noise_mapping_emb

    def forward(self, data, noise_mapping_attr):
        edge_mapping_idx, noise_mapping_attr = to_undirected(
            data.edge_index_mapping,
            noise_mapping_attr,
        )
        graph_2 = (data.x_indicator == 1).squeeze(1)
        noise_mapping_emb = self.edge_embed(
            self.edge_pos_embed(noise_mapping_attr)
        )
        _, noise_mapping_emb = self.convolutional_pass(
            data.x,
            data.edge_index,
            data.edge_attr,
            edge_mapping_idx,
            noise_mapping_emb,
            data.batch,
            graph_2,
        )
        cost_matrix = self.cost_matrix(noise_mapping_emb)
        _, cost_matrix = to_undirected(edge_mapping_idx, cost_matrix)
        source_mask = (data.x_indicator[edge_mapping_idx[0]] == 0).squeeze(1)
        cost_matrix = cost_matrix[source_mask]
        return global_add_pool(
            cost_matrix,
            data.batch[data.edge_index_mapping[0]],
        ).squeeze(-1)


class RelationAwareDiffMatch(torch.nn.Module):
    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.relation_dim = relation_dim
        self.setup_layers()

    def setup_layers(self):
        self.hidden_dims = self.args.hidden_dim
        self.num_layers = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            input_dim = self.number_labels if layer == 0 else self.hidden_dims[layer - 1]
            network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.hidden_dims[layer]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dims[layer], self.hidden_dims[layer]),
            )
            noise_dim = self.hidden_dims[layer] if layer == 0 else self.hidden_dims[layer - 1]
            self.conv_layers.append(
                GINEConv(network, train_eps=True, edge_dim=self.relation_dim)
            )
            self.agnn_layers.append(
                AGNN(
                    self.hidden_dims[layer],
                    self.hidden_dims[0] // 2,
                    noise_dim,
                )
            )
            self.gns.append(GraphNorm(self.hidden_dims[layer]))

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[0] // 2, self.hidden_dims[0] // 2),
        )
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0], normalize=False)
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.map_matrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def convolutional_pass(
        self,
        features,
        graph_edge_index,
        graph_edge_attr,
        edge_mapping_idx,
        noise_mapping_emb,
        time_emb,
        batch,
        graph_2,
    ):
        graph_batch = batch * 2
        graph_batch[graph_2] += 1
        for layer in range(self.num_layers):
            features = torch.relu(
                self.gns[layer](
                    self.conv_layers[layer](
                        features,
                        graph_edge_index,
                        graph_edge_attr,
                    ),
                    batch=graph_batch,
                )
            )
            features, noise_mapping_emb = self.agnn_layers[layer](
                features,
                edge_mapping_idx,
                noise_mapping_emb,
                time_emb,
                batch,
            )
        return features, noise_mapping_emb

    def forward(self, data, noise_mapping_attr, timestep):
        edge_mapping_idx, noise_mapping_attr = to_undirected(
            data.edge_index_mapping,
            noise_mapping_attr,
        )
        graph_2 = (data.x_indicator == 1).squeeze(1)
        time_emb = self.time_embed(timestep_embedding(timestep, self.hidden_dims[0]))
        noise_mapping_emb = self.edge_embed(
            self.edge_pos_embed(noise_mapping_attr)
        )
        _, noise_mapping_emb = self.convolutional_pass(
            data.x,
            data.edge_index,
            data.edge_attr,
            edge_mapping_idx,
            noise_mapping_emb,
            time_emb,
            data.batch,
            graph_2,
        )
        map_matrix = self.map_matrix(noise_mapping_emb)
        _, map_matrix = to_undirected(edge_mapping_idx, map_matrix)
        source_mask = (data.x_indicator[edge_mapping_idx[0]] == 0).squeeze(1)
        return map_matrix[source_mask]
