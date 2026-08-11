"""V15: project GINE input edges while preserving the V11 unit-GED evaluator."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_v11_trainer_class():
    module_path = V11_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v11_relation_aware_trainer_for_v15", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V11 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RelationAwareTrainer


V11RelationAwareTrainer = _load_v11_trainer_class()


def canonical_endpoint(source, target):
    source = int(source)
    target = int(target)
    return (source, target) if source <= target else (target, source)


def project_last_write_edges(edge_pairs, edge_ids, relation_features):
    """Reproduce undirected nx.Graph.add_edge last-write-wins semantics."""
    if not (len(edge_pairs) == len(edge_ids) == len(relation_features)):
        raise ValueError("Edges, relation IDs, and relation features must align.")

    last_index = {}
    multiplicity = Counter()
    for index, edge in enumerate(edge_pairs):
        if len(edge) != 2:
            raise ValueError(f"Expected a two-node edge, got {edge}.")
        endpoint = canonical_endpoint(edge[0], edge[1])
        last_index[endpoint] = index
        multiplicity[endpoint] += 1

    selected_indices = sorted(last_index.values())
    # The endpoint key is undirected, but retain the final raw orientation so
    # graphs without parallel endpoints remain byte-for-byte V11 inputs.
    projected_edges = [list(edge_pairs[index]) for index in selected_indices]
    parallel_multiplicities = [count for count in multiplicity.values() if count > 1]
    return {
        "graph": projected_edges,
        "edge_ids": [edge_ids[index] for index in selected_indices],
        "relation_features": [relation_features[index] for index in selected_indices],
        "selected_indices": selected_indices,
        "raw_edges": len(edge_pairs),
        "projected_edges": len(projected_edges),
        "dropped_edges": len(edge_pairs) - len(projected_edges),
        "parallel_endpoint_pairs": len(parallel_multiplicities),
        "max_multiplicity": max(parallel_multiplicities, default=1 if edge_pairs else 0),
    }


def _expanded_edge_index(edges, node_count):
    expanded = list(edges)
    expanded += [[target, source] for source, target in edges]
    expanded += [[node, node] for node in range(node_count)]
    if not expanded:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(expanded, dtype=torch.long).t().contiguous()


class V15ProjectedRelationTrainer(V11RelationAwareTrainer):
    version = "v15_benchmark_projected_relation_input"
    projection_revision = "undirected_last_raw_write_gine_input_only_v1"

    def __init__(self, args):
        self.v15_mode = getattr(args, "v15_mode", "raw")
        if self.v15_mode not in {"raw", "projected_input"}:
            raise ValueError("V15_MODE must be raw or projected_input.")
        super().__init__(args)

    def load_data(self):
        super().load_data()
        self.projected_graphs = []
        totals = Counter()
        max_multiplicity = 0

        for graph in self.graphs:
            projected = project_last_write_edges(
                graph["graph"],
                graph["edge_ids"],
                graph["relation_features"],
            )
            self.projected_graphs.append(projected)
            totals["raw_edges"] += projected["raw_edges"]
            totals["projected_edges"] += projected["projected_edges"]
            totals["dropped_edges"] += projected["dropped_edges"]
            totals["parallel_endpoint_pairs"] += projected["parallel_endpoint_pairs"]
            totals["affected_graphs"] += int(projected["dropped_edges"] > 0)
            max_multiplicity = max(max_multiplicity, projected["max_multiplicity"])

        graph_count = len(self.graphs)
        raw_edges = totals["raw_edges"]
        self.projection_diagnostics = {
            "revision": self.projection_revision,
            "mode": self.v15_mode,
            "graphs": graph_count,
            "affected_graphs": totals["affected_graphs"],
            "raw_edges": raw_edges,
            "projected_edges": totals["projected_edges"],
            "dropped_edges": totals["dropped_edges"],
            "dropped_fraction": round(
                totals["dropped_edges"] / raw_edges if raw_edges else 0.0,
                6,
            ),
            "parallel_endpoint_pairs": totals["parallel_endpoint_pairs"],
            "max_multiplicity": max_multiplicity,
            "model_edge_view": (
                "v11_raw_multirelation"
                if self.v15_mode == "raw"
                else "undirected_simple_last_write"
            ),
            "unit_ged_edge_view": "unchanged_v11_raw_legacy",
        }
        print(
            "V15 projection diagnostics:",
            json.dumps(self.projection_diagnostics, sort_keys=True),
        )

    def transfer_data_to_torch(self):
        super().transfer_data_to_torch()
        self.projected_edge_index = []
        self.projected_edge_ids = []
        self.projected_relation_edge_attr = []

        for graph, projected in zip(self.graphs, self.projected_graphs):
            edge_index = _expanded_edge_index(projected["graph"], graph["n"])
            relation = torch.tensor(
                projected["relation_features"], dtype=torch.float
            ).reshape(-1, self.relation_dim)
            self_loops = torch.zeros((graph["n"], self.relation_dim), dtype=torch.float)
            edge_attr = torch.cat([relation, relation, self_loops], dim=0)
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise RuntimeError("Projected relation attributes do not align with edges.")
            self.projected_edge_index.append(edge_index)
            self.projected_edge_ids.append(projected["edge_ids"])
            self.projected_relation_edge_attr.append(edge_attr)

    @staticmethod
    def _pair_edge_labels(edge_ids_1, edge_ids_2, n1, n2):
        edge_vocab = {}
        next_label = 1
        for edge_id in edge_ids_1 + edge_ids_2:
            if edge_id not in edge_vocab:
                edge_vocab[edge_id] = next_label
                next_label += 1
        labels_1 = [edge_vocab[edge_id] for edge_id in edge_ids_1]
        labels_2 = [edge_vocab[edge_id] for edge_id in edge_ids_2]
        tensor_1 = torch.tensor(labels_1 + labels_1 + [0] * n1, dtype=torch.long)
        tensor_2 = torch.tensor(labels_2 + labels_2 + [0] * n2, dtype=torch.long)
        return torch.cat([tensor_1, tensor_2], dim=0)

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)

        # Candidate GED and BPR labels keep the exact V11 edge view in both modes.
        data.unit_cost_edge_index = data.edge_index.clone()
        data.unit_cost_edge_labels = data.edge_labels.clone()
        graph_1, graph_2 = data.i_j[0].tolist()
        data.v15_raw_edge_counts = data.m.clone()
        data.v15_projected_edge_counts = torch.tensor(
            [[
                self.projected_graphs[graph_1]["projected_edges"],
                self.projected_graphs[graph_2]["projected_edges"],
            ]],
            dtype=torch.long,
        )

        if self.v15_mode == "raw":
            return data

        n1 = int(data.n[0, 0].item())
        n2 = int(data.n[0, 1].item())
        data.edge_index = torch.cat(
            [
                self.projected_edge_index[graph_1],
                self.projected_edge_index[graph_2] + n1,
            ],
            dim=1,
        )
        data.edge_attr = torch.cat(
            [
                self.projected_relation_edge_attr[graph_1],
                self.projected_relation_edge_attr[graph_2],
            ],
            dim=0,
        )
        data.edge_labels = self._pair_edge_labels(
            self.projected_edge_ids[graph_1],
            self.projected_edge_ids[graph_2],
            n1,
            n2,
        )
        if not (
            data.edge_index.shape[1]
            == data.edge_attr.shape[0]
            == data.edge_labels.shape[0]
        ):
            raise RuntimeError("Projected pair edges, labels, and attributes are misaligned.")
        return data

    def _pair_labeled_adjacency(self, batch, batch_idx):
        n1 = int(batch.n[batch_idx, 0].item())
        n2 = int(batch.n[batch_idx, 1].item())
        node_offset = (
            int(torch.sum(batch.n[:batch_idx], dim=1).sum().item())
            if batch_idx > 0
            else 0
        )
        cost_edges = batch.unit_cost_edge_index
        cost_labels = batch.unit_cost_edge_labels
        edge_batch = batch.batch[cost_edges[0]]
        pair_mask = edge_batch == batch_idx
        pair_edges = cost_edges[:, pair_mask]
        pair_labels = cost_labels[pair_mask]
        pair_indicator = batch.x_indicator[pair_edges[0]].squeeze(1)

        adj_1 = torch.zeros((n1, n1), dtype=torch.long, device=cost_edges.device)
        adj_2 = torch.zeros((n2, n2), dtype=torch.long, device=cost_edges.device)
        edge_mask_1 = (pair_indicator == 0) & (pair_labels > 0)
        if edge_mask_1.any():
            local_edges_1 = pair_edges[:, edge_mask_1] - node_offset
            adj_1[local_edges_1[0], local_edges_1[1]] = pair_labels[edge_mask_1]
        edge_mask_2 = (pair_indicator == 1) & (pair_labels > 0)
        if edge_mask_2.any():
            local_edges_2 = pair_edges[:, edge_mask_2] - (node_offset + n1)
            adj_2[local_edges_2[0], local_edges_2[1]] = pair_labels[edge_mask_2]
        return adj_1, adj_2

    def _compute_single_ged_from_dense_solution(self, solution, data):
        n1 = int(data.n[0, 0].item())
        n2 = int(data.n[0, 1].item())
        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        unmatched_cols = [column for column in range(n2) if column not in mapped_cols]
        permutation = torch.tensor(
            mapped_cols + unmatched_cols, dtype=torch.long, device=solution.device
        )

        cost_edges = data.unit_cost_edge_index
        cost_labels = data.unit_cost_edge_labels
        indicator = data.x_indicator[cost_edges[0]].squeeze(1)
        edge_mask_1 = (indicator == 0) & (cost_labels > 0)
        edge_mask_2 = (indicator == 1) & (cost_labels > 0)
        edge_index_1 = cost_edges[:, edge_mask_1]
        edge_index_2 = cost_edges[:, edge_mask_2] - n1
        edge_label_1 = cost_labels[edge_mask_1]
        edge_label_2 = cost_labels[edge_mask_2]

        adj_1 = torch.zeros((n1, n1), dtype=torch.long, device=solution.device)
        adj_2 = torch.zeros((n2, n2), dtype=torch.long, device=solution.device)
        if edge_index_1.numel() > 0:
            adj_1[edge_index_1[0], edge_index_1[1]] = edge_label_1
        if edge_index_2.numel() > 0:
            adj_2[edge_index_2[0], edge_index_2[1]] = edge_label_2

        if self.args.cost_mode == "containment":
            mapped_adj_2 = adj_2.index_select(0, permutation[:n1]).index_select(
                1, permutation[:n1]
            )
            overlap_mask = torch.triu(
                torch.ones((n1, n1), dtype=torch.bool, device=solution.device),
                diagonal=1,
            )
            overlap_edge_cost = torch.count_nonzero(
                adj_1[overlap_mask] != mapped_adj_2[overlap_mask]
            ).item()
            edge_cost = abs(len(edge_label_2) // 2 - len(edge_label_1) // 2)
            return float((n2 - n1) + edge_cost + overlap_edge_cost)

        padded_adj_1 = torch.zeros((n2, n2), dtype=torch.long, device=solution.device)
        padded_adj_1[:n1, :n1] = adj_1
        permuted_adj_2 = adj_2.index_select(0, permutation).index_select(1, permutation)
        upper_mask = torch.triu(
            torch.ones((n2, n2), dtype=torch.bool, device=solution.device), diagonal=1
        )
        edge_cost = torch.count_nonzero(
            padded_adj_1[upper_mask] != permuted_adj_2[upper_mask]
        ).item()
        return float((n2 - n1) + edge_cost)

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update(
            {
                "version": self.version,
                "v15_mode": self.v15_mode,
                "projection_revision": self.projection_revision,
                "projection_diagnostics": self.projection_diagnostics,
                "preserves_cost_mode": "unit",
                "unit_ged_edge_view": "unchanged_v11_raw_legacy",
                "primary_metrics": ["mae", "acc"],
            }
        )
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("V15 metadata:", json.dumps(self.projection_diagnostics, sort_keys=True))
        return payload
