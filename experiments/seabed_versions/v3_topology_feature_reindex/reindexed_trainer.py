"""V3 audit: align node features and entity IDs with topology indices."""

import json
from pathlib import Path
import sys
import time


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
for path in (PROJECT_ROOT, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audited_trainer import EditPathAuditTrainer
from src.SEABED.utils import get_file_paths
from topology_reindex import derive_topology_feature_order, reorder_features


class TopologyFeatureReindexTrainer(EditPathAuditTrainer):
    version = "v3_topology_feature_reindex"
    feature_revision = "kg_edge_topology_reindex_v1"

    def load_data(self):
        start = time.time()
        super().load_data()

        graph_paths = []
        for split in ("train", "val", "test"):
            graph_paths.extend(
                get_file_paths(str(Path(self.args.dataset_root) / split), "json")
            )
        if len(graph_paths) != len(self.graphs):
            raise RuntimeError(
                f"Graph path index has {len(graph_paths)} entries, "
                f"loader has {len(self.graphs)} graphs."
            )

        corrected_node_ids = []
        changed_graphs = 0
        reassigned_nodes = 0
        consistent_edges_before = 0
        consistent_edges_after = 0
        edge_count = 0
        fully_consistent_graphs_before = 0
        fully_consistent_graphs_after = 0

        for graph_id, (graph, graph_path) in enumerate(zip(self.graphs, graph_paths)):
            with open(graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            graph_payload = payload["0"] if "0" in payload else payload
            reindex = derive_topology_feature_order(graph_payload)

            if len(reindex.node_ids) != graph["n"]:
                raise RuntimeError(
                    f"Node count mismatch while reindexing graph {graph_id}: "
                    f"topology={len(reindex.node_ids)}, loader={graph['n']}."
                )

            graph["features"] = reorder_features(graph["features"], reindex.permutation)
            corrected_node_ids.append(reindex.node_ids)
            graph_consistent_edges_after = sum(
                reindex.node_ids[int(edge[0])] == str(triple[0])
                and reindex.node_ids[int(edge[1])] == str(triple[2])
                for edge, triple in zip(
                    graph_payload.get("edge_indices", []),
                    graph_payload.get("KG", []),
                )
            )
            if graph_consistent_edges_after != reindex.edge_count:
                raise RuntimeError(
                    f"Feature-topology reindex failed independent verification "
                    f"for graph {graph_id}: consistent={graph_consistent_edges_after}, "
                    f"edges={reindex.edge_count}."
                )

            changed_graphs += int(reindex.changed)
            reassigned_nodes += reindex.reassigned_nodes
            consistent_edges_before += reindex.consistent_edges_before
            consistent_edges_after += graph_consistent_edges_after
            edge_count += reindex.edge_count
            fully_consistent_graphs_before += int(reindex.fully_consistent_before)
            fully_consistent_graphs_after += int(
                graph_consistent_edges_after == reindex.edge_count
            )

        graph_count = len(self.graphs)
        self.corrected_node_ids = corrected_node_ids
        self.reindex_diagnostics = {
            "revision": self.feature_revision,
            "graphs": graph_count,
            "graphs_changed": changed_graphs,
            "reassigned_nodes": reassigned_nodes,
            "consistent_edges_before": consistent_edges_before,
            "consistent_edges_after": consistent_edges_after,
            "edges": edge_count,
            "edge_consistency_before": round(
                consistent_edges_before / edge_count if edge_count else 1.0,
                6,
            ),
            "edge_consistency_after": round(
                consistent_edges_after / edge_count if edge_count else 1.0,
                6,
            ),
            "fully_consistent_graphs_before": fully_consistent_graphs_before,
            "fully_consistent_graphs_after": fully_consistent_graphs_after,
        }
        if consistent_edges_after != edge_count or fully_consistent_graphs_after != graph_count:
            raise RuntimeError("Feature-topology reindex did not repair every edge.")

        self.load_data_time = time.time() - start
        print("Feature-topology reindex:", json.dumps(self.reindex_diagnostics, sort_keys=True))

    def _node_ids(self, graph_id):
        return self.corrected_node_ids[graph_id]

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        result = super().score(testing_graph_set, test_k, top_k_approach)
        result["feature_topology_reindex"] = self.reindex_diagnostics

        result_stem = (
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}"
            f"_k{test_k}_{self.args.repair_mode}"
        )
        result_path = self._result_file_path(result_stem)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("Recorded feature-topology diagnostics:", result_path)
        return result
