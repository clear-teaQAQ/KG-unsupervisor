"""Frozen V15 inference with same-mapping executable-path cost auditing."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
for path in (PROJECT_ROOT, CURRENT_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_evaluator import audit_path, build_simple_edit_path, build_simple_graph  # noqa: E402
from relation_trainer import V15ProjectedRelationTrainer  # noqa: E402


def signed_error_bucket(error):
    error = int(error)
    if error <= -2:
        return "<=-2"
    if error == -1:
        return "-1"
    if error == 0:
        return "0"
    if error == 1:
        return "+1"
    return ">=+2"


def magnitude_bucket(error):
    magnitude = abs(int(error))
    if magnitude == 0:
        return "0"
    if magnitude == 1:
        return "1"
    if magnitude == 2:
        return "2"
    return ">=3"


def aggregate_costs(costs, ground_truth):
    costs = np.asarray(costs, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    errors = costs - ground_truth
    return {
        "pairs": int(len(costs)),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "mse": round(float(np.mean(errors ** 2)), 4),
        "acc": round(float(np.mean(errors == 0)), 4),
        "fea": round(float(np.mean(errors >= 0)), 4),
        "mean_signed_error": round(float(np.mean(errors)), 4),
        "signed_error_buckets": dict(
            sorted(Counter(signed_error_bucket(error) for error in errors).items())
        ),
        "absolute_error_buckets": dict(
            sorted(Counter(magnitude_bucket(error) for error in errors).items())
        ),
    }


class V15FrozenMappingAuditTrainer(V15ProjectedRelationTrainer):
    audit_revision = "v15_frozen_same_mapping_simple_path_v1"

    def _simple_graph_view(self, graph_id):
        graph_id = int(graph_id)
        if graph_id not in self._simple_graph_cache:
            graph = self.graphs[graph_id]
            self._simple_graph_cache[graph_id] = build_simple_graph(
                graph["n"],
                graph["graph"],
                graph["edge_ids"],
            )
        return self._simple_graph_cache[graph_id]

    def diffusion_ged_parallel(self, batch, test_k=100):
        legacy_cost, solution, running_time = super().diffusion_ged_parallel(
            batch, test_k
        )
        mapping = np.argmax(solution.detach().cpu().numpy(), axis=1).tolist()
        source_gid, target_gid = [int(value) for value in batch.i_j[0].tolist()]
        ground_truth = float(batch.ged.item())
        source_graph = self._simple_graph_view(source_gid)
        target_graph = self._simple_graph_view(target_gid)
        simple_path = build_simple_edit_path(mapping, source_graph, target_graph)
        path_audit = audit_path(simple_path, source_graph, target_graph)
        if not all(path_audit.values()):
            raise RuntimeError(
                "Executable-path invariant failed: "
                f"graph_ids={[source_gid, target_gid]}, audit={path_audit}"
            )

        simple_cost = float(simple_path["total_cost"])
        record = {
            "pair_index": len(self._mapping_audit_records),
            "graph_ids": [source_gid, target_gid],
            "node_counts": [source_graph.num_nodes, target_graph.num_nodes],
            "ground_truth": ground_truth,
            "mapping": mapping,
            "legacy_selected_cost": float(legacy_cost),
            "simple_path_same_mapping_cost": simple_cost,
            "legacy_error": float(legacy_cost - ground_truth),
            "simple_path_error": float(simple_cost - ground_truth),
            "simple_minus_legacy": float(simple_cost - legacy_cost),
            "simple_path_breakdown": simple_path["cost_breakdown"],
            "simple_matched_edge_count": simple_path["matched_edge_count"],
            "path_audit": path_audit,
        }
        self._mapping_audit_records.append(record)
        self._mapping_audit_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._mapping_audit_handle.flush()
        return legacy_cost, solution, running_time

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if top_k_approach != "parallel":
            raise ValueError("Frozen mapping audit requires parallel inference.")
        self._simple_graph_cache = {}
        self._mapping_audit_records = []
        path_file = self.result_dir / (
            f"mappings_SEABED_SWDF_{testing_graph_set}_k{test_k}_"
            f"{self.run_timestamp}.jsonl"
        )
        self._mapping_audit_handle = path_file.open("w", encoding="utf-8")
        try:
            legacy_payload = super().score(
                testing_graph_set, test_k, top_k_approach
            )
        finally:
            self._mapping_audit_handle.close()

        if not self._mapping_audit_records:
            raise RuntimeError("Frozen mapping audit received no evaluated pairs.")
        ground_truth = [
            record["ground_truth"] for record in self._mapping_audit_records
        ]
        legacy_costs = [
            record["legacy_selected_cost"] for record in self._mapping_audit_records
        ]
        simple_costs = [
            record["simple_path_same_mapping_cost"]
            for record in self._mapping_audit_records
        ]
        disagreements = [
            record["simple_minus_legacy"] for record in self._mapping_audit_records
        ]
        disagreement_counts = Counter(disagreements)
        result = {
            "audit_revision": self.audit_revision,
            "checkpoint_path": getattr(self, "loaded_checkpoint_path", None),
            "selection_policy": (
                "unchanged V15 legacy unit-cost best-of-k; executable simple "
                "cost is evaluated after selection on the same mapping"
            ),
            "inference_is_new_seeded_sample": True,
            "seed": int(getattr(self.args, "audit_seed", 0)),
            "test_k": test_k,
            "cost_and_task_unchanged": {
                "cost_mode": self.args.cost_mode,
                "ged_column": self.args.ged_column,
                "ground_truth_changed": False,
                "candidate_selection_changed": False,
                "primary_benchmark_overwritten": False,
            },
            "legacy_selected": aggregate_costs(legacy_costs, ground_truth),
            "simple_path_same_mapping": aggregate_costs(simple_costs, ground_truth),
            "evaluator_disagreement": {
                "pairs": len(disagreements),
                "different_cost_pairs": int(
                    np.count_nonzero(np.asarray(disagreements) != 0)
                ),
                "different_cost_rate": round(
                    float(np.mean(np.asarray(disagreements) != 0)), 4
                ),
                "mean_simple_minus_legacy": round(
                    float(np.mean(disagreements)), 4
                ),
                "simple_minus_legacy_counts": {
                    str(int(cost)): count
                    for cost, count in sorted(disagreement_counts.items())
                },
            },
            "operation_cost_means": {
                operation: round(
                    float(
                        np.mean(
                            [
                                record["simple_path_breakdown"][operation]
                                for record in self._mapping_audit_records
                            ]
                        )
                    ),
                    4,
                )
                for operation in (
                    "node_insertions",
                    "node_deletions",
                    "edge_insertions",
                    "edge_deletions",
                    "relation_substitutions",
                )
            },
            "mapping_file": str(path_file.resolve()),
            "legacy_result_payload": legacy_payload,
        }
        result_path = self.result_dir / (
            f"frozen_mapping_audit_SEABED_SWDF_{testing_graph_set}_k{test_k}_"
            f"{self.run_timestamp}.json"
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print("Frozen mapping audit:", json.dumps(result, indent=2))
        print("Saved mapping audit:", result_path)
        print("Saved per-pair mappings:", path_file)
        return result

