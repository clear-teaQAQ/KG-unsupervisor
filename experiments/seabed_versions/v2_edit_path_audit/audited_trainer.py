"""V2 audit: executable paths for the frozen V1 correspondence."""

from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V1_DIR = CURRENT_DIR.parent / "v1_certified_repair"
for path in (PROJECT_ROOT, V1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_evaluator import (
    audit_path,
    build_multirelation_edit_path,
    build_multirelation_graph,
    build_simple_edit_path,
    build_simple_graph,
    shared_entity_alignment,
)
from certified_trainer import CertifiedRepairTrainer
from src.SEABED.utils import get_file_paths


class EditPathAuditTrainer(CertifiedRepairTrainer):
    version = "v2_edit_path_audit"
    path_revision = "dual_executable_path_v1"

    def __init__(self, args):
        super().__init__(args)
        self.graph_paths = []
        for split in ("train", "val", "test"):
            self.graph_paths.extend(get_file_paths(str(Path(args.dataset_root) / split), "json"))
        if len(self.graph_paths) != len(self.graphs):
            raise RuntimeError(
                f"Graph path index has {len(self.graph_paths)} entries, loader has {len(self.graphs)} graphs."
            )
        self._node_id_cache = {}
        self._view_cache = {}

    def _node_ids(self, graph_id):
        if graph_id not in self._node_id_cache:
            with open(self.graph_paths[graph_id], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            graph_payload = payload["0"] if "0" in payload else payload
            node_ids = [str(node.get("id", index)) for index, node in enumerate(graph_payload["node_features"])]
            if len(node_ids) != self.graphs[graph_id]["n"]:
                raise RuntimeError(f"Node ID count mismatch for graph {graph_id}.")
            self._node_id_cache[graph_id] = node_ids
        return self._node_id_cache[graph_id]

    def _graph_views(self, graph_id):
        if graph_id not in self._view_cache:
            graph = self.graphs[graph_id]
            self._view_cache[graph_id] = (
                build_simple_graph(graph["n"], graph["graph"], graph["edge_ids"]),
                build_multirelation_graph(graph["n"], graph["graph"], graph["edge_ids"]),
            )
        return self._view_cache[graph_id]

    @staticmethod
    def _path_metrics(costs, ground_truth, audits, breakdowns):
        costs = np.asarray(costs, dtype=float)
        ground_truth = np.asarray(ground_truth, dtype=float)
        excess = costs - ground_truth
        normalized = excess / np.maximum(ground_truth, 1.0)
        operation_names = (
            "node_insertions",
            "node_deletions",
            "edge_insertions",
            "edge_deletions",
            "relation_substitutions",
        )
        return {
            "cost": CertifiedRepairTrainer._aggregate_cost_metrics(costs, ground_truth),
            "validity": {
                "mapping_validity_rate": round(float(np.mean([a["mapping_valid"] for a in audits])), 4),
                "cost_consistency_rate": round(float(np.mean([a["cost_consistent"] for a in audits])), 4),
                "replay_success_rate": round(float(np.mean([a["replay_success"] for a in audits])), 4),
            },
            "optimality": {
                "optimal_path_rate": round(float(np.mean(excess == 0)), 4),
                "negative_excess_rate": round(float(np.mean(excess < 0)), 4),
                "mean_excess_cost": round(float(np.mean(excess)), 4),
                "median_excess_cost": round(float(np.median(excess)), 4),
                "p95_excess_cost": round(float(np.percentile(excess, 95)), 4),
                "max_excess_cost": round(float(np.max(excess)), 4),
                "mean_normalized_gap": round(float(np.mean(normalized)), 4),
            },
            "average_operation_cost": {
                name: round(float(np.mean([breakdown[name] for breakdown in breakdowns])), 4)
                for name in operation_names
            },
        }

    def _ranking_metrics(self, grouped_predictions, grouped_ground_truth):
        rho = []
        tau = []
        pk = {k: [] for k in (1, 5, 10, 15, 20)}
        for graph_id, predictions in grouped_predictions.items():
            targets = grouped_ground_truth[graph_id]
            rho.append(spearmanr(predictions, targets)[0])
            tau.append(kendalltau(predictions, targets)[0])
            for k in pk:
                pk[k].append(self.cal_pk(k, predictions, targets))
        return {
            "rho": self._mean_rank_metric(rho),
            "tau": self._mean_rank_metric(tau),
            **{f"pk{k}": round(float(np.mean(values)), 3) for k, values in pk.items()},
        }

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if testing_graph_set != self.args.testset:
            raise ValueError(f"V2 initialized {self.args.testset}, not {testing_graph_set}.")
        if top_k_approach != "parallel":
            raise ValueError("V2 audits the frozen parallel V1 inference path only.")

        self.model.eval()
        ground_truth = []
        dense_initial_costs = []
        dense_final_costs = []
        simple_costs = []
        multi_costs = []
        simple_audits = []
        multi_audits = []
        simple_breakdowns = []
        multi_breakdowns = []
        repair_iterations = []
        repair_candidates = []
        inference_times = []
        path_times = []
        shared_entities = 0
        aligned_shared_entities = 0
        simple_matched_edges = 0
        simple_source_edges = 0
        multi_matched_edges = 0
        multi_source_edges = 0
        grouped_ground_truth = {}
        grouped_costs = {
            "v1_dense": {},
            "simple_graph": {},
            "multirelation": {},
        }

        result_stem = (
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}"
            f"_k{test_k}_{self.args.repair_mode}"
        )
        path_file = None
        path_handle = None
        saved_paths = 0
        if self.args.save_paths:
            path_file = self.result_dir / f"paths_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}_k{test_k}_{self.args.repair_mode}_{self.run_timestamp}.jsonl"
            path_handle = open(path_file, "w", encoding="utf-8")

        try:
            for pair_index, batch in enumerate(
                tqdm(
                    self.evaluation_data_loader,
                    total=len(self.evaluation_data_loader),
                    unit="pair",
                    dynamic_ncols=True,
                    desc=f"Eval {self.version} {testing_graph_set}",
                    file=sys.stdout,
                )
            ):
                batch.to(self.device)
                self.current_pair_index = pair_index
                _, _, inference_time = self.diffusion_ged_parallel(batch, test_k)
                repair = self.last_repair
                mapping = repair.mapping.detach().cpu().tolist()
                source_gid = int(batch.i_j[0][0].item())
                target_gid = int(batch.i_j[0][1].item())
                gt = float(batch.ged.item())

                path_start = time.time()
                source_simple, source_multi = self._graph_views(source_gid)
                target_simple, target_multi = self._graph_views(target_gid)
                source_node_ids = self._node_ids(source_gid)
                target_node_ids = self._node_ids(target_gid)
                simple_path = build_simple_edit_path(
                    mapping,
                    source_simple,
                    target_simple,
                    source_node_ids,
                    target_node_ids,
                )
                multi_path = build_multirelation_edit_path(
                    mapping,
                    source_multi,
                    target_multi,
                    source_node_ids,
                    target_node_ids,
                )
                simple_audit = audit_path(simple_path, source_simple, target_simple)
                multi_audit = audit_path(multi_path, source_multi, target_multi)
                if not all(simple_audit.values()) or not all(multi_audit.values()):
                    raise RuntimeError(
                        "V2 executable-path invariant failed: "
                        f"pair_index={pair_index}, graph_ids={[source_gid, target_gid]}, "
                        f"simple={simple_audit}, multirelation={multi_audit}"
                    )
                entity_alignment = shared_entity_alignment(
                    mapping,
                    source_node_ids,
                    target_node_ids,
                )
                path_time = time.time() - path_start

                ground_truth.append(gt)
                dense_initial_costs.append(repair.initial_cost)
                dense_final_costs.append(repair.final_cost)
                simple_costs.append(simple_path["total_cost"])
                multi_costs.append(multi_path["total_cost"])
                simple_audits.append(simple_audit)
                multi_audits.append(multi_audit)
                simple_breakdowns.append(simple_path["cost_breakdown"])
                multi_breakdowns.append(multi_path["cost_breakdown"])
                repair_iterations.append(repair.iterations)
                repair_candidates.append(repair.candidates_evaluated)
                inference_times.append(inference_time)
                path_times.append(path_time)
                shared_entities += entity_alignment["shared_entities"]
                aligned_shared_entities += entity_alignment["aligned_shared_entities"]
                simple_matched_edges += simple_path["matched_edge_count"]
                simple_source_edges += len(source_simple.edges)
                multi_matched_edges += multi_path["matched_edge_count"]
                multi_source_edges += sum(sum(relations.values()) for relations in source_multi.edges.values())

                grouped_ground_truth.setdefault(source_gid, []).append(gt)
                grouped_costs["v1_dense"].setdefault(source_gid, []).append(repair.final_cost)
                grouped_costs["simple_graph"].setdefault(source_gid, []).append(simple_path["total_cost"])
                grouped_costs["multirelation"].setdefault(source_gid, []).append(multi_path["total_cost"])

                save_this_path = path_handle is not None and (
                    self.args.max_saved_paths == 0 or saved_paths < self.args.max_saved_paths
                )
                if save_this_path:
                    path_record = {
                        "pair_index": pair_index,
                        "source_gid": source_gid,
                        "target_gid": target_gid,
                        "source_file": Path(self.graph_paths[source_gid]).name,
                        "target_file": Path(self.graph_paths[target_gid]).name,
                        "gt_ged": gt,
                        "v1_dense": {
                            "initial_cost": repair.initial_cost,
                            "final_cost": repair.final_cost,
                            "iterations": repair.iterations,
                            "candidates_evaluated": repair.candidates_evaluated,
                        },
                        "shared_entity_alignment": entity_alignment,
                        "simple_graph_path": simple_path,
                        "multirelation_path": multi_path,
                    }
                    path_handle.write(json.dumps(path_record, ensure_ascii=True) + "\n")
                    saved_paths += 1
        finally:
            if path_handle is not None:
                path_handle.close()

        if not ground_truth:
            raise RuntimeError("V2 received no graph pairs to audit.")

        dense_final = np.asarray(dense_final_costs)
        simple_array = np.asarray(simple_costs)
        multi_array = np.asarray(multi_costs)
        dense_reductions = np.asarray(dense_initial_costs) - dense_final
        result = {
            "version": self.version,
            "path_revision": self.path_revision,
            "frozen_inference_source": "v1_certified_repair/deterministic_dense_v4",
            "config": self._run_config(),
            "checkpoint_path": self.args.checkpoint_path,
            "num_pairs": len(ground_truth),
            "semantics": {
                "simple_graph": "undirected endpoint, one predicate, raw edge-list last-write-wins",
                "multirelation": "undirected endpoint, complete predicate multiset",
                "node_substitution_cost": 0,
                "node_insertion_deletion_cost": 1,
                "edge_insertion_deletion_cost": 1,
                "relation_substitution_cost": 1,
            },
            "v1_dense_reference": {
                "initial": self._aggregate_cost_metrics(dense_initial_costs, ground_truth),
                "final": self._aggregate_cost_metrics(dense_final_costs, ground_truth),
                "ranking": self._ranking_metrics(grouped_costs["v1_dense"], grouped_ground_truth),
                "improved_pair_rate": round(float(np.mean(dense_reductions > 0)), 4),
                "average_cost_reduction": round(float(np.mean(dense_reductions)), 4),
            },
            "simple_graph_path": {
                **self._path_metrics(simple_costs, ground_truth, simple_audits, simple_breakdowns),
                "ranking": self._ranking_metrics(grouped_costs["simple_graph"], grouped_ground_truth),
                "relation_preservation_rate": round(
                    simple_matched_edges / simple_source_edges if simple_source_edges else 1.0,
                    4,
                ),
            },
            "multirelation_path": {
                **self._path_metrics(multi_costs, ground_truth, multi_audits, multi_breakdowns),
                "ranking": self._ranking_metrics(grouped_costs["multirelation"], grouped_ground_truth),
                "relation_preservation_rate": round(
                    multi_matched_edges / multi_source_edges if multi_source_edges else 1.0,
                    4,
                ),
            },
            "evaluator_disagreement": {
                "simple_vs_v1_dense_rate": round(float(np.mean(simple_array != dense_final)), 4),
                "multirelation_vs_v1_dense_rate": round(float(np.mean(multi_array != dense_final)), 4),
                "simple_vs_multirelation_rate": round(float(np.mean(simple_array != multi_array)), 4),
                "mean_simple_minus_v1_dense": round(float(np.mean(simple_array - dense_final)), 4),
                "mean_multirelation_minus_v1_dense": round(float(np.mean(multi_array - dense_final)), 4),
                "mean_multirelation_minus_simple": round(float(np.mean(multi_array - simple_array)), 4),
            },
            "semantic_proxy": {
                "shared_entities": shared_entities,
                "aligned_shared_entities": aligned_shared_entities,
                "shared_entity_alignment_rate": round(
                    aligned_shared_entities / shared_entities if shared_entities else 0.0,
                    4,
                ),
                "pairs_without_shared_entities_are_excluded": True,
            },
            "repair": {
                "average_iterations": round(float(np.mean(repair_iterations)), 4),
                "average_candidates_evaluated": round(float(np.mean(repair_candidates)), 2),
            },
            "timing": {
                "inference_seconds_per_pair": round(float(np.mean(inference_times)), 5),
                "path_audit_seconds_per_pair": round(float(np.mean(path_times)), 5),
            },
            "saved_paths": saved_paths,
            "path_file": str(path_file) if path_file is not None else None,
        }

        result_path = self._result_file_path(result_stem)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
        print("Saved result:", result_path)
        if path_file is not None:
            print("Saved executable paths:", path_file)
        return result
