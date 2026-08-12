"""Rerank the same frozen V15 candidates with official simple-graph cost."""

from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
for path in (PROJECT_ROOT, CURRENT_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frozen_mapping_audit import aggregate_costs  # noqa: E402
from path_evaluator import audit_path, build_simple_edit_path, build_simple_graph  # noqa: E402
from relation_trainer import V15ProjectedRelationTrainer  # noqa: E402
from src.SEABED.trainer import InferenceSchedule  # noqa: E402


class V15CandidateRerankAuditTrainer(V15ProjectedRelationTrainer):
    audit_revision = "v15_frozen_candidate_official_rerank_v1"

    def _simple_graph_view(self, graph_id):
        graph_id = int(graph_id)
        if graph_id not in self._simple_graph_cache:
            graph = self.graphs[graph_id]
            self._simple_graph_cache[graph_id] = build_simple_graph(
                graph["n"], graph["graph"], graph["edge_ids"]
            )
        return self._simple_graph_cache[graph_id]

    def _decode_candidates(self, batch, test_k):
        data = batch[0]
        candidates = Batch.from_data_list([data for _ in range(test_k)])
        mapping_t = torch.randn_like(
            candidates.edge_attr_mapping, device=self.device
        )
        mapping_t = (mapping_t > 0).long()
        schedule = InferenceSchedule(
            T=self.diffusion.T,
            inference_T=self.args.inference_diffusion_steps,
        )
        for step in range(self.args.inference_diffusion_steps):
            t1, t2 = schedule(step)
            mapping_t = self.categorical_denoise_step(
                candidates,
                mapping_t,
                np.asarray([t1], dtype=int),
                np.asarray([t2], dtype=int),
            )

        n1 = int(batch.n[0, 0].item())
        n2 = int(batch.n[0, 1].item())
        scores = torch.zeros((test_k, n1, n2), device=self.device)
        mapping_edges = candidates.edge_index_mapping
        local_edges = (
            mapping_edges
            - candidates.batch[mapping_edges[0]] * (n1 + n2)
        )
        local_edges[1] -= n1
        scores[
            candidates.batch[mapping_edges[0]],
            local_edges[0],
            local_edges[1],
        ] = mapping_t.squeeze(-1)

        batch_ids = torch.arange(test_k, device=self.device)
        unavailable = torch.zeros_like(scores, dtype=torch.bool)
        source_solution = torch.zeros_like(scores, dtype=torch.bool)
        for _ in range(n1):
            flat_scores = scores.view(test_k, -1)
            best = torch.argmax(flat_scores, dim=-1)
            rows = best // n2
            columns = best % n2
            source_solution[batch_ids, rows, columns] = True
            unavailable[batch_ids, rows, :] = True
            unavailable[batch_ids, :, columns] = True
            scores = flat_scores.view(test_k, n1, n2)
            scores[unavailable] = float("-inf")

        if not torch.all(source_solution.sum(dim=2) == 1):
            raise RuntimeError("Candidate decoding produced a source row without a match.")
        if not torch.all(source_solution.sum(dim=1) <= 1):
            raise RuntimeError("Candidate decoding produced a non-injective mapping.")
        return source_solution

    def diffusion_ged_parallel(self, batch, test_k=100):
        start_time = time.time()
        source_solutions = self._decode_candidates(batch, test_k)
        n1 = int(batch.n[0, 0].item())
        source_gid, target_gid = [int(value) for value in batch.i_j[0].tolist()]
        ground_truth = float(batch.ged.item())
        source_graph = self._simple_graph_view(source_gid)
        target_graph = self._simple_graph_view(target_gid)

        legacy_costs = []
        official_costs = []
        mappings = []
        official_paths = []
        for candidate_index in range(test_k):
            solution = source_solutions[candidate_index]
            legacy_costs.append(
                float(self._compute_single_ged_from_dense_solution(solution, batch[0]))
            )
            mapping = torch.argmax(solution.long(), dim=1).detach().cpu().tolist()
            path = build_simple_edit_path(mapping, source_graph, target_graph)
            path_check = audit_path(path, source_graph, target_graph)
            if not all(path_check.values()):
                raise RuntimeError(
                    "Official candidate path failed replay: "
                    f"pair={len(self._rerank_records)}, candidate={candidate_index}, "
                    f"graph_ids={[source_gid, target_gid]}, audit={path_check}"
                )
            mappings.append(mapping)
            official_paths.append(path)
            official_costs.append(float(path["total_cost"]))

        legacy_index = int(np.argmin(legacy_costs))
        official_index = int(np.argmin(official_costs))
        legacy_selected_official_cost = official_costs[legacy_index]
        official_selected_cost = official_costs[official_index]
        if official_selected_cost < ground_truth:
            raise RuntimeError(
                "Official executable candidate cost is below ground truth: "
                f"pair={len(self._rerank_records)}, cost={official_selected_cost}, "
                f"ground_truth={ground_truth}"
            )

        unique_mappings = len({tuple(mapping) for mapping in mappings})
        record = {
            "pair_index": len(self._rerank_records),
            "graph_ids": [source_gid, target_gid],
            "node_counts": [source_graph.num_nodes, target_graph.num_nodes],
            "ground_truth": ground_truth,
            "test_k": test_k,
            "unique_candidate_mappings": unique_mappings,
            "legacy_selected": {
                "candidate_index": legacy_index,
                "mapping": mappings[legacy_index],
                "legacy_cost": legacy_costs[legacy_index],
                "official_cost": legacy_selected_official_cost,
            },
            "official_selected": {
                "candidate_index": official_index,
                "mapping": mappings[official_index],
                "official_cost": official_selected_cost,
                "cost_breakdown": official_paths[official_index]["cost_breakdown"],
            },
            "selection_same_candidate": legacy_index == official_index,
            "official_improvement_over_legacy_selection": (
                legacy_selected_official_cost - official_selected_cost
            ),
            "candidate_coverage": official_selected_cost == ground_truth,
            "legacy_candidate_cost_min": min(legacy_costs),
            "legacy_candidate_cost_max": max(legacy_costs),
            "official_candidate_cost_min": min(official_costs),
            "official_candidate_cost_max": max(official_costs),
        }
        self._rerank_records.append(record)
        self._rerank_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._rerank_handle.flush()

        # The base score now reports the official-reranked complete-test metric.
        selected_solution = source_solutions[official_index, :n1]
        return official_selected_cost, selected_solution, time.time() - start_time

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if top_k_approach != "parallel":
            raise ValueError("Candidate rerank audit requires parallel inference.")
        self._simple_graph_cache = {}
        self._rerank_records = []
        pair_file = self.result_dir / (
            f"candidate_rerank_pairs_SEABED_SWDF_{testing_graph_set}_k{test_k}_"
            f"{self.run_timestamp}.jsonl"
        )
        self._rerank_handle = pair_file.open("w", encoding="utf-8")
        try:
            official_score_payload = super().score(
                testing_graph_set, test_k, top_k_approach
            )
        finally:
            self._rerank_handle.close()

        if not self._rerank_records:
            raise RuntimeError("Candidate rerank audit evaluated no pairs.")
        ground_truth = [record["ground_truth"] for record in self._rerank_records]
        legacy_min_costs = [
            record["legacy_selected"]["legacy_cost"]
            for record in self._rerank_records
        ]
        legacy_selected_official_costs = [
            record["legacy_selected"]["official_cost"]
            for record in self._rerank_records
        ]
        official_min_costs = [
            record["official_selected"]["official_cost"]
            for record in self._rerank_records
        ]
        improvements = [
            record["official_improvement_over_legacy_selection"]
            for record in self._rerank_records
        ]
        improvement_counts = Counter(improvements)
        coverage_count = sum(
            record["candidate_coverage"] for record in self._rerank_records
        )
        same_selection_count = sum(
            record["selection_same_candidate"] for record in self._rerank_records
        )
        expected_pairs = len(self.split_pairs[testing_graph_set])
        complete_test_acc = (
            testing_graph_set == "test"
            and len(self._rerank_records) == expected_pairs
            and int(getattr(self.args, "max_test_pairs", 0)) == 0
        )
        result = {
            "audit_revision": self.audit_revision,
            "checkpoint_path": getattr(self, "loaded_checkpoint_path", None),
            "seed": int(getattr(self.args, "audit_seed", 0)),
            "test_k": test_k,
            "pairs": len(self._rerank_records),
            "expected_split_pairs": expected_pairs,
            "complete_test_acc": complete_test_acc,
            "cost_and_task_unchanged": {
                "cost_mode": self.args.cost_mode,
                "ged_column": self.args.ged_column,
                "ground_truth_changed": False,
                "checkpoint_changed": False,
                "candidates_changed_between_selections": False,
                "training_performed": False,
            },
            "legacy_cost_selected": aggregate_costs(
                legacy_min_costs, ground_truth
            ),
            "legacy_selected_official_replay": aggregate_costs(
                legacy_selected_official_costs, ground_truth
            ),
            "official_candidate_rerank": aggregate_costs(
                official_min_costs, ground_truth
            ),
            "candidate_coverage": {
                "covered_pairs": coverage_count,
                "total_pairs": len(self._rerank_records),
                "rate": round(coverage_count / len(self._rerank_records), 4),
                "definition": (
                    "at least one of the same test_k candidates has official "
                    "last-write executable cost equal to column-3 ground truth"
                ),
            },
            "selection_comparison": {
                "same_candidate_pairs": same_selection_count,
                "same_candidate_rate": round(
                    same_selection_count / len(self._rerank_records), 4
                ),
                "official_rerank_improved_pairs": int(
                    np.count_nonzero(np.asarray(improvements) > 0)
                ),
                "mean_official_cost_reduction": round(
                    float(np.mean(improvements)), 4
                ),
                "official_cost_reduction_counts": {
                    str(int(value)): count
                    for value, count in sorted(improvement_counts.items())
                },
            },
            "candidate_diversity": {
                "mean_unique_mappings": round(
                    float(
                        np.mean(
                            [
                                record["unique_candidate_mappings"]
                                for record in self._rerank_records
                            ]
                        )
                    ),
                    4,
                ),
                "min_unique_mappings": min(
                    record["unique_candidate_mappings"]
                    for record in self._rerank_records
                ),
                "max_unique_mappings": max(
                    record["unique_candidate_mappings"]
                    for record in self._rerank_records
                ),
            },
            "pair_file": str(pair_file.resolve()),
            "official_base_score_payload": official_score_payload,
        }
        result_path = self.result_dir / (
            f"candidate_rerank_audit_SEABED_SWDF_{testing_graph_set}_k{test_k}_"
            f"{self.run_timestamp}.json"
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print("Candidate rerank audit:", json.dumps(result, indent=2))
        print("Saved candidate rerank audit:", result_path)
        print("Saved per-pair rerank records:", pair_file)
        return result
