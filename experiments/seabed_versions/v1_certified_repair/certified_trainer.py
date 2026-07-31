"""Inference-only V1 trainer that leaves the frozen V0 implementation untouched."""

from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.SEABED.trainer import InferenceSchedule, Trainer as BaselineTrainer

from repair import (
    RepairResult,
    deterministic_labeled_adjacency,
    permutation_unit_cost,
    permutation_unit_costs,
    repair_mapping,
    size_lower_bound,
    unit_cost,
)


class CertifiedRepairTrainer(BaselineTrainer):
    version = "v1_certified_repair"
    repair_cost_revision = "deterministic_dense_v4"

    def __init__(self, args):
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []
        self.founded_ged = []
        self.project_root = PROJECT_ROOT
        self.result_dir = (self.project_root / self.args.result_path).resolve()
        self.model_dir = (self.project_root / self.args.model_path).resolve()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        print("version =", self.version)
        print("device =", self.device)

        self.load_data()
        self.transfer_data_to_torch()
        self.setup_model()
        split_limit = getattr(self.args, f"max_{self.args.testset}_pairs")
        selected_pairs = self._limit_pairs(self.split_pairs[self.args.testset], split_limit)
        self.evaluation_graphs = [self.pack_graph_pair(pair) for pair in selected_pairs]
        self.evaluation_data_loader = DataLoader(self.evaluation_graphs, batch_size=1, shuffle=False)
        print(f"Generated {len(self.evaluation_graphs)} {self.args.testset} graph pairs.")
        self.last_repair = None

    def load_explicit_checkpoint(self, checkpoint_path):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        try:
            state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded checkpoint:", checkpoint)

    def diffusion_ged_parallel(self, batch, test_k=100):
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
        new_batch = Batch.from_data_list([data for _ in range(num_parallel_sampling)])
        mapping_t = torch.randn_like(new_batch.edge_attr_mapping, device=self.device)
        mapping_t = (mapping_t > 0).long()
        time_schedule = InferenceSchedule(
            T=self.diffusion.T,
            inference_T=self.args.inference_diffusion_steps,
        )

        for step in range(self.args.inference_diffusion_steps):
            t1, t2 = time_schedule(step)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(new_batch, mapping_t, t1, t2)

        n1 = int(batch.n[0, 0].item())
        n2 = int(batch.n[0, 1].item())
        pred_matching_matrix = torch.zeros(
            (num_parallel_sampling, n1, n2),
            device=self.device,
        )
        mapping_edge_idx = new_batch.edge_index_mapping
        batch_mapping_edge_idx = (
            mapping_edge_idx
            - new_batch.batch[mapping_edge_idx[0]] * (n1 + n2)
        )
        batch_mapping_edge_idx[1] -= n1
        pred_matching_matrix[
            new_batch.batch[mapping_edge_idx[0]],
            batch_mapping_edge_idx[0],
            batch_mapping_edge_idx[1],
        ] = mapping_t.squeeze(-1)

        batch_idx = torch.arange(num_parallel_sampling, device=self.device)
        greedy_mask = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
        source_solution = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
        for _ in range(n1):
            flat_matrix = pred_matching_matrix.view(num_parallel_sampling, -1)
            argmax_result = torch.argmax(flat_matrix, dim=-1)
            rows = argmax_result // n2
            columns = argmax_result % n2
            source_solution[batch_idx, rows, columns] = True
            greedy_mask[batch_idx, rows, :] = True
            greedy_mask[batch_idx, :, columns] = True
            pred_matching_matrix = flat_matrix.view(num_parallel_sampling, n1, n2)
            pred_matching_matrix[greedy_mask] = float("-inf")

        row_valid = torch.all(source_solution.sum(dim=2) == 1, dim=1)
        column_valid = torch.all(source_solution.sum(dim=1) <= 1, dim=1)
        valid_candidates = row_valid & column_valid
        if not torch.all(valid_candidates):
            invalid_count = int(torch.count_nonzero(~valid_candidates).item())
            raise RuntimeError(
                f"V0 greedy decoding produced {invalid_count}/{test_k} non-injective source mappings."
            )

        occupied = torch.any(source_solution, dim=1)
        target_ids = torch.arange(n2, device=self.device)
        unmatched = target_ids.unsqueeze(0).expand(num_parallel_sampling, -1)[~occupied]
        unmatched = unmatched.view(num_parallel_sampling, n2 - n1)
        source_mappings = torch.argmax(source_solution.long(), dim=2)
        permutations = torch.cat([source_mappings, unmatched], dim=1)

        adj_1, adj_2 = deterministic_labeled_adjacency(
            n1,
            n2,
            data.edge_index,
            data.edge_labels,
            self.device,
        )
        ged_values = permutation_unit_costs(
            permutations,
            adj_1,
            adj_2,
            candidate_batch_size=self.args.repair_candidate_batch_size,
        )
        min_ged_idx = int(torch.argmin(ged_values).item())
        initial_ged = int(ged_values[min_ged_idx].item())
        initial_permutation = permutations[min_ged_idx]
        initial_mapping = initial_permutation[:n1]
        verified_permutation_cost = permutation_unit_cost(
            initial_permutation,
            adj_1,
            adj_2,
        )
        verified_initial_cost = unit_cost(initial_mapping, adj_1, adj_2)
        if verified_permutation_cost != initial_ged or verified_initial_cost != initial_ged:
            graph_ids = batch.i_j[0].tolist()
            raise RuntimeError(
                "V1 cost mismatch after complete-permutation verification: "
                f"pair_index={self.current_pair_index}, graph_ids={graph_ids}, "
                f"baseline={initial_ged}, full={verified_permutation_cost}, "
                f"canonical={verified_initial_cost}, mapping={initial_mapping.tolist()}, "
                f"permutation={initial_permutation.tolist()}"
            )

        if self.args.repair_mode == "two_swap":
            repair = repair_mapping(
                initial_mapping,
                adj_1,
                adj_2,
                max_iterations=self.args.repair_max_iterations,
                candidate_batch_size=self.args.repair_candidate_batch_size,
            )
        else:
            bound = size_lower_bound(adj_1, adj_2)
            repair = RepairResult(
                mapping=initial_mapping,
                initial_cost=verified_initial_cost,
                final_cost=verified_initial_cost,
                lower_bound=bound,
                iterations=0,
                candidates_evaluated=0,
            )

        repaired_solution = torch.zeros((n1, n2), dtype=torch.bool, device=self.device)
        repaired_solution[torch.arange(repair.mapping.numel(), device=self.device), repair.mapping] = True
        self.last_repair = repair
        return float(repair.final_cost), repaired_solution, time.time() - start_time

    @staticmethod
    def _mean_rank_metric(values):
        finite_values = [value for value in values if not np.isnan(value)]
        return round(float(np.mean(finite_values)), 3) if finite_values else 0.0

    @staticmethod
    def _aggregate_cost_metrics(predictions, ground_truth):
        errors = np.asarray(predictions, dtype=float) - np.asarray(ground_truth, dtype=float)
        return {
            "mse": round(float(np.mean(errors ** 2)), 3),
            "mae": round(float(np.mean(np.abs(errors))), 3),
            "acc": round(float(np.mean(errors == 0)), 3),
            "fea": round(float(np.mean(errors >= 0)), 3),
        }

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if testing_graph_set != self.args.testset:
            raise ValueError(f"V1 initialized {self.args.testset}, not {testing_graph_set}.")
        if top_k_approach != "parallel":
            raise ValueError("V1 currently isolates the parallel V0 inference path only.")

        self.model.eval()
        initial_predictions = []
        final_predictions = []
        ground_truth = []
        evaluator_lower_bounds = []
        graph_size_lower_bounds = []
        repair_iterations = []
        candidates_evaluated = []
        pair_times = []
        pair_details = []
        grouped_predictions = {}
        grouped_ground_truth = {}

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
            final_ged, _, running_time = self.diffusion_ged_parallel(batch, test_k)
            repair = self.last_repair
            gt = float(batch.ged.item())
            source_idx = int(batch.i_j[0][0].item())
            graph_size_bound = int(
                torch.abs(batch.n[0, 1] - batch.n[0, 0]).item()
                + torch.abs(batch.m[0, 1] - batch.m[0, 0]).item()
            )

            initial_predictions.append(repair.initial_cost)
            final_predictions.append(repair.final_cost)
            ground_truth.append(gt)
            evaluator_lower_bounds.append(repair.lower_bound)
            graph_size_lower_bounds.append(graph_size_bound)
            repair_iterations.append(repair.iterations)
            candidates_evaluated.append(repair.candidates_evaluated)
            pair_times.append(running_time)
            grouped_predictions.setdefault(source_idx, []).append(final_ged)
            grouped_ground_truth.setdefault(source_idx, []).append(gt)

            if self.args.save_pair_details:
                pair_details.append(
                    {
                        "pair_index": pair_index,
                        "source_gid": source_idx,
                        "target_gid": int(batch.i_j[0][1].item()),
                        "gt": gt,
                        "evaluator_lower_bound": repair.lower_bound,
                        "graph_size_lower_bound": graph_size_bound,
                        "initial_cost": repair.initial_cost,
                        "final_cost": repair.final_cost,
                        "iterations": repair.iterations,
                        "candidates_evaluated": repair.candidates_evaluated,
                        "certified": repair.certified,
                        "time": running_time,
                    }
                )

        initial_metrics = self._aggregate_cost_metrics(initial_predictions, ground_truth)
        final_metrics = self._aggregate_cost_metrics(final_predictions, ground_truth)
        reductions = np.asarray(initial_predictions) - np.asarray(final_predictions)
        evaluator_bounds = np.asarray(evaluator_lower_bounds)
        graph_size_bounds = np.asarray(graph_size_lower_bounds)
        initial_residuals = np.asarray(initial_predictions) - evaluator_bounds
        final_residuals = np.asarray(final_predictions) - evaluator_bounds

        rho = []
        tau = []
        pk = {k: [] for k in (1, 5, 10, 15, 20)}
        for graph_id in grouped_predictions:
            predictions = grouped_predictions[graph_id]
            targets = grouped_ground_truth[graph_id]
            rho.append(spearmanr(predictions, targets)[0])
            tau.append(kendalltau(predictions, targets)[0])
            for k in pk:
                pk[k].append(self.cal_pk(k, predictions, targets))

        result = {
            "version": self.version,
            "repair_cost_revision": self.repair_cost_revision,
            "config": self._run_config(),
            "checkpoint_path": self.args.checkpoint_path,
            "num_pairs": len(final_predictions),
            "initial": initial_metrics,
            "final": final_metrics,
            "repair": {
                "improved_pair_rate": round(float(np.mean(reductions > 0)), 4),
                "average_cost_reduction": round(float(np.mean(reductions)), 4),
                "max_cost_reduction": int(np.max(reductions)),
                "initial_lower_bound_hit_rate": round(float(np.mean(initial_residuals == 0)), 4),
                "final_lower_bound_hit_rate": round(float(np.mean(final_residuals == 0)), 4),
                "label_equals_evaluator_lower_bound_rate": round(
                    float(np.mean(np.asarray(ground_truth) == evaluator_bounds)), 4
                ),
                "label_equals_graph_size_lower_bound_rate": round(
                    float(np.mean(np.asarray(ground_truth) == graph_size_bounds)), 4
                ),
                "final_below_graph_size_lower_bound_rate": round(
                    float(np.mean(np.asarray(final_predictions) < graph_size_bounds)), 4
                ),
                "average_initial_residual": round(float(np.mean(initial_residuals)), 4),
                "average_final_residual": round(float(np.mean(final_residuals)), 4),
                "average_iterations": round(float(np.mean(repair_iterations)), 4),
                "average_candidates_evaluated": round(float(np.mean(candidates_evaluated)), 2),
            },
            "ranking": {
                "rho": self._mean_rank_metric(rho),
                "tau": self._mean_rank_metric(tau),
                **{f"pk{k}": round(float(np.mean(values)), 3) for k, values in pk.items()},
            },
            "time_per_pair": round(float(np.mean(pair_times)), 5),
            "pair_details": pair_details if self.args.save_pair_details else None,
        }

        result_path = self._result_file_path(
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}_k{test_k}_{self.args.repair_mode}"
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        print(json.dumps({key: value for key, value in result.items() if key != "pair_details"}, indent=2))
        print("Saved result:", result_path)
        return result
