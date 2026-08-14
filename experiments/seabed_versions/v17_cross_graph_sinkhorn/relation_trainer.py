"""V17 trainer: V16 official graph semantics with a cross-graph matcher."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V16_DIR = CURRENT_DIR.parent / "v16_unified_official_graph"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_v16():
    path = V16_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location("v16_for_v17", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V16 trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.V16UnifiedOfficialGraphTrainer


V16Trainer = _load_v16()
from src.GEDRanker.loss_fn import bpr_loss, hinge_loss, mapping_loss  # noqa: E402
from v17_models import (  # noqa: E402
    CrossGraphSinkhornMatcher,
    direct_sinkhorn_candidates,
    direct_sinkhorn_rollout,
)


class V17CrossGraphSinkhornTrainer(V16Trainer):
    version = "v17_cross_graph_sinkhorn"
    v17_revision = "direct_cross_graph_gine_batched_gumbel_sinkhorn_v3"

    def setup_model(self):
        self.sinkhorn_iterations = max(int(self.args.gumbel_iteration), 20)
        self.model = CrossGraphSinkhornMatcher(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        # Keep the ranking critic unchanged so the only model change is the
        # candidate-generation architecture.
        from experiments.seabed_versions.v11_relation_aware_ged_training.relation_models import (
            RelationAwareDiscriminator,
        )
        self.D = RelationAwareDiscriminator(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizerD = torch.optim.RMSprop(
            self.D.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        # V17 is intentionally not a diffusion model. Training and inference
        # both consume the same static cross-graph assignment logits.

    def process_batch(self, batch, indices):
        best_ged = batch.best_ged
        pred_mapping_logits = self.model(batch)
        best_mapping_label = batch.best_mapping_label.to(pred_mapping_logits.dtype)
        last_mapping_label = batch.last_mapping_label.to(pred_mapping_logits.dtype)
        pred_solution, pred_solution_soft = direct_sinkhorn_rollout(
            pred_mapping_logits,
            batch,
            self.args.tau,
            self.sinkhorn_iterations,
        )
        pred_ged = self._compute_batch_ged(pred_solution, batch)

        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_n.squeeze(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_n.squeeze(-1))
        normalize_last_ged = torch.exp(-batch.last_ged / batch.avg_n.squeeze(-1))
        alpha = max(1 - self.cur_epoch / (self.args.model_epoch_end / 2), 0)

        if alpha > 0:
            current_score = self.D(batch, pred_solution_soft.detach())
            best_score = self.D(batch, best_mapping_label)
            last_score = self.D(batch, last_mapping_label)
            if self.args.unsupervised_approach == "BPR":
                discriminator_loss = bpr_loss(
                    current_score,
                    best_score,
                    last_score,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif self.args.unsupervised_approach == "Hinge":
                discriminator_loss = hinge_loss(
                    current_score,
                    best_score,
                    last_score,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif self.args.unsupervised_approach == "GED":
                discriminator_loss = ((current_score - normalize_curr_ged) ** 2).sum()
            else:
                discriminator_loss = torch.zeros((), device=self.device)

            if discriminator_loss.requires_grad:
                self.optimizerD.zero_grad()
                discriminator_loss.backward()
                self.optimizerD.step()
        else:
            discriminator_loss = torch.zeros((), device=self.device)

        map_loss = mapping_loss(
            pred_mapping_logits, batch, best_mapping_label
        )
        if self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            ged_loss = -self.D(batch, pred_solution_soft).sum()
        else:
            ged_loss = torch.zeros((), device=self.device)

        total_loss = map_loss + ged_loss * alpha
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        new_solution = 0
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        for index in range(len(indices)):
            graph = self.training_graphs[indices[index]]
            mask = mapping_batch == index
            graph.last_mapping_label = pred_solution[mask].to(
                graph.last_mapping_label.device
            )
            graph.last_ged = pred_ged[index].to(graph.last_ged.device)
            if pred_ged[index] < best_ged[index]:
                new_solution += 1
                graph.best_ged = pred_ged[index].to(graph.best_ged.device)
                graph.best_mapping_label = pred_solution[mask].to(
                    graph.best_mapping_label.device
                )

        return (
            total_loss.item(),
            discriminator_loss.item(),
            pred_ged.sum().item(),
            batch.ged.sum().item(),
            best_ged.sum().item(),
            map_loss.item(),
            ged_loss.item(),
            new_solution,
        )

    @torch.no_grad()
    def diffusion_ged_parallel(self, batch, test_k=100):
        """Direct best-of-K Sinkhorn inference; no diffusion posterior."""
        start_time = time.time()
        data = batch[0]
        source_nodes = int(data.n[0, 0].item())
        target_nodes = int(data.n[0, 1].item())
        pair_logits = self.model(batch)
        candidates, _ = direct_sinkhorn_candidates(
            pair_logits,
            source_nodes,
            target_nodes,
            sample_count=test_k,
            tau=self.args.tau,
            iterations=self.sinkhorn_iterations,
            stochastic=test_k > 1,
            include_deterministic=True,
        )
        costs = torch.tensor(
            [
                self._compute_single_ged_from_dense_solution(candidate, data)
                for candidate in candidates
            ],
            device=self.device,
        )
        best_index = int(torch.argmin(costs).item())
        return (
            float(costs[best_index].item()),
            candidates[best_index],
            time.time() - start_time,
        )

    def diffusion_ged_sequential(self, batch, test_k=100):
        return self.diffusion_ged_parallel(batch, test_k)

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update({
            "version": self.version,
            "v17_revision": self.v17_revision,
            "generator_architecture": "cross_graph_gine_attention",
            "assignment_relaxation": "direct_padded_gumbel_sinkhorn",
            "training_uses_diffusion": False,
            "inference_uses_diffusion": False,
            "inference_candidate_sampling": "deterministic_plus_gumbel_best_of_k",
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "cost_mode": "unit",
            "ged_column": 3,
            "ground_truth_changed": False,
            "preference_definition_changed": False,
        })
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
