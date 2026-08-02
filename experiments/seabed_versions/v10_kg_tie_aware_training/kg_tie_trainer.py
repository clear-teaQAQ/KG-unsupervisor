"""V10 trainer: teach the generator GED-optimal, identity-aware pseudo-labels."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V4_DIR = CURRENT_DIR.parent / "v4_corrected_training"
for path in (PROJECT_ROOT, V4_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from corrected_training_trainer import CorrectedTrainingTrainer
from kg_objective import (
    exact_anchor_mask,
    lexicographic_update_masks,
    selected_anchor_counts,
)
from src.GEDRanker.loss_fn import (
    bpr_loss,
    hinge_loss,
    mapping_loss,
    roll_out,
    roll_out_gumbel,
)


class KGTieAwareTrainer(CorrectedTrainingTrainer):
    version = "v10_kg_tie_aware_training"
    objective_revision = "ged_primary_exact_anchor_tie_v1"

    def __init__(self, args):
        self.training_diagnostics = []
        self._epoch_update_stats = None
        self._raw_anchor_records = []
        super().__init__(args)

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)
        data.exact_anchor_mask = exact_anchor_mask(
            data.x,
            data.edge_index_mapping,
        ).to(torch.float)
        data.available_anchor_count = data.exact_anchor_mask.sum().reshape(1)
        return data

    def _anchor_counts(self, solution_sparse, batch):
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        batch_size = int(torch.max(mapping_batch).item()) + 1
        return selected_anchor_counts(
            solution_sparse,
            batch.exact_anchor_mask,
            mapping_batch,
            batch_size,
        )

    def init_roll_out(self):
        print("start initial roll out")
        for batch, indices in self.training_data_loader:
            batch.to(self.device)
            _, pred_solution = roll_out(batch.best_mapping_label, batch)
            pred_ged = self._compute_batch_ged(pred_solution, batch)
            pred_anchors = self._anchor_counts(pred_solution, batch)
            mapping_batch = batch.batch[batch.edge_index_mapping[0]]
            for index in range(len(indices)):
                graph = self.training_graphs[indices[index]]
                graph.best_ged = pred_ged[index]
                graph.best_mapping_label = pred_solution[
                    mapping_batch == index
                ].to(graph.best_mapping_label.device)
                graph.best_anchor_count = pred_anchors[index].reshape(1).to(
                    graph.best_mapping_label.device
                )
                graph.last_mapping_label = graph.best_mapping_label.clone()
                graph.last_ged = pred_ged[index]
        print("roll out finished")

    def process_batch(self, batch, indices):
        batch_size = int((torch.max(batch.batch) + 1).item())
        best_mapping_label, best_ged = batch.best_mapping_label, batch.best_ged
        best_anchors = batch.best_anchor_count.reshape(-1)
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        best_mapping_onehot = F.one_hot(
            best_mapping_label.long(),
            num_classes=2,
        ).float()
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        diffused_mapping = self.diffusion.sample(
            best_mapping_onehot,
            t,
            mapping_batch,
        )
        t = torch.from_numpy(t).float()
        pred_mapping_label = self.model(
            batch,
            diffused_mapping.to(self.device),
            t.to(self.device),
        )

        if self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            _, pred_solution, pred_solution_gumbel_sparse = roll_out_gumbel(
                pred_mapping_label,
                batch,
                self.args.tau,
                self.args.gumbel_iteration,
            )
            pred_ged = self._compute_batch_ged(pred_solution, batch)
        else:
            _, pred_solution = roll_out(pred_mapping_label, batch)
            pred_ged = self._compute_batch_ged(pred_solution, batch)

        pred_anchors = self._anchor_counts(pred_solution, batch)
        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_n.squeeze(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_n.squeeze(-1))
        normalize_last_ged = torch.exp(-batch.last_ged / batch.avg_n.squeeze(-1))
        alpha = (
            0
            if self.args.unsupervised_approach == "plain"
            else max(
                1 - self.cur_epoch / (self.args.model_epoch_end / 2),
                0,
            )
        )

        if alpha > 0 and self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            d_pred_curr = self.D(batch, pred_solution_gumbel_sparse.detach())
            if self.args.unsupervised_approach in ["BPR", "Hinge"]:
                d_pred_best = self.D(batch, batch.best_mapping_label)
                d_pred_last = self.D(batch, batch.last_mapping_label)

            if self.args.unsupervised_approach == "BPR":
                d_loss = bpr_loss(
                    d_pred_curr,
                    d_pred_best,
                    d_pred_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif self.args.unsupervised_approach == "Hinge":
                d_loss = hinge_loss(
                    d_pred_curr,
                    d_pred_best,
                    d_pred_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            else:
                d_loss = ((d_pred_curr - normalize_curr_ged) ** 2).sum()
            self.optimizerD.zero_grad()
            d_loss.backward()
            self.optimizerD.step()
        else:
            d_loss = torch.tensor([0], device=self.device)

        map_loss = mapping_loss(
            pred_mapping_label,
            batch,
            best_mapping_label,
        )
        if self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            d_pred_ged = self.D(batch, pred_solution_gumbel_sparse)
            ged_loss = -d_pred_ged.sum()
        else:
            ged_loss = torch.tensor([0], device=map_loss.device)

        losses = map_loss + ged_loss * alpha
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        strict_mask, semantic_tie_mask = lexicographic_update_masks(
            pred_ged,
            best_ged,
            pred_anchors,
            best_anchors,
        )
        update_mask = strict_mask | semantic_tie_mask
        for index in range(len(indices)):
            graph = self.training_graphs[indices[index]]
            mask = mapping_batch == index
            graph.last_mapping_label = pred_solution[mask].to(
                graph.last_mapping_label.device
            )
            graph.last_ged = pred_ged[index].to(graph.last_ged.device)
            if update_mask[index]:
                graph.best_ged = pred_ged[index].to(graph.best_ged.device)
                graph.best_mapping_label = pred_solution[mask].to(
                    graph.best_mapping_label.device
                )
                graph.best_anchor_count = pred_anchors[index].reshape(1).to(
                    graph.best_anchor_count.device
                )

        if self._epoch_update_stats is not None:
            self._epoch_update_stats["candidates"] += batch_size
            self._epoch_update_stats["candidate_anchors"] += float(
                pred_anchors.sum().item()
            )
            self._epoch_update_stats["strict_ged_updates"] += int(
                strict_mask.sum().item()
            )
            self._epoch_update_stats["equal_ged_anchor_updates"] += int(
                semantic_tie_mask.sum().item()
            )

        return (
            losses.item(),
            d_loss.item(),
            pred_ged.sum().item(),
            batch.ged.sum().item(),
            best_ged.sum().item(),
            map_loss.item(),
            ged_loss.item(),
            int(update_mask.sum().item()),
        )

    def fit(self):
        self._epoch_update_stats = {
            "candidates": 0,
            "candidate_anchors": 0.0,
            "strict_ged_updates": 0,
            "equal_ged_anchor_updates": 0,
        }
        super().fit()
        best_anchor_total = sum(
            float(graph.best_anchor_count.item())
            for graph in self.training_graphs
        )
        available_anchor_total = sum(
            float(graph.available_anchor_count.item())
            for graph in self.training_graphs
        )
        stats = {
            "epoch": self.cur_epoch + 1,
            **self._epoch_update_stats,
            "best_anchor_total": best_anchor_total,
            "available_anchor_total": available_anchor_total,
            "best_anchor_recall": (
                best_anchor_total / available_anchor_total
                if available_anchor_total
                else None
            ),
        }
        self.training_diagnostics.append(stats)
        print("V10 KG training diagnostics:", json.dumps(stats, sort_keys=True))
        diagnostics_path = self.result_dir / (
            f"training_diagnostics_{self.args.dataset}_{self.run_timestamp}.json"
        )
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": self.version,
                    "objective_revision": self.objective_revision,
                    "epochs": self.training_diagnostics,
                },
                handle,
                indent=2,
            )

    def diffusion_ged_parallel(self, batch, test_k=100):
        ged, solution, running_time = super().diffusion_ged_parallel(batch, test_k)
        n1 = int(batch.n[0, 0].item())
        n2 = int(batch.n[0, 1].item())
        anchor_matrix = batch.exact_anchor_mask.reshape(n1, n2).bool()
        selected = int((solution[:n1].bool() & anchor_matrix).sum().item())
        available = int(batch.available_anchor_count.item())
        self._raw_anchor_records.append(
            {
                "selected": selected,
                "available": available,
                "mapped_nodes": n1,
            }
        )
        return ged, solution, running_time

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if top_k_approach != "parallel":
            raise ValueError("V10 raw correspondence metrics require parallel inference.")
        self._raw_anchor_records = []
        result = super().score(testing_graph_set, test_k, top_k_approach)
        selected = sum(record["selected"] for record in self._raw_anchor_records)
        available = sum(record["available"] for record in self._raw_anchor_records)
        mapped_nodes = sum(record["mapped_nodes"] for record in self._raw_anchor_records)
        covered = [record for record in self._raw_anchor_records if record["available"]]
        perfect = sum(
            record["selected"] == record["available"]
            for record in covered
        )
        raw_metrics = {
            "selection": "first minimum-GED sample from unchanged best-of-k inference",
            "postprocessing": "none",
            "test_k": test_k,
            "pairs": len(self._raw_anchor_records),
            "pairs_with_exact_anchors": len(covered),
            "pairs_with_perfect_anchor_recall": perfect,
            "selected_exact_anchors": selected,
            "available_exact_anchors": available,
            "exact_anchor_recall": selected / available if available else None,
            "exact_anchor_rate_over_mapped_nodes": (
                selected / mapped_nodes if mapped_nodes else None
            ),
        }

        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(
            {
                "version": self.version,
                "objective_revision": self.objective_revision,
                "loaded_checkpoint_path": getattr(
                    self,
                    "loaded_checkpoint_path",
                    None,
                ),
                "raw_correspondence": raw_metrics,
                "training_diagnostics": self.training_diagnostics,
            }
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("V10 raw correspondence:", json.dumps(raw_metrics, sort_keys=True))
        return payload

    def load_explicit_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.loaded_checkpoint_path = str(Path(checkpoint_path).resolve())
        print("Loaded checkpoint:", self.loaded_checkpoint_path)
