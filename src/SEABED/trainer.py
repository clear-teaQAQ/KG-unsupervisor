import json
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
GEDRANKER_DIR = CURRENT_DIR.parent / "GEDRanker"
if str(GEDRANKER_DIR) not in sys.path:
    sys.path.insert(1, str(GEDRANKER_DIR))

from diffusion_schedulers import CategoricalDiffusion, InferenceSchedule
from loss_fn import bpr_loss, hinge_loss, mapping_loss, roll_out, roll_out_gumbel
from models import DiffMatch, Discriminator

from utils import load_dataset


class DatasetWithIdx(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return len(self.data)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []
        self.founded_ged = []
        self.project_root = CURRENT_DIR.parent.parent.resolve()
        self.result_dir = (self.project_root / self.args.result_path).resolve()
        self.model_dir = (self.project_root / self.args.model_path).resolve()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu =", self.use_gpu)
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        self.load_data()
        self.transfer_data_to_torch()
        self.setup_model()
        self.init_graph_pairs()
        self.training_data_loader = DataLoader(DatasetWithIdx(self.training_graphs), batch_size=self.args.batch_size, shuffle=True)
        self.validation_data_loader = DataLoader(self.validation_graphs, batch_size=1, shuffle=False)
        self.testing_data_loader = DataLoader(self.testing_graphs, batch_size=1, shuffle=False)
        self.init_roll_out()

    def setup_model(self):
        self.model = DiffMatch(self.args, self.number_of_labels).to(self.device)
        self.D = Discriminator(self.args, self.number_of_labels).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizerD = torch.optim.RMSprop(
            self.D.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)

    def load_data(self):
        start = time.time()
        self.graphs, self.split_pairs, self.split_ranges = load_dataset(
            self.args.dataset_root,
            use_raw_features=bool(self.args.use_raw_features),
            ged_column=self.args.ged_column,
        )
        self.number_of_labels = len(self.graphs[0]["features"][0])
        print("Load {} graphs.".format(len(self.graphs)))
        print("Feature dimension:", self.number_of_labels)
        self.load_data_time = time.time() - start

    def transfer_data_to_torch(self):
        start = time.time()
        self.edge_index = []
        self.features = []
        self.edge_ids = [graph["edge_ids"] for graph in self.graphs]
        self.gid = [graph["gid"] for graph in self.graphs]
        self.gn = [graph["n"] for graph in self.graphs]
        self.gm = [graph["m"] for graph in self.graphs]

        for graph in self.graphs:
            edge = graph["graph"]
            edge = edge + [[dst, src] for src, dst in edge]
            edge = edge + [[node_id, node_id] for node_id in range(graph["n"])]
            self.edge_index.append(torch.tensor(edge).t().long())
            self.features.append(torch.tensor(graph["features"]).float())

        print("Feature shape of 1st graph:", self.features[0].shape)
        self.to_torch_time = time.time() - start

    @staticmethod
    def _full_mapping_sparse(n1, n2, offset):
        row_index = torch.arange(n1).repeat_interleave(n2)
        col_index = torch.arange(n2).repeat(n1) + offset
        edge_index_mapping = torch.stack([row_index, col_index], dim=0)
        edge_attr_mapping = torch.zeros((n1 * n2, 1), dtype=torch.float)
        return edge_index_mapping, edge_attr_mapping

    def pack_graph_pair(self, pair):
        id_1, id_2, real_ged = pair
        if self.gn[id_1] > self.gn[id_2]:
            id_1, id_2 = id_2, id_1

        n1, m1 = self.gn[id_1], self.gm[id_1]
        n2, m2 = self.gn[id_2], self.gm[id_2]

        new_data = Data()
        new_data.i_j = torch.tensor([[id_1, id_2]])
        new_data.n = torch.tensor([[n1, n2]])
        new_data.m = torch.tensor([[m1, m2]])
        new_data.avg_n = torch.tensor([[(n1 + n2) / 2]])
        new_data.higher_bound = torch.tensor([[max(n1, n2) + max(m1, m2)]])
        new_data.x = torch.cat([self.features[id_1], self.features[id_2]], dim=0)
        edge_vocab = {}
        next_label = 1
        for edge_id in self.edge_ids[id_1] + self.edge_ids[id_2]:
            if edge_id not in edge_vocab:
                edge_vocab[edge_id] = next_label
                next_label += 1

        edge_labels_1 = [edge_vocab[edge_id] for edge_id in self.edge_ids[id_1]]
        edge_labels_2 = [edge_vocab[edge_id] for edge_id in self.edge_ids[id_2]]
        edge_label_tensor_1 = torch.tensor(edge_labels_1 + edge_labels_1 + [0] * n1, dtype=torch.long)
        edge_label_tensor_2 = torch.tensor(edge_labels_2 + edge_labels_2 + [0] * n2, dtype=torch.long)
        new_data.edge_index = torch.cat([self.edge_index[id_1], self.edge_index[id_2] + n1], dim=1)
        new_data.edge_labels = torch.cat([edge_label_tensor_1, edge_label_tensor_2], dim=0)
        new_data.x_indicator = torch.cat([torch.zeros((n1, 1)), torch.ones((n2, 1))], dim=0)
        edge_index_mapping, edge_attr_mapping = self._full_mapping_sparse(n1, n2, n1)
        new_data.edge_index_mapping = edge_index_mapping
        new_data.edge_attr_mapping = edge_attr_mapping
        new_data.ged = torch.tensor([real_ged], dtype=torch.float)
        new_data.best_mapping_label = torch.rand_like(new_data.edge_attr_mapping)
        return new_data

    def _pair_labeled_adjacency(self, batch, batch_idx):
        n1 = int(batch.n[batch_idx, 0].item())
        n2 = int(batch.n[batch_idx, 1].item())
        node_offset = int(torch.sum(batch.n[:batch_idx], dim=1).sum().item()) if batch_idx > 0 else 0

        edge_batch = batch.batch[batch.edge_index[0]]
        pair_mask = edge_batch == batch_idx
        pair_edges = batch.edge_index[:, pair_mask]
        pair_labels = batch.edge_labels[pair_mask]
        pair_indicator = batch.x_indicator[pair_edges[0]].squeeze(1)

        adj_1 = torch.zeros((n1, n1), dtype=torch.long, device=batch.edge_index.device)
        adj_2 = torch.zeros((n2, n2), dtype=torch.long, device=batch.edge_index.device)

        edge_mask_1 = (pair_indicator == 0) & (pair_labels > 0)
        if edge_mask_1.any():
            local_edges_1 = pair_edges[:, edge_mask_1] - node_offset
            adj_1[local_edges_1[0], local_edges_1[1]] = pair_labels[edge_mask_1]

        edge_mask_2 = (pair_indicator == 1) & (pair_labels > 0)
        if edge_mask_2.any():
            local_edges_2 = pair_edges[:, edge_mask_2] - (node_offset + n1)
            adj_2[local_edges_2[0], local_edges_2[1]] = pair_labels[edge_mask_2]

        return adj_1, adj_2

    def _compute_batch_ged(self, solution_sparse, batch):
        batch_size = int(torch.max(batch.batch).item()) + 1
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        results = []

        for batch_idx in range(batch_size):
            n1 = int(batch.n[batch_idx, 0].item())
            n2 = int(batch.n[batch_idx, 1].item())
            pair_solution = solution_sparse[mapping_batch == batch_idx].squeeze(-1)
            pair_solution = pair_solution.view(n1, n2)
            mapped_cols = torch.argmax(pair_solution.float(), dim=1).tolist()
            unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
            permutation = torch.tensor(mapped_cols + unmatched_cols, dtype=torch.long, device=batch.edge_index.device)

            adj_1, adj_2 = self._pair_labeled_adjacency(batch, batch_idx)
            if self.args.cost_mode == "containment":
                mapped_adj_2 = adj_2.index_select(0, permutation[:n1]).index_select(1, permutation[:n1])
                overlap_mask = torch.triu(torch.ones((n1, n1), dtype=torch.bool, device=batch.edge_index.device), diagonal=1)
                overlap_edge_cost = torch.count_nonzero(adj_1[overlap_mask] != mapped_adj_2[overlap_mask]).float()
                node_cost = torch.tensor(float(n2 - n1), device=batch.edge_index.device)
                edge_cost = torch.tensor(float(torch.max(batch.m[batch_idx]).item() - torch.min(batch.m[batch_idx]).item()), device=batch.edge_index.device)
                results.append(node_cost + edge_cost + overlap_edge_cost)
            else:
                padded_adj_1 = torch.zeros((n2, n2), dtype=torch.long, device=batch.edge_index.device)
                padded_adj_1[:n1, :n1] = adj_1
                permuted_adj_2 = adj_2.index_select(0, permutation).index_select(1, permutation)
                upper_mask = torch.triu(torch.ones((n2, n2), dtype=torch.bool, device=batch.edge_index.device), diagonal=1)
                edge_cost = torch.count_nonzero(padded_adj_1[upper_mask] != permuted_adj_2[upper_mask]).float()
                node_cost = torch.tensor(float(n2 - n1), device=batch.edge_index.device)
                results.append(edge_cost + node_cost)

        return torch.stack(results)

    def _compute_single_ged_from_dense_solution(self, solution, data):
        n1 = int(data.n[0, 0].item())
        n2 = int(data.n[0, 1].item())
        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
        permutation = torch.tensor(mapped_cols + unmatched_cols, dtype=torch.long, device=solution.device)

        edge_mask_1 = (data.x_indicator[data.edge_index[0]].squeeze(1) == 0) & (data.edge_labels > 0)
        edge_mask_2 = (data.x_indicator[data.edge_index[0]].squeeze(1) == 1) & (data.edge_labels > 0)
        edge_index_1 = data.edge_index[:, edge_mask_1]
        edge_index_2 = data.edge_index[:, edge_mask_2] - n1
        edge_label_1 = data.edge_labels[edge_mask_1]
        edge_label_2 = data.edge_labels[edge_mask_2]

        adj_1 = torch.zeros((n1, n1), dtype=torch.long, device=solution.device)
        adj_2 = torch.zeros((n2, n2), dtype=torch.long, device=solution.device)
        if edge_index_1.numel() > 0:
            adj_1[edge_index_1[0], edge_index_1[1]] = edge_label_1
        if edge_index_2.numel() > 0:
            adj_2[edge_index_2[0], edge_index_2[1]] = edge_label_2

        if self.args.cost_mode == "containment":
            mapped_adj_2 = adj_2.index_select(0, permutation[:n1]).index_select(1, permutation[:n1])
            overlap_mask = torch.triu(torch.ones((n1, n1), dtype=torch.bool, device=solution.device), diagonal=1)
            overlap_edge_cost = torch.count_nonzero(adj_1[overlap_mask] != mapped_adj_2[overlap_mask]).item()
            edge_cost = abs(len(edge_label_2) // 2 - len(edge_label_1) // 2)
            return float((n2 - n1) + edge_cost + overlap_edge_cost)
        else:
            padded_adj_1 = torch.zeros((n2, n2), dtype=torch.long, device=solution.device)
            padded_adj_1[:n1, :n1] = adj_1
            permuted_adj_2 = adj_2.index_select(0, permutation).index_select(1, permutation)
            upper_mask = torch.triu(torch.ones((n2, n2), dtype=torch.bool, device=solution.device), diagonal=1)
            edge_cost = torch.count_nonzero(padded_adj_1[upper_mask] != permuted_adj_2[upper_mask]).item()
            return float((n2 - n1) + edge_cost)

    @staticmethod
    def _limit_pairs(pairs, limit):
        if limit and limit > 0:
            return pairs[:limit]
        return pairs

    def init_graph_pairs(self):
        start = time.time()
        train_pairs = self._limit_pairs(self.split_pairs["train"], self.args.max_train_pairs)
        val_pairs = self._limit_pairs(self.split_pairs["val"], self.args.max_val_pairs)
        test_pairs = self._limit_pairs(self.split_pairs["test"], self.args.max_test_pairs)
        self.training_graphs = [self.pack_graph_pair(pair) for pair in train_pairs]
        self.validation_graphs = [self.pack_graph_pair(pair) for pair in val_pairs]
        self.testing_graphs = [self.pack_graph_pair(pair) for pair in test_pairs]
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} validation graph pairs.".format(len(self.validation_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generation time:", time.time() - start)

    def init_roll_out(self):
        print("start initial roll out")
        for batch, indices in self.training_data_loader:
            batch.to(self.device)
            _, pred_solution = roll_out(batch.best_mapping_label, batch)
            pred_ged = self._compute_batch_ged(pred_solution, batch)
            for index in range(len(indices)):
                graph = self.training_graphs[indices[index]]
                graph.best_ged = pred_ged[index]
                graph.best_mapping_label = pred_solution[batch.batch[batch.edge_index_mapping[0]] == index].to(graph.best_mapping_label.device)
                graph.last_mapping_label = graph.best_mapping_label.clone()
                graph.last_ged = pred_ged[index]
        print("roll out finished")

    def _model_checkpoint_path(self, epoch):
        return self.model_dir / (
            f"{self.args.dataset}_{epoch}_{self.args.model_name}_{self.args.unsupervised_approach}_{self.run_timestamp}.pt"
        )

    def _legacy_model_checkpoint_path(self, epoch):
        return self.model_dir / f"{self.args.dataset}_{epoch}_{self.args.model_name}_{self.args.unsupervised_approach}.pt"

    def _result_file_path(self, stem):
        return self.result_dir / f"{stem}_{self.run_timestamp}.json"

    def _run_config(self):
        return {
            key: value
            for key, value in sorted(vars(self.args).items())
            if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
        }

    def _result_stem(self, prefix, testing_graph_set=None):
        parts = [
            prefix,
            self.args.dataset,
        ]
        if testing_graph_set is not None:
            parts.append(testing_graph_set)
        parts.extend(
            [
                self.args.unsupervised_approach,
                f"gedcol{self.args.ged_column}",
                self.args.cost_mode,
            ]
        )
        return "_".join(parts)

    def fit(self):
        print("\nModel training.\n")
        start = time.time()
        self.model.train()

        with tqdm(
            total=len(self.training_graphs),
            unit="pair",
            leave=True,
            dynamic_ncols=True,
            desc=f"Train epoch {self.cur_epoch + 1}",
            file=sys.stdout,
        ) as progress:
            g_loss_sum = 0
            d_loss_sum = 0
            map_loss_sum = 0
            ged_loss_sum = 0
            main_index = 0
            total_new_solution = 0
            total_pred_ged = 0
            total_gt_ged = 0
            total_curr_best_ged = 0

            for batch_index, (batch, idx) in enumerate(self.training_data_loader):
                batch.to(self.device)
                metrics = self.process_batch(batch, idx)
                batch_total_loss, batch_D_loss, rollout_ged, gt_ged, curr_best_ged, batch_map_loss, batch_ged_loss, new_solution = metrics

                total_curr_best_ged += curr_best_ged
                total_new_solution += new_solution
                total_pred_ged += rollout_ged
                total_gt_ged += gt_ged
                g_loss_sum += batch_total_loss
                d_loss_sum += batch_D_loss
                map_loss_sum += batch_map_loss
                ged_loss_sum += batch_ged_loss
                current_batch_size = int((torch.max(batch.batch) + 1).item())
                main_index += current_batch_size

                loss = g_loss_sum / main_index
                d_loss = d_loss_sum / main_index
                map_loss = map_loss_sum / main_index
                ged_loss = ged_loss_sum / main_index
                progress.update(current_batch_size)
                progress.set_postfix(
                    {
                        "batch": batch_index,
                        "g": round(1000 * loss, 2),
                        "d": round(1000 * d_loss, 2),
                        "map": round(1000 * map_loss, 2),
                        "ged": round(1000 * ged_loss, 2),
                        "new": new_solution,
                    },
                    refresh=True,
                )

        training_time = time.time() - start
        training_loss = round(1000 * loss, 3)
        training_d_loss = round(1000 * d_loss, 3)
        training_map_loss = round(1000 * map_loss, 3)
        training_ged_loss = round(1000 * ged_loss, 3)
        self.founded_ged.append(total_curr_best_ged)
        self.results.append(
            (
                "model_name",
                "dataset",
                "graph_set",
                "epoch",
                "training_time",
                "generator training_loss",
                "discriminator training_loss",
                "mapping training_loss",
                "ged training loss",
                "new solutions",
                "pred ged",
                "current best ged",
                "gt_ged",
            )
        )
        self.results.append(
            (
                self.args.model_name,
                self.args.dataset,
                "train",
                self.cur_epoch + 1,
                training_time,
                training_loss,
                training_d_loss,
                training_map_loss,
                training_ged_loss,
                total_new_solution,
                total_pred_ged,
                total_curr_best_ged,
                total_gt_ged,
            )
        )
        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")
        with open(
            self._result_file_path(self._result_stem("pathlength_SEABED")),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(self.founded_ged, handle)

    def process_batch(self, batch, indices):
        batch_size = (torch.max(batch.batch) + 1).item()
        best_mapping_label, best_ged = batch.best_mapping_label, batch.best_ged
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        best_mapping_onehot = F.one_hot(best_mapping_label.long(), num_classes=2).float()
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        diffused_mapping = self.diffusion.sample(best_mapping_onehot, t, mapping_batch)
        t = torch.from_numpy(t).float()
        pred_mapping_label = self.model(batch, diffused_mapping.to(self.device), t.to(self.device))

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

        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_n.squeeze(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_n.squeeze(-1))
        normalize_last_ged = torch.exp(-batch.last_ged / batch.avg_n.squeeze(-1))
        alpha = 0 if self.args.unsupervised_approach == "plain" else max(1 - self.cur_epoch / (self.args.model_epoch_end / 2), 0)

        if alpha > 0 and self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            D_pred_ged_curr = self.D(batch, pred_solution_gumbel_sparse.detach())
            if self.args.unsupervised_approach in ["BPR", "Hinge"]:
                D_pred_ged_best = self.D(batch, batch.best_mapping_label)
                D_pred_ged_last = self.D(batch, batch.last_mapping_label)

            if self.args.unsupervised_approach == "BPR":
                D_loss = bpr_loss(
                    D_pred_ged_curr,
                    D_pred_ged_best,
                    D_pred_ged_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif self.args.unsupervised_approach == "Hinge":
                D_loss = hinge_loss(
                    D_pred_ged_curr,
                    D_pred_ged_best,
                    D_pred_ged_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            else:
                D_loss = ((D_pred_ged_curr - normalize_curr_ged) ** 2).sum()
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()
        else:
            D_loss = torch.tensor([0], device=self.device)

        map_loss = mapping_loss(pred_mapping_label, batch, best_mapping_label)

        if self.args.unsupervised_approach in ["BPR", "Hinge", "GED"]:
            D_pred_ged = self.D(batch, pred_solution_gumbel_sparse)
            ged_loss = -(D_pred_ged).sum()
        else:
            ged_loss = torch.tensor([0], device=map_loss.device)

        losses = map_loss + ged_loss * alpha
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        new_solution = 0
        for index in range(len(indices)):
            graph = self.training_graphs[indices[index]]
            mask = batch.batch[batch.edge_index_mapping[0]] == index
            graph.last_mapping_label = pred_solution[mask].to(graph.last_mapping_label.device)
            graph.last_ged = pred_ged[index].to(graph.last_ged.device)
            if pred_ged[index] < best_ged[index]:
                new_solution += 1
                graph.best_ged = pred_ged[index].to(graph.best_ged.device)
                graph.best_mapping_label = pred_solution[mask].to(graph.best_mapping_label.device)

        return (
            losses.item(),
            D_loss.item(),
            pred_ged.sum().item(),
            batch.ged.sum().item(),
            best_ged.sum().item(),
            map_loss.item(),
            ged_loss.item(),
            new_solution,
        )

    def diffusion_ged_parallel(self, batch, test_k=100):
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
        new_batch = Batch().from_data_list([data for _ in range(num_parallel_sampling)])
        mapping_t = torch.randn_like(new_batch.edge_attr_mapping, device=self.device)
        mapping_t = (mapping_t > 0).long()
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        for step in range(steps):
            t1, t2 = time_schedule(step)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(new_batch, mapping_t, t1, t2)

        n1 = batch.n[0, 0].item()
        n2 = batch.n[0, 1].item()
        pred_matching_matrix = torch.zeros((num_parallel_sampling, n1, n2), device=self.device)
        mapping_edge_idx = new_batch.edge_index_mapping
        batch_mapping_edge_idx = mapping_edge_idx - new_batch.batch[mapping_edge_idx[0]] * (n1 + n2)
        batch_mapping_edge_idx[1] -= n1
        pred_matching_matrix[new_batch.batch[mapping_edge_idx[0]], batch_mapping_edge_idx[0], batch_mapping_edge_idx[1]] = mapping_t.squeeze(-1)

        batch_idx = torch.arange(num_parallel_sampling, device=pred_matching_matrix.device)
        greedy_mask = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
        solution = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)

        for _ in range(min(n1, n2)):
            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling, -1)
            argmax_result = torch.argmax(pred_matching_matrix, dim=-1)
            rows = argmax_result // n2
            columns = argmax_result % n2
            solution[batch_idx, rows, columns] = True
            greedy_mask[batch_idx, rows, :] = True
            greedy_mask[batch_idx, :, columns] = True
            pred_matching_matrix = pred_matching_matrix.view(num_parallel_sampling, n1, n2)
            pred_matching_matrix[greedy_mask] = float("-inf")

        zeros_column = torch.where(~torch.any(solution == 1, dim=1))
        solution = torch.cat([solution, torch.zeros(num_parallel_sampling, n2 - n1, n2, device=solution.device)], dim=1)
        solution[zeros_column[0], torch.arange(n1, n2, device=solution.device).repeat(num_parallel_sampling), zeros_column[1]] = 1
        extracted_mapping = torch.nonzero(solution)

        ged_values = []
        for sample_idx in range(num_parallel_sampling):
            ged_values.append(self._compute_single_ged_from_dense_solution(solution[sample_idx], batch[0]))
        ged = torch.tensor(ged_values, device=self.device)
        min_ged_idx = torch.argmin(ged)
        min_ged = ged[min_ged_idx].item()
        return min_ged, solution[min_ged_idx, :n1], time.time() - start_time

    def diffusion_ged_sequential(self, batch, test_k=100):
        import dgl
        from gedgnn_kbest import KBestMSolver

        start_time = time.time()
        mapping_t = torch.randn_like(batch.edge_attr_mapping, device=self.device)
        mapping_t = (mapping_t > 0).long()
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)

        for step in range(steps):
            t1, t2 = time_schedule(step)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            mapping_t = self.categorical_denoise_step(batch, mapping_t, t1, t2)

        mapping_t = torch.softmax(mapping_t.squeeze(-1), dim=0).unsqueeze(-1)
        mapping_t = (mapping_t * 1e9 + 1).round()

        n1 = batch.n[0, 0].item()
        n2 = batch.n[0, 1].item()
        x1 = batch.x[:n1]
        x2 = batch.x[n1:]
        x1_edge = batch.edge_index[:, batch.edge_index[0] < n1]
        x2_edge = batch.edge_index[:, batch.edge_index[0] >= n1] - n1
        g1 = dgl.graph((x1_edge[0], x1_edge[1]), num_nodes=n1)
        g2 = dgl.graph((x2_edge[0], x2_edge[1]), num_nodes=n2)
        g1.ndata["f"] = x1
        g2.ndata["f"] = x2
        pred_matching_matrix = torch.zeros((n1, n2), device=self.device)
        pred_matching_matrix[batch.edge_index_mapping[0], batch.edge_index_mapping[1] - n1] = mapping_t.squeeze(-1)
        solver = KBestMSolver(pred_matching_matrix, g1, g2)
        solver.get_matching(test_k)
        return solver.min_ged, [], time.time() - start_time

    def categorical_denoise_step(self, data, mapping_t, t1, t2):
        batch_size = torch.max(data.batch) + 1
        t1 = torch.from_numpy(t1).repeat(batch_size)
        with torch.no_grad():
            pred_mapping_label = self.model(data, mapping_t, t1.float().to(self.device))
        prob_mapping = torch.sigmoid(pred_mapping_label)
        prob_mapping = torch.cat([1 - prob_mapping, prob_mapping], dim=-1)
        return self.categorical_posterior(t2, t1, prob_mapping, mapping_t, data.batch[data.edge_index_mapping[0]])

    def categorical_posterior(self, target_t, t, x0_pred_prob, xt, mapping_batch):
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
        target_t = target_t.repeat(t.shape[0])
        Q_t = (np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]).reshape(t.shape[0], 2, 2)
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device).reshape(t.shape[0], 2, 2)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device).reshape(t.shape[0], 2, 2)

        x0_pred_prob = x0_pred_prob.unsqueeze(1)
        xt = F.one_hot(xt.long(), num_classes=2).float()
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t[mapping_batch].permute((0, 2, 1)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[:, 0]
        x_t_target_prob_part_3 = (Q_bar_t_source[:, 0][mapping_batch].unsqueeze(1) * xt).sum(dim=-1, keepdim=True)
        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3
        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        x_t_target_prob_part_2_new = Q_bar_t_target[:, 1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[:, 1][mapping_batch].unsqueeze(1) * xt).sum(dim=-1, keepdim=True)
        x_t_target_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new[mapping_batch].unsqueeze(1)) / x_t_target_prob_part_3_new
        sum_x_t_target_prob += x_t_target_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t[0] > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
        return xt

    def save(self, epoch):
        torch.save(
            self.model.state_dict(),
            self._model_checkpoint_path(epoch),
        )

    def load(self, epoch):
        checkpoint_path = self._model_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            checkpoint_path = self._legacy_model_checkpoint_path(epoch)
        self.model.load_state_dict(torch.load(checkpoint_path))

    @staticmethod
    def cal_pk(num, pre, gt):
        if not pre:
            return 0.0
        num = min(num, len(pre))
        pred_order = sorted(range(len(pre)), key=lambda idx: pre[idx])[:num]
        gt_order = sorted(range(len(gt)), key=lambda idx: gt[idx])[:num]
        return len(set(pred_order) & set(gt_order)) / float(num)

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if testing_graph_set == "test":
            loader = self.testing_data_loader
        elif testing_graph_set == "val":
            loader = self.validation_data_loader
        else:
            loader = DataLoader(self.training_graphs, batch_size=1, shuffle=False)

        print("\n\nEvalute SEABED with {} topk {} on {} set.\n".format(top_k_approach, test_k, testing_graph_set))
        self.model.eval()
        num = 0
        time_usage = 0
        mse = []
        mae = []
        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk1 = []
        pk5 = []
        pk10 = []
        pk15 = []
        pk20 = []
        pres = {}
        gts = {}
        start_time = time.time()

        for batch in tqdm(
            loader,
            total=len(loader),
            unit="pair",
            leave=True,
            dynamic_ncols=True,
            desc=f"Eval {testing_graph_set}",
            file=sys.stdout,
        ):
            batch.to(self.device)
            gt = batch.ged.item()
            if top_k_approach == "parallel":
                pre_ged, _, running_time = self.diffusion_ged_parallel(batch, test_k)
            else:
                pre_ged, _, running_time = self.diffusion_ged_sequential(batch, test_k)
            round_pre_ged = round(pre_ged)

            num += 1
            time_usage += running_time
            source_idx = batch.i_j[0][0].item()
            pres.setdefault(source_idx, []).append(pre_ged)
            gts.setdefault(source_idx, []).append(gt)
            mse.append((pre_ged - gt) ** 2)
            mae.append(abs(pre_ged - gt))
            if round_pre_ged == gt:
                num_acc += 1
                num_fea += 1
            elif round_pre_ged > gt:
                num_fea += 1

        for graph_id in pres:
            rho.append(spearmanr(pres[graph_id], gts[graph_id])[0])
            tau.append(kendalltau(pres[graph_id], gts[graph_id])[0])
            pk1.append(self.cal_pk(1, pres[graph_id], gts[graph_id]))
            pk5.append(self.cal_pk(5, pres[graph_id], gts[graph_id]))
            pk10.append(self.cal_pk(10, pres[graph_id], gts[graph_id]))
            pk15.append(self.cal_pk(15, pres[graph_id], gts[graph_id]))
            pk20.append(self.cal_pk(20, pres[graph_id], gts[graph_id]))

        run_time = round((time.time() - start_time) / num, 5)
        time_usage = round(time_usage / num, 5)
        mse = round(np.mean(mse), 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(sum(rho) / len(rho), 3)
        tau = round(sum(tau) / len(tau), 3)
        pk1 = round(np.mean(pk1), 3)
        pk5 = round(np.mean(pk5), 3)
        pk10 = round(np.mean(pk10), 3)
        pk15 = round(np.mean(pk15), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            (
                "model_name",
                "topk_approach",
                "dataset",
                "graph_set",
                "#testing_pairs",
                "time_usage(s/p)",
                "run_time(s/p)",
                "mse",
                "mae",
                "acc",
                "fea",
                "rho",
                "tau",
                "pk1",
                "pk5",
                "pk10",
                "pk15",
                "pk20",
            )
        )
        self.results.append(
            (
                self.args.model_name,
                top_k_approach,
                self.args.dataset,
                testing_graph_set,
                num,
                time_usage,
                run_time,
                mse,
                mae,
                acc,
                fea,
                rho,
                tau,
                pk1,
                pk5,
                pk10,
                pk15,
                pk20,
            )
        )
        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")

        with open(
            self._result_file_path(self._result_stem("result_SEABED", testing_graph_set)),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "config": self._run_config(),
                    "mse": mse,
                    "time": time_usage,
                    "run_time": run_time,
                    "mae": mae,
                    "acc": acc,
                    "fea": fea,
                    "rho": rho,
                    "tau": tau,
                    "pk1": pk1,
                    "pk5": pk5,
                    "pk10": pk10,
                    "pk15": pk15,
                    "pk20": pk20,
                },
                handle,
            )
