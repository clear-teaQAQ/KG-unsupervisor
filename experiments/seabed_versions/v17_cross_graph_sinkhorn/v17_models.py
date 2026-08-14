"""V17: direct cross-graph relation-aware assignment generation."""

from pathlib import Path
import sys

import torch
from torch_geometric.nn import GINEConv, GraphNorm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CrossGraphSinkhornMatcher(torch.nn.Module):
    """Generate dense source-target assignment logits for each graph pair."""

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__()
        hidden_dims = list(getattr(args, "hidden_dim", [128, 64, 32]))
        self.hidden_dims = hidden_dims
        self.relation_dim = int(relation_dim)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer, hidden_dim in enumerate(hidden_dims):
            input_dim = number_of_labels if layer == 0 else hidden_dims[layer - 1]
            network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GINEConv(network, train_eps=True, edge_dim=self.relation_dim)
            )
            self.norms.append(GraphNorm(hidden_dim))

        embed_dim = hidden_dims[-1]
        attention_dim = max(embed_dim // 2, 16)
        self.query = torch.nn.Linear(embed_dim, attention_dim, bias=False)
        self.key = torch.nn.Linear(embed_dim, attention_dim, bias=False)
        self.pair_scorer = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, 1),
        )
        self.scale = attention_dim ** -0.5

    def _encode(self, data):
        features = data.x
        for conv, norm in zip(self.convs, self.norms):
            features = torch.relu(norm(conv(features, data.edge_index, data.edge_attr), data.batch))
        return features

    def forward(self, data):
        features = self._encode(data)
        pair_logits = []
        graph_batch = data.batch
        indicators = data.x_indicator.squeeze(-1).bool()
        pair_count = int(data.n.shape[0])
        for pair_index in range(pair_count):
            nodes = graph_batch == pair_index
            source = features[nodes & ~indicators]
            target = features[nodes & indicators]
            q = self.query(source)
            k = self.key(target)
            attention = (q @ k.transpose(0, 1)) * self.scale
            source_expanded = source[:, None, :].expand(-1, target.shape[0], -1)
            target_expanded = target[None, :, :].expand(source.shape[0], -1, -1)
            pair_features = torch.cat(
                [
                    source_expanded,
                    target_expanded,
                    torch.abs(source_expanded - target_expanded),
                    source_expanded * target_expanded,
                    attention.unsqueeze(-1),
                ],
                dim=-1,
            )
            pair_logits.append(self.pair_scorer(pair_features).reshape(-1, 1))
        if not pair_logits:
            return features.new_empty((0, 1))
        return torch.cat(pair_logits, dim=0)


def _sample_gumbel_like(values):
    uniform = torch.rand_like(values).clamp_(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(uniform))


def _sinkhorn_square(logits, iterations, valid_mask=None):
    """Return (optionally masked) doubly-stochastic matrices in log-space."""
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    log_prob = logits.masked_fill(~valid_mask, float("-inf"))
    valid_rows = valid_mask.any(dim=-1, keepdim=True)
    valid_cols = valid_mask.any(dim=-2, keepdim=True)
    for _ in range(iterations):
        row_lse = torch.logsumexp(log_prob, dim=-1, keepdim=True)
        row_lse = torch.where(valid_rows, row_lse, torch.zeros_like(row_lse))
        log_prob = torch.where(
            valid_mask,
            log_prob - row_lse,
            torch.full_like(log_prob, float("-inf")),
        )
        col_lse = torch.logsumexp(log_prob, dim=-2, keepdim=True)
        col_lse = torch.where(valid_cols, col_lse, torch.zeros_like(col_lse))
        log_prob = torch.where(
            valid_mask,
            log_prob - col_lse,
            torch.full_like(log_prob, float("-inf")),
        )
    return log_prob.exp()


def _greedy_rectangular(scores, source_nodes):
    """Extract injective source-to-target assignments from batched scores."""
    sample_count, _, target_nodes = scores.shape
    working = scores[:, :source_nodes].clone()
    solution = torch.zeros_like(working, dtype=torch.bool)
    blocked_rows = torch.zeros_like(working, dtype=torch.bool)
    blocked_cols = torch.zeros_like(working, dtype=torch.bool)
    sample_index = torch.arange(sample_count, device=scores.device)

    for _ in range(source_nodes):
        masked = working.masked_fill(blocked_rows | blocked_cols, float("-inf"))
        selected = masked.reshape(sample_count, -1).argmax(dim=-1)
        rows = selected // target_nodes
        cols = selected % target_nodes
        solution[sample_index, rows, cols] = True
        blocked_rows[sample_index, rows, :] = True
        blocked_cols[sample_index, :, cols] = True
    return solution


def _batched_greedy_rectangular(scores, source_nodes, target_nodes):
    """Extract injective assignments for a padded variable-size batch."""
    batch_size, max_nodes, _ = scores.shape
    row_ids = torch.arange(max_nodes, device=scores.device).view(1, max_nodes, 1)
    col_ids = torch.arange(max_nodes, device=scores.device).view(1, 1, max_nodes)
    valid = (row_ids < source_nodes.view(-1, 1, 1)) & (
        col_ids < target_nodes.view(-1, 1, 1)
    )
    blocked = ~valid
    solution = torch.zeros_like(scores, dtype=torch.bool)
    batch_ids = torch.arange(batch_size, device=scores.device)
    max_source_nodes = int(source_nodes.max().item())

    for step in range(max_source_nodes):
        selected = scores.masked_fill(blocked, float("-inf")).flatten(1).argmax(dim=1)
        rows = selected // max_nodes
        cols = selected % max_nodes
        active = source_nodes > step
        active_batches = batch_ids[active]
        active_rows = rows[active]
        active_cols = cols[active]
        solution[active_batches, active_rows, active_cols] = True
        blocked[active_batches, active_rows, :] = True
        blocked[active_batches, :, active_cols] = True
    return solution


def direct_sinkhorn_candidates(
    pair_logits,
    source_nodes,
    target_nodes,
    sample_count,
    tau,
    iterations,
    stochastic=True,
    include_deterministic=False,
):
    """Sample legal rectangular assignments directly from pair logits.

    Rectangular matrices are padded with dummy source rows before Sinkhorn.
    The dummy rows represent unmatched target nodes (node insertions), while
    every real source row is assigned to one unique target column.
    """
    if source_nodes > target_nodes:
        raise ValueError("V17 expects graph pairs ordered with n1 <= n2.")
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")

    pair_matrix = pair_logits.reshape(source_nodes, target_nodes)
    square_logits = pair_matrix.new_zeros((sample_count, target_nodes, target_nodes))
    square_logits[:, :source_nodes] = pair_matrix.unsqueeze(0)

    if stochastic:
        noise = _sample_gumbel_like(square_logits)
        if include_deterministic:
            noise[0].zero_()
        square_logits = square_logits + noise

    soft_square = _sinkhorn_square(square_logits / tau, iterations)
    soft_real = soft_square[:, :source_nodes]
    hard_real = _greedy_rectangular(soft_real, source_nodes)
    return hard_real, soft_real


def direct_sinkhorn_rollout(pair_logits, batch, tau, iterations, stochastic=True):
    """Vectorized training rollout for a variable-size batch of graph pairs."""
    source_nodes = batch.n[:, 0].long()
    target_nodes = batch.n[:, 1].long()
    mapping_batch = batch.batch[batch.edge_index_mapping[0]]
    pair_node_counts = source_nodes + target_nodes
    node_offsets = torch.cumsum(pair_node_counts, dim=0) - pair_node_counts
    local_rows = batch.edge_index_mapping[0] - node_offsets[mapping_batch]
    local_cols = (
        batch.edge_index_mapping[1]
        - node_offsets[mapping_batch]
        - source_nodes[mapping_batch]
    )

    batch_size = int(batch.n.shape[0])
    max_target_nodes = int(target_nodes.max().item())
    square_logits = pair_logits.new_zeros(
        (batch_size, max_target_nodes, max_target_nodes)
    )
    square_logits[mapping_batch, local_rows, local_cols] = pair_logits.squeeze(-1)

    node_ids = torch.arange(max_target_nodes, device=pair_logits.device)
    valid_nodes = node_ids.unsqueeze(0) < target_nodes.unsqueeze(1)
    valid_square = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
    if stochastic:
        square_logits = square_logits + _sample_gumbel_like(square_logits)
    noisy_logits = square_logits / tau
    soft_square = _sinkhorn_square(noisy_logits, iterations, valid_square)
    hard_square = _batched_greedy_rectangular(
        soft_square, source_nodes, target_nodes
    )

    hard_sparse = hard_square[mapping_batch, local_rows, local_cols]
    soft_sparse = soft_square[mapping_batch, local_rows, local_cols]
    return (
        hard_sparse.unsqueeze(-1).to(pair_logits.dtype),
        soft_sparse.unsqueeze(-1),
    )
