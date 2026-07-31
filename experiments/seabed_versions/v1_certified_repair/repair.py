"""Exact local repair for the SEABED relation-labeled unit GED evaluator."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RepairResult:
    mapping: torch.Tensor
    initial_cost: int
    final_cost: int
    lower_bound: int
    iterations: int
    candidates_evaluated: int

    @property
    def certified(self):
        return self.final_cost == self.lower_bound


def deterministic_labeled_adjacency(n1, n2, edge_index, edge_labels, device):
    """Build V0-style dense adjacency with deterministic last-write-wins semantics."""
    edges = edge_index.detach().cpu()
    labels = edge_labels.detach().cpu()
    adj_1 = torch.zeros((n1, n1), dtype=torch.long)
    adj_2 = torch.zeros((n2, n2), dtype=torch.long)

    for edge_offset in range(edges.shape[1]):
        label = int(labels[edge_offset].item())
        if label <= 0:
            continue
        source = int(edges[0, edge_offset].item())
        target = int(edges[1, edge_offset].item())
        if source < n1:
            adj_1[source, target] = label
        else:
            adj_2[source - n1, target - n1] = label

    return adj_1.to(device), adj_2.to(device)


def permutation_unit_costs(permutations, adj_1, adj_2, candidate_batch_size=2048):
    """Vectorized exact V0 costs for complete target permutations."""
    n1 = adj_1.shape[0]
    n2 = adj_2.shape[0]
    if permutations.ndim != 2 or permutations.shape[1] != n2:
        raise ValueError("Expected a [num_candidates, n2] permutation tensor.")

    padded_adj_1 = torch.zeros((n2, n2), dtype=adj_1.dtype, device=adj_1.device)
    padded_adj_1[:n1, :n1] = adj_1
    upper = torch.triu_indices(n2, n2, offset=1, device=adj_1.device)
    source_values = padded_adj_1[upper[0], upper[1]]
    costs = []
    for offset in range(0, permutations.shape[0], candidate_batch_size):
        chunk = permutations[offset : offset + candidate_batch_size]
        permuted_adj_2 = adj_2[chunk[:, :, None], chunk[:, None, :]]
        target_values = permuted_adj_2[:, upper[0], upper[1]]
        edge_costs = torch.count_nonzero(target_values != source_values.unsqueeze(0), dim=1)
        costs.append(n2 - n1 + edge_costs)
    return torch.cat(costs)


def permutation_unit_cost(permutation, adj_1, adj_2):
    """Evaluate V0's cost from the complete target-node permutation."""
    n1 = adj_1.shape[0]
    n2 = adj_2.shape[0]
    if permutation.numel() != n2 or torch.unique(permutation).numel() != n2:
        raise ValueError("The complete target permutation must contain every target node exactly once.")

    return int(permutation_unit_costs(permutation.unsqueeze(0), adj_1, adj_2)[0].item())


def unit_cost(mapping, adj_1, adj_2):
    """Evaluate V0's unit cost from an injective G1-to-G2 node mapping."""
    n1 = adj_1.shape[0]
    n2 = adj_2.shape[0]
    if mapping.numel() != n1 or torch.unique(mapping).numel() != n1:
        raise ValueError("The source-to-target mapping must be injective and contain one target per source node.")
    occupied = torch.zeros(n2, dtype=torch.bool, device=mapping.device)
    occupied[mapping] = True
    unmatched = torch.arange(n2, device=mapping.device)[~occupied]
    permutation = torch.cat([mapping, unmatched])
    return permutation_unit_cost(permutation, adj_1, adj_2)


def size_lower_bound(adj_1, adj_2):
    """Return a lower bound consistent with the exact adjacency evaluator."""
    n1 = adj_1.shape[0]
    n2 = adj_2.shape[0]
    upper_1 = torch.triu_indices(n1, n1, offset=1, device=adj_1.device)
    upper_2 = torch.triu_indices(n2, n2, offset=1, device=adj_2.device)
    edges_1 = torch.count_nonzero(adj_1[upper_1[0], upper_1[1]])
    edges_2 = torch.count_nonzero(adj_2[upper_2[0], upper_2[1]])
    return int((n2 - n1 + torch.abs(edges_2 - edges_1)).item())


def _candidate_moves(mapping, n2):
    n1 = mapping.numel()
    matched_swaps = torch.triu_indices(n1, n1, offset=1, device=mapping.device).t()

    occupied = torch.zeros(n2, dtype=torch.bool, device=mapping.device)
    occupied[mapping] = True
    unmatched = torch.arange(n2, device=mapping.device)[~occupied]
    if unmatched.numel() == 0:
        replacements = torch.empty((0, 2), dtype=torch.long, device=mapping.device)
    else:
        rows = torch.arange(n1, device=mapping.device).repeat_interleave(unmatched.numel())
        targets = unmatched.repeat(n1)
        replacements = torch.stack([rows, targets], dim=1)
    return matched_swaps, replacements


def _best_neighbor(mapping, adj_1, adj_2, candidate_batch_size):
    n1 = mapping.numel()
    n2 = adj_2.shape[0]
    matched_swaps, replacements = _candidate_moves(mapping, n2)
    move_count = matched_swaps.shape[0] + replacements.shape[0]
    if move_count == 0:
        return mapping, unit_cost(mapping, adj_1, adj_2), 0

    target_ids = torch.arange(n2, device=mapping.device)

    best_mapping = mapping
    best_cost = unit_cost(mapping, adj_1, adj_2)
    offset = 0
    while offset < move_count:
        end = min(offset + candidate_batch_size, move_count)
        candidates = mapping.unsqueeze(0).repeat(end - offset, 1)
        candidate_ids = torch.arange(offset, end, device=mapping.device)

        swap_mask = candidate_ids < matched_swaps.shape[0]
        if swap_mask.any():
            candidate_rows = torch.nonzero(swap_mask, as_tuple=False).squeeze(1)
            swap_moves = matched_swaps[candidate_ids[swap_mask]]
            left_values = candidates[candidate_rows, swap_moves[:, 0]].clone()
            candidates[candidate_rows, swap_moves[:, 0]] = candidates[candidate_rows, swap_moves[:, 1]]
            candidates[candidate_rows, swap_moves[:, 1]] = left_values

        replacement_mask = ~swap_mask
        if replacement_mask.any():
            candidate_rows = torch.nonzero(replacement_mask, as_tuple=False).squeeze(1)
            replacement_ids = candidate_ids[replacement_mask] - matched_swaps.shape[0]
            replacement_moves = replacements[replacement_ids]
            candidates[candidate_rows, replacement_moves[:, 0]] = replacement_moves[:, 1]

        occupied = torch.zeros((candidates.shape[0], n2), dtype=torch.bool, device=mapping.device)
        occupied.scatter_(1, candidates, True)
        unmatched = target_ids.unsqueeze(0).expand(candidates.shape[0], -1)[~occupied]
        unmatched = unmatched.view(candidates.shape[0], n2 - n1)
        permutations = torch.cat([candidates, unmatched], dim=1)
        costs = permutation_unit_costs(
            permutations,
            adj_1,
            adj_2,
            candidate_batch_size=candidate_batch_size,
        )
        chunk_cost, chunk_index = torch.min(costs, dim=0)
        if int(chunk_cost.item()) < best_cost:
            best_cost = int(chunk_cost.item())
            best_mapping = candidates[int(chunk_index.item())].clone()
        offset = end

    return best_mapping, best_cost, move_count


def repair_mapping(mapping, adj_1, adj_2, max_iterations=20, candidate_batch_size=2048):
    """Run strict best-improvement hill climbing and stop at the exact bound."""
    mapping = mapping.long().clone()
    if torch.unique(mapping).numel() != mapping.numel():
        raise ValueError("The initial node mapping must be injective.")
    if mapping.numel() != adj_1.shape[0] or adj_2.shape[0] < adj_1.shape[0]:
        raise ValueError("Expected an n1-node source mapping into an n2-node target with n1 <= n2.")

    initial_cost = unit_cost(mapping, adj_1, adj_2)
    lower_bound = size_lower_bound(adj_1, adj_2)
    current_cost = initial_cost
    candidates_evaluated = 0
    iterations = 0

    while iterations < max_iterations and current_cost > lower_bound:
        candidate, candidate_cost, evaluated = _best_neighbor(
            mapping,
            adj_1,
            adj_2,
            candidate_batch_size,
        )
        candidates_evaluated += evaluated
        if candidate_cost >= current_cost:
            break
        mapping = candidate
        current_cost = candidate_cost
        iterations += 1

    return RepairResult(
        mapping=mapping,
        initial_cost=initial_cost,
        final_cost=current_cost,
        lower_bound=lower_bound,
        iterations=iterations,
        candidates_evaluated=candidates_evaluated,
    )
