"""KG-aware secondary objective for GED-primary pseudo-label selection."""

import torch


def lexicographic_update_masks(
    candidate_ged,
    best_ged,
    candidate_anchors,
    best_anchors,
):
    """Return disjoint masks for strict-GED and equal-GED anchor updates."""
    strict_ged = candidate_ged < best_ged
    semantic_tie = (candidate_ged == best_ged) & (
        candidate_anchors > best_anchors
    )
    return strict_ged, semantic_tie


def exact_anchor_mask(node_features, mapping_edge_index):
    """Mark candidate cross-graph edges whose endpoint embeddings are identical."""
    source = node_features[mapping_edge_index[0]]
    target = node_features[mapping_edge_index[1]]
    return torch.all(source == target, dim=-1, keepdim=True)


def selected_anchor_counts(solution_sparse, anchor_mask, mapping_batch, batch_size):
    """Count exact anchors selected by each matching in a graph-pair batch."""
    selected = solution_sparse.squeeze(-1).to(torch.float) * anchor_mask.squeeze(-1)
    counts = torch.zeros(batch_size, dtype=torch.float, device=selected.device)
    counts.scatter_add_(0, mapping_batch, selected)
    return counts
