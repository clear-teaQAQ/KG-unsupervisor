"""Exact cross-graph embedding anchors."""

import numpy as np


def exact_embedding_matrix(source_features, target_features):
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Feature matrices must be two-dimensional with equal widths.")
    return np.all(source[:, None, :] == target[None, :, :], axis=2)


def mapping_anchor_count(mapping, anchor_matrix):
    return int(
        sum(bool(anchor_matrix[source, target]) for source, target in enumerate(mapping))
    )

