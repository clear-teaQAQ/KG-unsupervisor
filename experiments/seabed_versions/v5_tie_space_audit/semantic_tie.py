"""Dual-cost-preserving local semantic repair utilities."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class TieRepairResult:
    mapping: tuple[int, ...]
    initial_score: float
    final_score: float
    iterations: int
    candidates_evaluated: int
    equal_cost_candidates: int
    initial_equal_cost_neighbors: int
    initial_improving_neighbors: int


def candidate_mappings(mapping, target_nodes):
    mapping = tuple(int(target) for target in mapping)
    for left, right in combinations(range(len(mapping)), 2):
        candidate = list(mapping)
        candidate[left], candidate[right] = candidate[right], candidate[left]
        yield tuple(candidate)

    unmatched = sorted(set(range(target_nodes)) - set(mapping))
    for source in range(len(mapping)):
        for target in unmatched:
            candidate = list(mapping)
            candidate[source] = target
            yield tuple(candidate)


def optimize_equal_cost_mapping(
    mapping,
    target_nodes,
    dual_cost,
    semantic_score,
    max_iterations=20,
    tolerance=1e-9,
):
    mapping = tuple(int(target) for target in mapping)
    fixed_cost = dual_cost(mapping)
    initial_score = float(semantic_score(mapping))
    current_score = initial_score
    iterations = 0
    candidates_evaluated = 0
    equal_cost_candidates = 0
    initial_equal_cost_neighbors = 0
    initial_improving_neighbors = 0

    while iterations < max_iterations:
        best_mapping = mapping
        best_score = current_score
        iteration_equal = 0
        iteration_improving = 0

        for candidate in candidate_mappings(mapping, target_nodes):
            candidates_evaluated += 1
            if dual_cost(candidate) != fixed_cost:
                continue
            equal_cost_candidates += 1
            iteration_equal += 1
            score = float(semantic_score(candidate))
            if score > current_score + tolerance:
                iteration_improving += 1
            if score > best_score + tolerance or (
                abs(score - best_score) <= tolerance and candidate < best_mapping
            ):
                best_mapping = candidate
                best_score = score

        if iterations == 0:
            initial_equal_cost_neighbors = iteration_equal
            initial_improving_neighbors = iteration_improving
        if best_score <= current_score + tolerance:
            break
        mapping = best_mapping
        current_score = best_score
        iterations += 1

    if dual_cost(mapping) != fixed_cost:
        raise RuntimeError("Semantic repair changed a protected graph cost.")
    return TieRepairResult(
        mapping=mapping,
        initial_score=initial_score,
        final_score=current_score,
        iterations=iterations,
        candidates_evaluated=candidates_evaluated,
        equal_cost_candidates=equal_cost_candidates,
        initial_equal_cost_neighbors=initial_equal_cost_neighbors,
        initial_improving_neighbors=initial_improving_neighbors,
    )


def shared_entity_alignment(mapping, source_ids, target_ids):
    target_id_set = set(target_ids)
    shared = sum(source_id in target_id_set for source_id in source_ids)
    aligned = sum(
        source_ids[source] in target_id_set
        and source_ids[source] == target_ids[target]
        for source, target in enumerate(mapping)
    )
    return aligned, shared


def cosine_similarity_matrix(source_features, target_features):
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature matrices must have matching dimensions.")
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    source = np.divide(source, source_norm, out=np.zeros_like(source), where=source_norm > 0)
    target = np.divide(target, target_norm, out=np.zeros_like(target), where=target_norm > 0)
    return source @ target.T


def matrix_mapping_score(mapping, score_matrix):
    return float(
        sum(score_matrix[source, target] for source, target in enumerate(mapping))
    )
