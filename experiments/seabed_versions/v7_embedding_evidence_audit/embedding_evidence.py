"""Embedding identity and nearest-neighbor evidence for graph pairs."""

import numpy as np


def cosine_similarity_matrix(source_features, target_features):
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Feature matrices must be two-dimensional with equal widths.")
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    source = np.divide(source, source_norm, out=np.zeros_like(source), where=source_norm > 0)
    target = np.divide(target, target_norm, out=np.zeros_like(target), where=target_norm > 0)
    return source @ target.T


def distribution(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": round(float(np.min(values)), 6),
        "p05": round(float(np.percentile(values, 5)), 6),
        "median": round(float(np.median(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "max": round(float(np.max(values)), 6),
    }


def analyze_pair_embeddings(source_ids, source_features, target_ids, target_features, tolerance=1e-9):
    source_ids = [str(entity_id) for entity_id in source_ids]
    target_ids = [str(entity_id) for entity_id in target_ids]
    source_features = np.asarray(source_features, dtype=float)
    target_features = np.asarray(target_features, dtype=float)
    if len(source_ids) != len(source_features) or len(target_ids) != len(target_features):
        raise ValueError("Entity-ID and feature counts must agree.")

    similarities = cosine_similarity_matrix(source_features, target_features)
    target_positions = {}
    for index, entity_id in enumerate(target_ids):
        target_positions.setdefault(entity_id, []).append(index)

    evidence = {
        "shared_entity_ids": [],
        "shared_exact": [],
        "correct_cosines": [],
        "correct_top1": [],
        "correct_unique_top1": [],
        "correct_strict_top1": [],
        "correct_reciprocal_top1": [],
        "correct_ranks": [],
        "correct_margins": [],
        "max_incorrect_cosines": [],
        "nonshared_max_cosines": [],
        "incorrect_exact_collisions": 0,
    }

    for source_index, entity_id in enumerate(source_ids):
        scores = similarities[source_index]
        correct_indices = target_positions.get(entity_id, [])
        incorrect_indices = [
            index for index, target_id in enumerate(target_ids) if target_id != entity_id
        ]
        exact_incorrect = sum(
            np.array_equal(source_features[source_index], target_features[target_index])
            for target_index in incorrect_indices
        )
        evidence["incorrect_exact_collisions"] += int(exact_incorrect)

        if not correct_indices:
            if len(scores):
                evidence["nonshared_max_cosines"].append(float(np.max(scores)))
            continue

        correct_score = max(float(scores[index]) for index in correct_indices)
        best_score = float(np.max(scores))
        top_indices = np.flatnonzero(np.abs(scores - best_score) <= tolerance).tolist()
        best_incorrect = (
            max(float(scores[index]) for index in incorrect_indices)
            if incorrect_indices
            else -1.0
        )
        best_correct_index = min(
            index for index in correct_indices if abs(float(scores[index]) - correct_score) <= tolerance
        )
        reverse_scores = similarities[:, best_correct_index]

        evidence["shared_entity_ids"].append(entity_id)
        evidence["shared_exact"].append(
            any(
                np.array_equal(source_features[source_index], target_features[target_index])
                for target_index in correct_indices
            )
        )
        evidence["correct_cosines"].append(correct_score)
        evidence["correct_top1"].append(correct_score >= best_score - tolerance)
        evidence["correct_unique_top1"].append(
            len(top_indices) == 1 and top_indices[0] in correct_indices
        )
        evidence["correct_strict_top1"].append(
            not incorrect_indices or correct_score > best_incorrect + tolerance
        )
        evidence["correct_reciprocal_top1"].append(
            float(reverse_scores[source_index]) >= float(np.max(reverse_scores)) - tolerance
        )
        evidence["correct_ranks"].append(
            1 + int(np.count_nonzero(scores > correct_score + tolerance))
        )
        evidence["correct_margins"].append(correct_score - best_incorrect)
        if incorrect_indices:
            evidence["max_incorrect_cosines"].append(best_incorrect)

    return evidence


def count_rate(values):
    total = len(values)
    count = sum(bool(value) for value in values)
    return {"count": count, "total": total, "rate": round(count / total if total else 0.0, 6)}

