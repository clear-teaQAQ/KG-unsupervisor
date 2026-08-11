"""KG data checks and feature alignment local to V12."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReindexResult:
    node_ids: tuple[str, ...]
    permutation: tuple[int, ...]
    consistent_edges_before: int
    edge_count: int

    @property
    def reassigned_nodes(self):
        return sum(index != raw_index for index, raw_index in enumerate(self.permutation))


def derive_topology_feature_order(payload):
    nodes = payload.get("node_features", [])
    edges = payload.get("edge_indices", [])
    relations = payload.get("edge_features", [])
    triples = payload.get("KG", [])
    if not (len(edges) == len(relations) == len(triples)):
        raise ValueError("edge_indices, edge_features, and KG must have equal lengths.")

    raw_ids = [str(node["id"]) for node in nodes]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Node entity IDs must be unique within each graph.")
    raw_position = {entity_id: index for index, entity_id in enumerate(raw_ids)}
    topology_ids = [None] * len(raw_ids)
    consistent = 0
    for edge_index, relation, triple in zip(edges, relations, triples):
        source, target = map(int, edge_index)
        source_id, predicate_id, target_id = map(str, triple)
        if str(relation["id"]) != predicate_id:
            raise ValueError("Relation ID does not match the KG predicate ID.")
        for node_index, entity_id in ((source, source_id), (target, target_id)):
            if node_index < 0 or node_index >= len(raw_ids):
                raise ValueError("KG edge contains an out-of-range node index.")
            previous = topology_ids[node_index]
            if previous is not None and previous != entity_id:
                raise ValueError("One topology node index maps to multiple entity IDs.")
            topology_ids[node_index] = entity_id
        consistent += int(raw_ids[source] == source_id and raw_ids[target] == target_id)

    assigned = {entity_id for entity_id in topology_ids if entity_id is not None}
    if not assigned.issubset(raw_position):
        raise ValueError("KG endpoints are missing from node_features.")
    remaining = iter(entity_id for entity_id in raw_ids if entity_id not in assigned)
    topology_ids = [next(remaining) if entity_id is None else entity_id for entity_id in topology_ids]
    if len(set(topology_ids)) != len(raw_ids):
        raise ValueError("Topology entity IDs are not a permutation of node_features IDs.")
    return ReindexResult(
        node_ids=tuple(topology_ids),
        permutation=tuple(raw_position[entity_id] for entity_id in topology_ids),
        consistent_edges_before=consistent,
        edge_count=len(edges),
    )


def reorder_features(features, permutation):
    if len(features) != len(permutation):
        raise ValueError("Feature and permutation lengths differ.")
    return [features[index] for index in permutation]
