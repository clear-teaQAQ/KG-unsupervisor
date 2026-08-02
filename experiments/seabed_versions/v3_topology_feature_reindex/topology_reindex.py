"""Recover the node-feature order implied by KG triples and edge indices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReindexResult:
    node_ids: list[str]
    permutation: list[int]
    consistent_edges_before: int
    edge_count: int
    reassigned_nodes: int

    @property
    def changed(self):
        return self.reassigned_nodes > 0

    @property
    def fully_consistent_before(self):
        return self.consistent_edges_before == self.edge_count


def derive_topology_feature_order(graph_payload):
    node_features = graph_payload.get("node_features", [])
    edge_indices = graph_payload.get("edge_indices", [])
    edge_features = graph_payload.get("edge_features", [])
    triples = graph_payload.get("KG", [])
    if len(edge_indices) != len(edge_features) or len(edge_indices) != len(triples):
        raise ValueError(
            "Cannot reindex a graph whose edge_indices, edge_features, and KG lengths differ."
        )

    raw_node_ids = [str(node["id"]) for node in node_features]
    if len(set(raw_node_ids)) != len(raw_node_ids):
        raise ValueError("Node entity IDs must be unique within each graph.")
    raw_position = {entity_id: index for index, entity_id in enumerate(raw_node_ids)}
    topology_ids = [None] * len(raw_node_ids)
    consistent_edges = 0

    for edge_offset, (edge, edge_feature, triple) in enumerate(
        zip(edge_indices, edge_features, triples)
    ):
        if len(edge) != 2 or len(triple) != 3:
            raise ValueError(f"Malformed edge/triple at offset {edge_offset}.")
        source_index, target_index = (int(edge[0]), int(edge[1]))
        if min(source_index, target_index) < 0 or max(source_index, target_index) >= len(raw_node_ids):
            raise ValueError(f"Out-of-range edge index at offset {edge_offset}: {edge}.")
        source_id, predicate_id, target_id = map(str, triple)
        if str(edge_feature["id"]) != predicate_id:
            raise ValueError(
                f"Predicate mismatch at edge {edge_offset}: "
                f"edge_features={edge_feature['id']}, KG={predicate_id}."
            )

        for node_index, entity_id in (
            (source_index, source_id),
            (target_index, target_id),
        ):
            previous = topology_ids[node_index]
            if previous is not None and previous != entity_id:
                raise ValueError(
                    f"Conflicting entity assignments for node index {node_index}: "
                    f"{previous} vs {entity_id}."
                )
            topology_ids[node_index] = entity_id

        if raw_node_ids[source_index] == source_id and raw_node_ids[target_index] == target_id:
            consistent_edges += 1

    assigned = {entity_id for entity_id in topology_ids if entity_id is not None}
    if not assigned.issubset(raw_position):
        missing = sorted(assigned - raw_position)
        raise ValueError(f"KG endpoint entities are missing from node_features: {missing[:3]}.")

    remaining_ids = [entity_id for entity_id in raw_node_ids if entity_id not in assigned]
    remaining_iter = iter(remaining_ids)
    topology_ids = [next(remaining_iter) if entity_id is None else entity_id for entity_id in topology_ids]
    if set(topology_ids) != set(raw_node_ids) or len(set(topology_ids)) != len(topology_ids):
        raise ValueError("Reconstructed topology IDs are not a permutation of node_features IDs.")

    permutation = [raw_position[entity_id] for entity_id in topology_ids]
    reassigned_nodes = sum(index != raw_index for index, raw_index in enumerate(permutation))
    return ReindexResult(
        node_ids=topology_ids,
        permutation=permutation,
        consistent_edges_before=consistent_edges,
        edge_count=len(edge_indices),
        reassigned_nodes=reassigned_nodes,
    )


def reorder_features(features, permutation):
    if len(features) != len(permutation):
        raise ValueError("Feature and permutation lengths differ.")
    return [features[raw_index] for raw_index in permutation]
