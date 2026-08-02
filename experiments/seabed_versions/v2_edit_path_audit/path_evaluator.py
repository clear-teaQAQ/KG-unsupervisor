"""Executable edit paths for simple-graph and multi-relation KG views."""

from collections import Counter
from dataclasses import dataclass


Endpoint = tuple[int, int]


def canonical_endpoint(source, target):
    return (source, target) if source <= target else (target, source)


def _validate_raw_graph(num_nodes, edge_indices, edge_labels):
    if len(edge_indices) != len(edge_labels):
        raise ValueError("edge_indices and edge_labels must have the same length.")
    for edge in edge_indices:
        if len(edge) != 2:
            raise ValueError(f"Expected a two-node edge, got {edge}.")
        if min(edge) < 0 or max(edge) >= num_nodes:
            raise ValueError(f"Edge {edge} is outside a {num_nodes}-node graph.")


@dataclass(frozen=True)
class SimpleGraphView:
    num_nodes: int
    edges: dict[Endpoint, str]


@dataclass(frozen=True)
class MultiRelationGraphView:
    num_nodes: int
    edges: dict[Endpoint, Counter]


def build_simple_graph(num_nodes, edge_indices, edge_labels):
    """Match nx.Graph.add_edge: undirected and last raw write wins."""
    _validate_raw_graph(num_nodes, edge_indices, edge_labels)
    edges = {}
    for (source, target), label in zip(edge_indices, edge_labels):
        edges[canonical_endpoint(int(source), int(target))] = str(label)
    return SimpleGraphView(num_nodes=num_nodes, edges=edges)


def build_multirelation_graph(num_nodes, edge_indices, edge_labels):
    """Preserve every predicate as an undirected endpoint-label multiset."""
    _validate_raw_graph(num_nodes, edge_indices, edge_labels)
    edges = {}
    for (source, target), label in zip(edge_indices, edge_labels):
        endpoint = canonical_endpoint(int(source), int(target))
        edges.setdefault(endpoint, Counter())[str(label)] += 1
    return MultiRelationGraphView(num_nodes=num_nodes, edges=edges)


def validate_mapping(mapping, source_nodes, target_nodes):
    values = [int(value) for value in mapping]
    if source_nodes > target_nodes:
        raise ValueError("V2 expects the packed GEDRanker order with source_nodes <= target_nodes.")
    if len(values) != source_nodes:
        raise ValueError(f"Expected {source_nodes} mapping entries, got {len(values)}.")
    if len(set(values)) != len(values):
        raise ValueError("The source-to-target mapping must be injective.")
    if values and (min(values) < 0 or max(values) >= target_nodes):
        raise ValueError("The source-to-target mapping contains an out-of-range target.")
    return values


def _mapped_endpoint(endpoint, mapping):
    return canonical_endpoint(mapping[endpoint[0]], mapping[endpoint[1]])


def _node_records(mapping, target_nodes, source_node_ids, target_node_ids):
    source_ids = source_node_ids or [None] * len(mapping)
    target_ids = target_node_ids or [None] * target_nodes
    correspondences = []
    for source, target in enumerate(mapping):
        record = {"source": source, "target": target, "cost": 0}
        if source_ids[source] is not None:
            record["source_id"] = source_ids[source]
        if target_ids[target] is not None:
            record["target_id"] = target_ids[target]
        correspondences.append(record)

    matched_targets = set(mapping)
    insertions = []
    for target in range(target_nodes):
        if target in matched_targets:
            continue
        record = {"target": target, "cost": 1}
        if target_ids[target] is not None:
            record["target_id"] = target_ids[target]
        insertions.append(record)
    return correspondences, insertions


def _base_path(representation, mapping, source_nodes, target_nodes, source_node_ids, target_node_ids):
    mapping = validate_mapping(mapping, source_nodes, target_nodes)
    correspondences, node_insertions = _node_records(
        mapping,
        target_nodes,
        source_node_ids,
        target_node_ids,
    )
    return mapping, {
        "representation": representation,
        "mapping": mapping,
        "node_correspondences": correspondences,
        "node_insertions": node_insertions,
        "node_deletions": [],
        "edge_deletions": [],
        "edge_insertions": [],
        "relation_substitutions": [],
        "matched_edge_count": 0,
    }


def build_simple_edit_path(
    mapping,
    source_graph,
    target_graph,
    source_node_ids=None,
    target_node_ids=None,
):
    mapping, path = _base_path(
        "undirected_simple_last_write",
        mapping,
        source_graph.num_nodes,
        target_graph.num_nodes,
        source_node_ids,
        target_node_ids,
    )
    mapped_source = {
        _mapped_endpoint(endpoint, mapping): relation
        for endpoint, relation in source_graph.edges.items()
    }

    for endpoint in sorted(set(mapped_source) | set(target_graph.edges)):
        source_relation = mapped_source.get(endpoint)
        target_relation = target_graph.edges.get(endpoint)
        if source_relation is None:
            path["edge_insertions"].append(
                {"endpoints": list(endpoint), "relation": target_relation, "cost": 1}
            )
        elif target_relation is None:
            path["edge_deletions"].append(
                {"endpoints": list(endpoint), "relation": source_relation, "cost": 1}
            )
        elif source_relation != target_relation:
            path["relation_substitutions"].append(
                {
                    "endpoints": list(endpoint),
                    "source_relation": source_relation,
                    "target_relation": target_relation,
                    "cost": 1,
                }
            )
        else:
            path["matched_edge_count"] += 1

    return _finalize_path(path)


def _expanded_relations(counter):
    return [relation for relation in sorted(counter) for _ in range(counter[relation])]


def build_multirelation_edit_path(
    mapping,
    source_graph,
    target_graph,
    source_node_ids=None,
    target_node_ids=None,
):
    mapping, path = _base_path(
        "undirected_predicate_multiset",
        mapping,
        source_graph.num_nodes,
        target_graph.num_nodes,
        source_node_ids,
        target_node_ids,
    )
    mapped_source = {
        _mapped_endpoint(endpoint, mapping): Counter(relations)
        for endpoint, relations in source_graph.edges.items()
    }

    for endpoint in sorted(set(mapped_source) | set(target_graph.edges)):
        source_relations = mapped_source.get(endpoint, Counter()).copy()
        target_relations = target_graph.edges.get(endpoint, Counter()).copy()
        common = source_relations & target_relations
        path["matched_edge_count"] += sum(common.values())
        source_relations -= common
        target_relations -= common

        source_left = _expanded_relations(source_relations)
        target_left = _expanded_relations(target_relations)
        substitutions = min(len(source_left), len(target_left))
        for index in range(substitutions):
            path["relation_substitutions"].append(
                {
                    "endpoints": list(endpoint),
                    "source_relation": source_left[index],
                    "target_relation": target_left[index],
                    "cost": 1,
                }
            )
        for relation in source_left[substitutions:]:
            path["edge_deletions"].append(
                {"endpoints": list(endpoint), "relation": relation, "cost": 1}
            )
        for relation in target_left[substitutions:]:
            path["edge_insertions"].append(
                {"endpoints": list(endpoint), "relation": relation, "cost": 1}
            )

    return _finalize_path(path)


def _finalize_path(path):
    breakdown = {
        "node_insertions": sum(operation["cost"] for operation in path["node_insertions"]),
        "node_deletions": sum(operation["cost"] for operation in path["node_deletions"]),
        "edge_insertions": sum(operation["cost"] for operation in path["edge_insertions"]),
        "edge_deletions": sum(operation["cost"] for operation in path["edge_deletions"]),
        "relation_substitutions": sum(
            operation["cost"] for operation in path["relation_substitutions"]
        ),
    }
    path["cost_breakdown"] = breakdown
    path["total_cost"] = sum(breakdown.values())
    return path


def _operation_cost(path):
    operation_groups = (
        "node_insertions",
        "node_deletions",
        "edge_insertions",
        "edge_deletions",
        "relation_substitutions",
    )
    return sum(operation["cost"] for group in operation_groups for operation in path[group])


def replay_simple_path(path, source_graph, target_graph):
    mapping = validate_mapping(path["mapping"], source_graph.num_nodes, target_graph.num_nodes)
    nodes = set(mapping)
    edges = {
        _mapped_endpoint(endpoint, mapping): relation
        for endpoint, relation in source_graph.edges.items()
    }
    for operation in path["node_insertions"]:
        nodes.add(operation["target"])
    for operation in path["edge_deletions"]:
        endpoint = tuple(operation["endpoints"])
        if edges.get(endpoint) != operation["relation"]:
            return False
        del edges[endpoint]
    for operation in path["relation_substitutions"]:
        endpoint = tuple(operation["endpoints"])
        if edges.get(endpoint) != operation["source_relation"]:
            return False
        edges[endpoint] = operation["target_relation"]
    for operation in path["edge_insertions"]:
        endpoint = tuple(operation["endpoints"])
        if endpoint in edges:
            return False
        edges[endpoint] = operation["relation"]
    return nodes == set(range(target_graph.num_nodes)) and edges == target_graph.edges


def _decrement(counter, relation):
    if counter[relation] <= 0:
        return False
    counter[relation] -= 1
    if counter[relation] == 0:
        del counter[relation]
    return True


def replay_multirelation_path(path, source_graph, target_graph):
    mapping = validate_mapping(path["mapping"], source_graph.num_nodes, target_graph.num_nodes)
    nodes = set(mapping)
    edges = {
        _mapped_endpoint(endpoint, mapping): Counter(relations)
        for endpoint, relations in source_graph.edges.items()
    }
    for operation in path["node_insertions"]:
        nodes.add(operation["target"])
    for operation in path["edge_deletions"]:
        endpoint = tuple(operation["endpoints"])
        if endpoint not in edges or not _decrement(edges[endpoint], operation["relation"]):
            return False
        if not edges[endpoint]:
            del edges[endpoint]
    for operation in path["relation_substitutions"]:
        endpoint = tuple(operation["endpoints"])
        if endpoint not in edges or not _decrement(edges[endpoint], operation["source_relation"]):
            return False
        edges.setdefault(endpoint, Counter())[operation["target_relation"]] += 1
    for operation in path["edge_insertions"]:
        endpoint = tuple(operation["endpoints"])
        edges.setdefault(endpoint, Counter())[operation["relation"]] += 1
    return nodes == set(range(target_graph.num_nodes)) and edges == target_graph.edges


def audit_path(path, source_graph, target_graph):
    try:
        validate_mapping(path["mapping"], source_graph.num_nodes, target_graph.num_nodes)
        mapping_valid = True
    except ValueError:
        mapping_valid = False
    cost_consistent = _operation_cost(path) == path["total_cost"] == sum(
        path["cost_breakdown"].values()
    )
    if not mapping_valid:
        replay_success = False
    elif path["representation"] == "undirected_simple_last_write":
        replay_success = replay_simple_path(path, source_graph, target_graph)
    elif path["representation"] == "undirected_predicate_multiset":
        replay_success = replay_multirelation_path(path, source_graph, target_graph)
    else:
        raise ValueError(f"Unknown path representation: {path['representation']}")
    return {
        "mapping_valid": mapping_valid,
        "cost_consistent": cost_consistent,
        "replay_success": replay_success,
    }


def shared_entity_alignment(mapping, source_node_ids, target_node_ids):
    source_positions = {}
    target_positions = {}
    for index, entity_id in enumerate(source_node_ids):
        source_positions.setdefault(entity_id, []).append(index)
    for index, entity_id in enumerate(target_node_ids):
        target_positions.setdefault(entity_id, []).append(index)
    shared = {
        entity_id
        for entity_id in source_positions.keys() & target_positions.keys()
        if len(source_positions[entity_id]) == 1 and len(target_positions[entity_id]) == 1
    }
    aligned = sum(
        int(mapping[source_positions[entity_id][0]] == target_positions[entity_id][0])
        for entity_id in shared
    )
    return {"shared_entities": len(shared), "aligned_shared_entities": aligned}
