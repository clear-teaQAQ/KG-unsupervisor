import unittest

from topology_reindex import derive_topology_feature_order, reorder_features


def payload(node_ids, edges, predicates, triples):
    return {
        "node_features": [{"id": node_id} for node_id in node_ids],
        "edge_indices": edges,
        "edge_features": [{"id": predicate} for predicate in predicates],
        "KG": triples,
    }


class TopologyReindexTest(unittest.TestCase):
    def test_consistent_graph_is_unchanged(self):
        graph = payload(
            ["a", "b", "c"],
            [[0, 1], [1, 2]],
            ["p", "q"],
            [["a", "p", "b"], ["b", "q", "c"]],
        )
        result = derive_topology_feature_order(graph)
        self.assertEqual(result.permutation, [0, 1, 2])
        self.assertTrue(result.fully_consistent_before)
        self.assertFalse(result.changed)

    def test_shuffled_features_are_reordered_to_topology(self):
        graph = payload(
            ["c", "a", "b"],
            [[0, 1], [1, 2]],
            ["p", "q"],
            [["a", "p", "b"], ["b", "q", "c"]],
        )
        result = derive_topology_feature_order(graph)
        self.assertEqual(result.node_ids, ["a", "b", "c"])
        self.assertEqual(result.permutation, [1, 2, 0])
        self.assertEqual(result.consistent_edges_before, 0)
        self.assertEqual(reorder_features(["C", "A", "B"], result.permutation), ["A", "B", "C"])

    def test_isolated_nodes_keep_stable_relative_order(self):
        graph = payload(
            ["isolated-1", "b", "a", "isolated-2"],
            [[0, 1]],
            ["p"],
            [["a", "p", "b"]],
        )
        result = derive_topology_feature_order(graph)
        self.assertEqual(result.node_ids, ["a", "b", "isolated-1", "isolated-2"])

    def test_conflicting_index_assignments_are_rejected(self):
        graph = payload(
            ["a", "b", "c"],
            [[0, 1], [0, 2]],
            ["p", "q"],
            [["a", "p", "b"], ["c", "q", "b"]],
        )
        with self.assertRaises(ValueError):
            derive_topology_feature_order(graph)

    def test_predicate_mismatch_is_rejected(self):
        graph = payload(["a", "b"], [[0, 1]], ["wrong"], [["a", "p", "b"]])
        with self.assertRaises(ValueError):
            derive_topology_feature_order(graph)


if __name__ == "__main__":
    unittest.main()
