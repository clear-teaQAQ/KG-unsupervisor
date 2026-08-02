import unittest

from path_evaluator import (
    audit_path,
    build_multirelation_edit_path,
    build_multirelation_graph,
    build_simple_edit_path,
    build_simple_graph,
    shared_entity_alignment,
    validate_mapping,
)


class PathEvaluatorTest(unittest.TestCase):
    def test_simple_graph_is_undirected_last_write_wins(self):
        graph = build_simple_graph(
            3,
            [[0, 1], [1, 0], [0, 1], [2, 2]],
            ["first", "second", "last", "loop"],
        )
        self.assertEqual(graph.edges, {(0, 1): "last", (2, 2): "loop"})

    def test_simple_path_replays_and_accounts_for_every_operation(self):
        source = build_simple_graph(2, [[0, 1], [0, 0]], ["old", "drop"])
        target = build_simple_graph(3, [[0, 1], [1, 2]], ["new", "insert"])
        path = build_simple_edit_path([0, 1], source, target)
        self.assertEqual(path["cost_breakdown"]["node_insertions"], 1)
        self.assertEqual(path["cost_breakdown"]["edge_deletions"], 1)
        self.assertEqual(path["cost_breakdown"]["edge_insertions"], 1)
        self.assertEqual(path["cost_breakdown"]["relation_substitutions"], 1)
        self.assertEqual(path["total_cost"], 4)
        self.assertEqual(
            audit_path(path, source, target),
            {"mapping_valid": True, "cost_consistent": True, "replay_success": True},
        )

    def test_multirelation_path_preserves_duplicates_and_uses_substitution(self):
        source = build_multirelation_graph(
            2,
            [[0, 1], [1, 0], [0, 1]],
            ["keep", "old", "old"],
        )
        target = build_multirelation_graph(
            2,
            [[0, 1], [0, 1], [1, 0]],
            ["keep", "new", "old"],
        )
        path = build_multirelation_edit_path([1, 0], source, target)
        self.assertEqual(path["matched_edge_count"], 2)
        self.assertEqual(len(path["relation_substitutions"]), 1)
        self.assertEqual(path["total_cost"], 1)
        self.assertTrue(audit_path(path, source, target)["replay_success"])

    def test_multirelation_and_simple_costs_can_disagree(self):
        source_simple = build_simple_graph(2, [[0, 1], [0, 1]], ["a", "b"])
        target_simple = build_simple_graph(2, [[0, 1]], ["b"])
        source_multi = build_multirelation_graph(2, [[0, 1], [0, 1]], ["a", "b"])
        target_multi = build_multirelation_graph(2, [[0, 1]], ["b"])
        self.assertEqual(build_simple_edit_path([0, 1], source_simple, target_simple)["total_cost"], 0)
        self.assertEqual(build_multirelation_edit_path([0, 1], source_multi, target_multi)["total_cost"], 1)

    def test_invalid_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_mapping([0, 0], 2, 3)
        with self.assertRaises(ValueError):
            validate_mapping([0, 3], 2, 3)

    def test_shared_entity_alignment_is_a_proxy_over_unique_ids(self):
        result = shared_entity_alignment(
            [1, 0, 2],
            ["a", "b", "duplicate", "duplicate"],
            ["b", "a", "c", "duplicate", "duplicate"],
        )
        self.assertEqual(result, {"shared_entities": 2, "aligned_shared_entities": 2})


if __name__ == "__main__":
    unittest.main()
