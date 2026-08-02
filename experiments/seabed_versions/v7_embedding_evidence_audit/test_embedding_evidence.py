import unittest

from embedding_evidence import analyze_pair_embeddings, count_rate, distribution


class EmbeddingEvidenceTest(unittest.TestCase):
    def test_exact_shared_vectors_are_strict_top1(self):
        result = analyze_pair_embeddings(
            ["a", "b"],
            [[1.0, 0.0], [0.0, 1.0]],
            ["b", "a", "c"],
            [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
        )
        self.assertEqual(result["shared_exact"], [True, True])
        self.assertEqual(result["correct_unique_top1"], [True, True])
        self.assertEqual(result["correct_strict_top1"], [True, True])
        self.assertEqual(result["correct_ranks"], [1, 1])

    def test_incorrect_exact_collision_breaks_unique_and_strict_top1(self):
        result = analyze_pair_embeddings(
            ["a"],
            [[1.0, 0.0]],
            ["a", "other"],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        self.assertEqual(result["correct_top1"], [True])
        self.assertEqual(result["correct_unique_top1"], [False])
        self.assertEqual(result["correct_strict_top1"], [False])
        self.assertEqual(result["incorrect_exact_collisions"], 1)

    def test_nonshared_source_is_recorded(self):
        result = analyze_pair_embeddings(
            ["a"], [[1.0, 0.0]], ["b"], [[0.0, 1.0]]
        )
        self.assertEqual(result["shared_entity_ids"], [])
        self.assertEqual(result["nonshared_max_cosines"], [0.0])

    def test_summary_helpers_handle_empty_input(self):
        self.assertEqual(count_rate([]), {"count": 0, "total": 0, "rate": 0.0})
        self.assertIsNone(distribution([])["mean"])


if __name__ == "__main__":
    unittest.main()

