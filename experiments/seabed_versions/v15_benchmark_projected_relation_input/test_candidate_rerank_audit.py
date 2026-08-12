import unittest

from candidate_rerank_audit import aggregate_costs


class CandidateRerankMetricsTest(unittest.TestCase):
    def test_complete_metrics(self):
        metrics = aggregate_costs([3, 4, 5, 8], [3, 3, 5, 6])
        self.assertEqual(metrics["pairs"], 4)
        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["mae"], 0.75)
        self.assertEqual(metrics["mse"], 1.25)
        self.assertEqual(metrics["fea"], 1.0)


if __name__ == "__main__":
    unittest.main()
