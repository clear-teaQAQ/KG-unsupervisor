import unittest

from frozen_mapping_audit import aggregate_costs, magnitude_bucket, signed_error_bucket


class ErrorBucketTest(unittest.TestCase):
    def test_signed_buckets(self):
        self.assertEqual(signed_error_bucket(-3), "<=-2")
        self.assertEqual(signed_error_bucket(-1), "-1")
        self.assertEqual(signed_error_bucket(0), "0")
        self.assertEqual(signed_error_bucket(1), "+1")
        self.assertEqual(signed_error_bucket(4), ">=+2")

    def test_magnitude_buckets(self):
        self.assertEqual(magnitude_bucket(0), "0")
        self.assertEqual(magnitude_bucket(-1), "1")
        self.assertEqual(magnitude_bucket(2), "2")
        self.assertEqual(magnitude_bucket(-4), ">=3")

    def test_aggregate_costs(self):
        result = aggregate_costs([1, 3, 3, 7], [1, 2, 4, 5])
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["mse"], 1.5)
        self.assertEqual(result["acc"], 0.25)
        self.assertEqual(result["fea"], 0.75)
        self.assertEqual(
            result["signed_error_buckets"],
            {"+1": 1, ">=+2": 1, "-1": 1, "0": 1},
        )


if __name__ == "__main__":
    unittest.main()
