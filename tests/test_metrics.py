import unittest
from src.evaluation.metrics import rating_metrics, precision_recall_at_k


class MetricTests(unittest.TestCase):
    def test_rating_metrics_exclude_zero_and_nonfinite(self):
        result = rating_metrics([1, 3, 0, 9], [2, 1, 100, float("nan")])
        self.assertEqual(result["rating_evaluated"], 2)
        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], (2.5 ** 0.5))

    def test_empty_metrics_are_not_zero(self):
        self.assertIsNone(rating_metrics([], [])["rmse"])
        self.assertEqual(precision_recall_at_k([], [], 10), (None, None))

    def test_ranking_short_list_and_duplicates(self):
        self.assertEqual(precision_recall_at_k(["a", "a", "b"], {"a", "c"}, 5), (0.2, 0.5))


if __name__ == "__main__":
    unittest.main()
