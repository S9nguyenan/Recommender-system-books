"""Use only source CSV records; never replace experiments with invented data."""

import importlib.util
import unittest
import numpy as np

from experiments.runner import create_model, prepare
from src.utils.config import DATA_DIR, ExperimentConfig


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig(max_users=60, max_items=400, eval_users=3,
                                      eval_ratings=20, epochs=2, factors=8)
        cls.books, cls.train, cls.test, _, cls.users, _ = prepare(cls.config, DATA_DIR)

    def test_models_exclude_train_history_and_keep_train_only(self):
        for name in ["popularity", "content_based", "user_cf", "item_cf", "svd"]:
            with self.subTest(model=name):
                model = create_model(name, self.config).fit(self.books, self.train)
                user = self.users[0]
                recs = model.recommend(user, 10)
                self.assertTrue(set(recs["Book-ID"]).isdisjoint(model.seen[user]))
                self.assertFalse(recs["Book-ID"].duplicated().any())
                self.assertTrue(np.isfinite(recs["score"]).all())
                self.assertEqual(len(model.train), len(self.train))
                self.assertTrue(model.recommend(-1, 10)["Book-ID"].isin(self.books["Book-ID"]).all())
                if model.supports_ratings:
                    book = self.test["Book-ID"].iloc[0]
                    self.assertAlmostEqual(model.predict(user, book), model.score_items(user)[model.item_index[book]])
                    self.assertTrue(1 <= model.predict(-1, "unknown-book") <= 10)

    @unittest.skipUnless(importlib.util.find_spec("lightfm") is not None,
                         "LightFM is not installed; no training result is claimed")
    def test_lightfm_on_real_data(self):
        model = create_model("lightfm", self.config).fit(self.books, self.train)
        user = self.users[0]
        recs = model.recommend(user, 10)
        self.assertTrue(set(recs["Book-ID"]).isdisjoint(model.seen[user]))
        self.assertTrue(np.isfinite(recs["score"]).all())

    def test_content_lookup_and_unknown_book(self):
        model = create_model("content_based", self.config).fit(self.books, self.train)
        book = self.books["Book-ID"].iloc[0]
        self.assertNotIn(book, model.similar_books(book)["Book-ID"].tolist())
        with self.assertRaises(ValueError):
            model.similar_books("unknown-book")

    def test_numpy_factorization_backend(self):
        from src.models.svd import SVDRecommender
        model = SVDRecommender(factors=8, epochs=2, backend="numpy").fit(self.books, self.train)
        user = self.users[0]
        scores = model.score_items(user)
        self.assertTrue(np.isfinite(scores).all())
        for book in self.books["Book-ID"].iloc[:5]:
            self.assertAlmostEqual(model.predict(user, book), scores[model.item_index[book]])

    def test_implicit_only_train_and_empty_catalog(self):
        implicit = self.train.loc[self.train["Book-Rating"].eq(0)]
        self.assertGreater(len(implicit), 0)
        for name in ["popularity", "content_based", "user_cf", "item_cf"]:
            with self.subTest(model=name):
                model = create_model(name, self.config).fit(self.books, implicit)
                self.assertTrue(np.isfinite(model.recommend(-1)["score"]).all())
        with self.assertRaisesRegex(ValueError, "empty"):
            create_model("popularity", self.config).fit(self.books.iloc[:0], implicit.iloc[:0])


if __name__ == "__main__":
    unittest.main()
