"""Shared catalog, exclusion and explicit-rating baseline."""

import numpy as np
import pandas as pd


class BaseRecommender:
    supports_ratings = False

    def _setup(self, books, interactions):
        if books.empty:
            raise ValueError("Book catalog is empty")
        if books["Book-ID"].duplicated().any():
            raise ValueError("Book catalog must contain unique Book-ID values")
        if not interactions["Book-ID"].isin(books["Book-ID"]).all():
            raise ValueError("Train interactions contain books outside the catalog")
        # Giữ thứ tự catalog để vị trí vector điểm khớp với mã sách.
        self.books = books.set_index("Book-ID", drop=False).copy()
        self.items = self.books.index.to_numpy()
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.train = interactions.copy()
        # Ghi cả tương tác 0 để tránh gợi ý lại sách đã xem.
        self.seen = interactions.groupby("User-ID")["Book-ID"].agg(set).to_dict()
        # Chỉ rating 1–10 tham gia thống kê; 0 là tương tác ngầm.
        self.explicit = interactions.loc[interactions["Book-Rating"].gt(0)].copy()
        self.global_mean = float(self.explicit["Book-Rating"].mean()) if len(self.explicit) else 5.5
        self.item_means = self.explicit.groupby("Book-ID")["Book-Rating"].mean().to_dict()
        self.user_means = self.explicit.groupby("User-ID")["Book-Rating"].mean().to_dict()
        counts = self.explicit.groupby("Book-ID")["Book-Rating"].agg(["count", "mean"])
        self.rating_counts = counts["count"].to_dict()
        # Bayesian shrinkage: sách ít đánh giá được kéo về trung bình train.
        # Trọng số 10 kéo điểm sách ít rating về trung bình chung.
        self.popularity = np.array([
            (self.rating_counts.get(i, 0) * self.item_means.get(i, self.global_mean)
             + 10 * self.global_mean) / (self.rating_counts.get(i, 0) + 10)
            for i in self.items], dtype=float)
        self.trained_items = set(interactions["Book-ID"])
        return self

    def baseline(self, user_id, book_id):
        # Thiếu dữ liệu: dùng trung bình sách, rồi user, rồi toàn bộ train.
        return float(self.item_means.get(book_id, self.user_means.get(user_id, self.global_mean)))

    def _result(self, scores, n, exclude=()):
        if n < 1:
            raise ValueError("n must be positive")
        scores = np.asarray(scores, dtype=float).copy()
        if scores.shape != (len(self.items),):
            raise ValueError("Expected one score per catalog item")
        # Gán âm vô cùng để loại sách khỏi Top-N.
        for book in exclude:
            idx = self.item_index.get(book)
            if idx is not None:
                scores[idx] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        # Stable tie-breaking follows the shared catalog order.
        # Xếp điểm giảm dần; bằng điểm thì giữ thứ tự catalog.
        order = valid[np.argsort(-scores[valid], kind="stable")[:n]]
        result = self.books.iloc[order][["Book-ID", "Book-Title", "Book-Author"]].copy()
        result["rating_count"] = [self.rating_counts.get(i, 0) for i in result["Book-ID"]]
        result["score"] = scores[order]
        return result.reset_index(drop=True)

    def recommend(self, user_id=None, n=10, exclude_seen=True):
        scores = self.score_items(user_id)
        exclude = self.seen.get(user_id, set()) if exclude_seen else ()
        return self._result(scores, n, exclude)

    def predict(self, user_id, book_id):
        raise NotImplementedError("This model does not predict explicit ratings")
