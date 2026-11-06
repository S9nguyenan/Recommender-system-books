"""LightFM logistic: positive/negative labels derived only from observed train."""

import os
import numpy as np
from scipy.sparse import coo_matrix
from src.models.base import BaseRecommender


class LightFMRecommender(BaseRecommender):
    def __init__(self, factors=32, epochs=15, random_seed=42, relevance_threshold=7,
                 loss="logistic"):
        self.factors, self.epochs = factors, epochs
        self.seed, self.threshold = random_seed, relevance_threshold
        self.loss = loss

    def fit(self, books, interactions):
        try:
            from lightfm import LightFM
        except ImportError as exc:
            raise ImportError("LightFM unavailable. Install requirements-optional.txt; see README.") from exc
        if os.name == "nt" and self.loss != "logistic":
            raise ValueError("Use LightFM logistic on Windows: ranking loss kernels crashed in this build")
        self._setup(books, interactions)
        positives = interactions.loc[interactions["Book-Rating"].eq(0)
                                     | interactions["Book-Rating"].ge(self.threshold)]
        if positives.empty:
            raise ValueError("LightFM requires positive train interactions")
        training = interactions if self.loss == "logistic" else positives
        self.users = training["User-ID"].unique()
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.positive_items = set(positives["Book-ID"])
        self.positive_users = set(positives["User-ID"])
        rows = training["User-ID"].map(self.user_index).to_numpy(dtype=np.int32)
        cols = training["Book-ID"].map(self.item_index).to_numpy(dtype=np.int32)
        # Nhãn logistic: tương tác 0 hoặc rating >= ngưỡng là +1; rating thấp là -1.
        # Không tạo rating giả hoặc dùng test để lấy negative. Ô chưa quan sát vẫn trống.
        labels = np.where(training["Book-Rating"].eq(0)
                          | training["Book-Rating"].ge(self.threshold), 1, -1).astype(np.float32)
        matrix = coo_matrix((labels, (rows, cols)),
                            shape=(len(self.users), len(self.items)))
        self.model = LightFM(no_components=self.factors, loss=self.loss, random_state=self.seed)
        self.model.fit(matrix, epochs=self.epochs, num_threads=1)
        self.backend_used = f"lightfm_{self.loss}"
        return self

    def score_items(self, user_id=None):
        uid = self.user_index.get(user_id)
        if uid is None or user_id not in self.positive_users:
            return self.popularity.copy()
        scores = self.model.predict(uid, np.arange(len(self.items), dtype=np.int32), num_threads=1)
        # Identity-only LightFM không học được item chưa có positive train.
        scores = np.asarray(scores, dtype=float)
        scores[[i not in self.positive_items for i in self.items]] = -np.inf
        return scores
