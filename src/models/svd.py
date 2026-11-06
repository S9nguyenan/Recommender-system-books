"""Biased matrix factorization on observed ratings (Surprise SVD preferred)."""

import numpy as np
import os
from src.models.base import BaseRecommender
from src.utils.config import PROJECT_ROOT


class SVDRecommender(BaseRecommender):
    supports_ratings = True

    def __init__(self, factors=32, epochs=15, learning_rate=0.005,
                 regularization=0.02, random_seed=42, backend="auto"):
        self.factors, self.epochs = factors, epochs
        self.lr, self.reg = learning_rate, regularization
        self.seed, self.backend = random_seed, backend

    def fit(self, books, interactions):
        self._setup(books, interactions)
        if self.explicit.empty:
            raise ValueError("SVD requires explicit train ratings")
        if self.backend not in {"auto", "surprise", "numpy"}:
            raise ValueError("backend must be auto, surprise or numpy")
        if self.backend != "numpy":
            # Surprise tạo thư mục cache khi import; đặt trong dự án, không dùng home.
            os.environ.setdefault("SURPRISE_DATA_FOLDER", str(PROJECT_ROOT / ".cache" / "surprise"))
            try:
                from surprise import Dataset, Reader, SVD
            except ImportError:
                if self.backend == "surprise":
                    raise ImportError("Install scikit-surprise, or use --svd-backend numpy")
            else:
                dataset = Dataset.load_from_df(self.explicit[["User-ID", "Book-ID", "Book-Rating"]],
                                               Reader(rating_scale=(1, 10)))
                self.model = SVD(n_factors=self.factors, n_epochs=self.epochs,
                                 lr_all=self.lr, reg_all=self.reg, random_state=self.seed)
                self.model.fit(dataset.build_full_trainset())
                self.backend_used = "surprise"
                return self
        self.backend_used = "numpy_observed_sgd"
        self.users = self.explicit["User-ID"].unique()
        self.user_index = {u: i for i, u in enumerate(self.users)}
        rng = np.random.default_rng(self.seed)
        self.p = rng.normal(0, 0.1, (len(self.users), self.factors))
        self.q = rng.normal(0, 0.1, (len(self.items), self.factors))
        self.bu = np.zeros(len(self.users)); self.bi = np.zeros(len(self.items))
        user = self.explicit["User-ID"].map(self.user_index).to_numpy(dtype=int)
        item = self.explicit["Book-ID"].map(self.item_index).to_numpy(dtype=int)
        rating = self.explicit["Book-Rating"].to_numpy(dtype=float)
        # Chỉ tối ưu trên rating đã quan sát. Ô trống và tương tác 0 không là nhãn.
        # Mini-batch gradients được cộng đúng cho các chỉ số user/item lặp lại.
        for _ in range(self.epochs):
            order = rng.permutation(len(rating))
            for start in range(0, len(order), 256):
                batch = order[start:start + 256]
                u, i = user[batch], item[batch]
                pu, qi = self.p[u].copy(), self.q[i].copy()
                error = rating[batch] - (self.global_mean + self.bu[u] + self.bi[i]
                                         + np.einsum("ij,ij->i", pu, qi))
                np.add.at(self.bu, u, self.lr * (error - self.reg * self.bu[u]))
                np.add.at(self.bi, i, self.lr * (error - self.reg * self.bi[i]))
                np.add.at(self.p, u, self.lr * (error[:, None] * qi - self.reg * pu))
                np.add.at(self.q, i, self.lr * (error[:, None] * pu - self.reg * qi))
        self.explicit_items = set(self.explicit["Book-ID"])
        return self

    def predict(self, user_id, book_id):
        if self.backend_used == "surprise":
            return float(self.model.predict(user_id, book_id).est)
        u, i = self.user_index.get(user_id), self.item_index.get(book_id)
        value = self.global_mean
        if u is not None:
            value += self.bu[u]
        if i is not None and book_id in self.explicit_items:
            value += self.bi[i]
            if u is not None:
                value += self.p[u] @ self.q[i]
        return float(np.clip(value, 1, 10))

    def score_items(self, user_id=None):
        if self.backend_used == "surprise":
            return np.array([self.predict(user_id, i) for i in self.items])
        u = self.user_index.get(user_id)
        scores = np.full(len(self.items), self.global_mean)
        if u is not None:
            scores += self.bu[u]
        known = np.array([i in self.explicit_items for i in self.items])
        scores[known] += self.bi[known]
        if u is not None:
            scores[known] += self.q[known] @ self.p[u]
        return np.clip(scores, 1, 10)
