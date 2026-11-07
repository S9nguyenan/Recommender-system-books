"""Lớp dùng chung cho lọc cộng tác (CF): tạo ma trận đánh giá thưa và chọn láng giềng."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.models.base import BaseRecommender


class SparseCF(BaseRecommender):
    supports_ratings = True

    def __init__(self, neighbors=30):
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.neighbors = neighbors

    def fit(self, books, interactions):
        self._setup(books, interactions)
        self.users = self.explicit["User-ID"].unique()
        self.user_index = {u: i for i, u in enumerate(self.users)}
        rows = self.explicit["User-ID"].map(self.user_index).to_numpy(dtype=int)
        cols = self.explicit["Book-ID"].map(self.item_index).to_numpy(dtype=int)
        values = self.explicit["Book-Rating"].to_numpy(dtype=float)
        # Ma trận thưa: hàng là user, cột là sách, giá trị là rating.
        self.matrix = csr_matrix((values, (rows, cols)), shape=(len(self.users), len(self.items)))
        # Ma trận 0/1 đánh dấu có rating để tính tổng trọng số cho từng sách.
        self.observed = self.matrix.copy()
        self.observed.data[:] = 1
        # Chuẩn hóa vector để tính cosine bằng nhân ma trận.
        self.user_norm = normalize(self.matrix) if len(self.users) else self.matrix.copy()
        self.item_norm = normalize(self.matrix.T.tocsr()) if len(self.users) else self.matrix.T.tocsr()
        self._cache = {}
        return self

    def _top(self, similarities, exclude=None):
        values = np.asarray(similarities).ravel()
        if exclude is not None:
            values[exclude] = 0
        # Giữ tối đa neighbors láng giềng có độ giống dương.
        idx = np.flatnonzero(values > 0)
        return idx[np.argsort(-values[idx], kind="stable")[:self.neighbors]], values

    def predict(self, user_id, book_id):
        if book_id not in self.item_index:
            return self.baseline(user_id, book_id)
        return float(np.clip(self.score_items(user_id)[self.item_index[book_id]], 1, 10))
