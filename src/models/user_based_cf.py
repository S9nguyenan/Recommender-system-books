"""User cosine k-nearest neighbors using only explicit train ratings."""

import numpy as np
from src.models.cf_common import SparseCF


class UserBasedCF(SparseCF):
    def score_items(self, user_id=None):
        if user_id in self._cache:
            return self._cache[user_id]
        # Khởi tạo điểm dự phòng cho sách không có láng giềng đánh giá.
        scores = np.array([self.baseline(user_id, i) for i in self.items])
        uid = self.user_index.get(user_id)
        if uid is not None:
            # Chỉ tính một hàng tương đồng; không tạo ma trận user x user.
            similarity = (self.user_norm @ self.user_norm[uid].T).toarray().ravel()
            idx, similarity = self._top(similarity, uid)
            if len(idx):
                weights = similarity[idx]
                # Dự đoán = tổng (rating × độ giống user) / tổng độ giống user đã chấm sách.
                numerator = np.asarray(self.matrix[idx].T @ weights).ravel()
                denominator = np.asarray(self.observed[idx].T @ weights).ravel()
                # Chỉ thay điểm khi mẫu số dương để tránh chia cho 0.
                np.divide(numerator, denominator, out=scores, where=denominator > 0)
        # Cache có giới hạn, tránh giữ vector cho mọi người dùng trên full data.
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[user_id] = scores
        return scores
