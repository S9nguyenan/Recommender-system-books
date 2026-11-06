"""Item cosine similarities restricted to each user's rated history."""

import numpy as np
from src.models.cf_common import SparseCF


class ItemBasedCF(SparseCF):
    def score_items(self, user_id=None):
        if user_id in self._cache:
            return self._cache[user_id]
        scores = np.array([self.baseline(user_id, i) for i in self.items])
        uid = self.user_index.get(user_id)
        if uid is not None:
            history = self.matrix[uid]
            # Mỗi block chỉ có (số sách ứng viên trong block) x (lịch sử user).
            for start in range(0, len(self.items), 256):
                sim = (self.item_norm[start:start + 256] @ self.item_norm[history.indices].T).toarray()
                if sim.shape[1] > self.neighbors:
                    remove = np.argpartition(sim, -self.neighbors, axis=1)[:, :-self.neighbors]
                    np.put_along_axis(sim, remove, 0, axis=1)
                denominator = sim.sum(axis=1)
                numerator = sim @ history.data
                block = scores[start:start + len(sim)]
                np.divide(numerator, denominator, out=block, where=denominator > 0)
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[user_id] = scores
        return scores
