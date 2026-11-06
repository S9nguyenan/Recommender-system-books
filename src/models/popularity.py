"""Popularity based on explicit rating count and shrunk mean."""

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def fit(self, books, interactions):
        # Tính thống kê rating train và điểm phổ biến trong lớp cha.
        return self._setup(books, interactions)

    def score_items(self, user_id=None):
        # Mọi user nhận cùng điểm; recommend() loại sách từng người đã xem.
        return self.popularity
