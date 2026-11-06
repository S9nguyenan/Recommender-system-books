"""Popularity based on explicit rating count and shrunk mean."""

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def fit(self, books, interactions):
        return self._setup(books, interactions)

    def score_items(self, user_id=None):
        return self.popularity
