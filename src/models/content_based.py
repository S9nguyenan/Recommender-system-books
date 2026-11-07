"""Gợi ý dựa trên nội dung: dùng TF-IDF để tìm sách có mô tả giống sách người dùng thích."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    def __init__(self, max_features=30000, relevance_threshold=7):
        self.max_features = max_features
        self.relevance_threshold = relevance_threshold

    def fit(self, books, interactions):
        self._setup(books, interactions)
        # Ghép tên, tác giả, nhà xuất bản thành mô tả sách.
        text = books[["Book-Title", "Book-Author", "Publisher"]].fillna("").agg(" ".join, axis=1)
        text = text.str.strip().replace("", "unknown")
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, dtype=np.float32,
                                          token_pattern=r"(?u)\b\w+\b")
        # Metadata catalog được biết trước; không dùng rating test để fit TF-IDF.
        self.features = self.vectorizer.fit_transform(text)
        # Chỉ lấy sách được chấm từ ngưỡng thích trở lên trong train.
        liked = self.explicit.loc[self.explicit["Book-Rating"].ge(self.relevance_threshold)]
        self.liked = liked.groupby("User-ID")["Book-ID"].agg(list).to_dict()
        return self

    def score_items(self, user_id=None):
        indices = [self.item_index[i] for i in self.liked.get(user_id, []) if i in self.item_index]
        # Chưa có sách yêu thích thì dùng điểm phổ biến.
        if not indices:
            return self.popularity.copy()
        # Trung bình vector sách đã thích tạo hồ sơ; chuẩn hóa để tính cosine.
        profile = normalize(np.asarray(self.features[indices].mean(axis=0)))
        return np.asarray(self.features @ profile.T).ravel()

    def similar_books(self, book_id, n=10):
        if book_id not in self.item_index:
            raise ValueError(f"Unknown Book-ID: {book_id}")
        # So cosine với catalog rồi loại chính sách đầu vào.
        scores = (self.features @ self.features[self.item_index[book_id]].T).toarray().ravel()
        return self._result(scores, n, [book_id])
