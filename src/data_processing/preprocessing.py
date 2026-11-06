"""Deterministic cleaning and user-wise holdout, with an audit report."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_processing.load_data import BookCrossingData
from src.utils.config import PreprocessingConfig


@dataclass
class PreparedData:
    books: pd.DataFrame
    users: pd.DataFrame
    interactions: pd.DataFrame
    explicit_ratings: pd.DataFrame
    implicit_interactions: pd.DataFrame
    report: dict


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in frame.columns:
        frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA)
    return frame


def _integer(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.mod(1).eq(0)).astype("Int64")


def _unique_keys(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """Collapse exact duplicates; refuse conflicting records rather than guess."""
    frame = frame.drop_duplicates()
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: conflicting duplicate key {keys}")
    return frame


def preprocess(data: BookCrossingData,
               config: PreprocessingConfig = PreprocessingConfig()) -> PreparedData:
    books, ratings, users = map(_normalize, (data.books, data.ratings, data.users))
    report = {"raw_rows": {"books": len(books), "ratings": len(ratings), "users": len(users)},
              "raw_missing": {name: frame.isna().sum().astype(int).to_dict()
                              for name, frame in [("books", books), ("ratings", ratings), ("users", users)]}}
    for name, frame in [("books", books), ("ratings", ratings), ("users", users)]:
        report[f"{name}_exact_duplicates"] = int(frame.duplicated().sum())

    report["missing_book_id_rows"] = int(books["Book-ID"].isna().sum())
    books = books.dropna(subset=["Book-ID"])
    books = _unique_keys(books, ["Book-ID"], "Books")
    # Source inspection found shifted metadata. Preserve the original text;
    # do not invent authors/titles or allow a year to become an author feature.
    year = _integer(books["Year-Of-Publication"])
    suspicious = year.isna() & books["Year-Of-Publication"].notna()
    books["Metadata-Needs-Review"] = suspicious
    report["metadata_review_book_ids"] = books.loc[suspicious, "Book-ID"].tolist()
    books.loc[suspicious, ["Book-Author", "Publisher", "Image-URL-S",
                           "Image-URL-M", "Image-URL-L"]] = pd.NA
    valid_year = year.between(config.min_year, config.max_year).fillna(False)
    report["invalid_or_missing_year"] = int((~valid_year).sum())
    books["Year-Of-Publication"] = year.where(valid_year)
    books["Book-Title"] = books["Book-Title"].fillna("Unknown title")
    for col in ["Book-Author", "Publisher"]:
        books[col] = books[col].fillna("Unknown")

    users["User-ID"] = _integer(users["User-ID"])
    valid_user = users["User-ID"].gt(0).fillna(False)
    report["invalid_user_rows"] = int((~valid_user).sum())
    users = _unique_keys(users.loc[valid_user], ["User-ID"], "Users").copy()
    age = _integer(users["Age"])
    valid_age = age.between(config.min_age, config.max_age).fillna(False)
    report["invalid_or_missing_age"] = int((~valid_age).sum())
    users["Age"] = age.where(valid_age)
    users["Location"] = users["Location"].fillna("Unknown")

    ratings["User-ID"] = _integer(ratings["User-ID"])
    ratings["Book-Rating"] = _integer(ratings["Book-Rating"])
    valid = (ratings["User-ID"].gt(0) & ratings["Book-ID"].notna()
             & ratings["Book-Rating"].between(0, 10)).fillna(False)
    report["invalid_rating_rows"] = int((~valid).sum())
    ratings = _unique_keys(ratings.loc[valid], ["User-ID", "Book-ID"], "Ratings")
    known_book = ratings["Book-ID"].isin(books["Book-ID"])
    known_user = ratings["User-ID"].isin(users["User-ID"])
    report["unknown_book_rating_rows"] = int((~known_book).sum())
    report["unknown_user_rating_rows"] = int((~known_user).sum())
    # A shared catalog makes later comparisons between six models consistent.
    ratings = ratings.loc[known_book & known_user].reset_index(drop=True)
    explicit = ratings.loc[ratings["Book-Rating"].gt(0)].reset_index(drop=True)
    implicit = ratings.loc[ratings["Book-Rating"].eq(0)].reset_index(drop=True)
    report["clean_rows"] = {"books": len(books), "users": len(users),
                            "interactions": len(ratings), "explicit": len(explicit),
                            "implicit": len(implicit)}
    return PreparedData(books.reset_index(drop=True), users.reset_index(drop=True),
                        ratings, explicit, implicit, report)


def split_interactions(interactions: pd.DataFrame, test_fraction: float = 0.2,
                       random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random holdout within each user; singleton users remain in train.

    Split ALL interactions together before deriving explicit/implicit training
    views. This prevents a held-out user/book pair entering another train view.
    There are no timestamps, so this is not a chronological evaluation.
    New items can occur in test and must be reported during later evaluation.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if interactions.duplicated(["User-ID", "Book-ID"]).any():
        raise ValueError("Split requires unique user/book pairs")
    frame = interactions.reset_index(drop=True)
    rng = np.random.default_rng(random_seed)
    # Shuffle once, then take a bounded number of rows from each user.
    shuffled = frame.iloc[rng.permutation(len(frame))]
    groups = shuffled.groupby("User-ID", sort=False)
    sizes = groups["Book-ID"].transform("size")
    counts = np.floor(sizes * test_fraction).astype(int).clip(lower=1)
    counts = np.minimum(counts, sizes - 1)
    test_indices = shuffled.index[groups.cumcount() < counts]
    mask = frame.index.isin(test_indices)
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)
