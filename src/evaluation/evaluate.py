"""Fixed, shared evaluation cohorts; never silently discard cold-start test rows."""

import numpy as np
import pandas as pd
from src.evaluation.metrics import rating_metrics, precision_recall_at_k


def evaluation_cohorts(test, max_users=100, max_ratings=2000, seed=42, threshold=7):
    explicit = test.loc[test["Book-Rating"].gt(0)].copy()
    if max_ratings and len(explicit) > max_ratings:
        explicit = explicit.sample(n=max_ratings, random_state=seed)
    # Group predictions by user, improving neighborhood model cache reuse.
    explicit = explicit.sort_values("User-ID", kind="stable")
    positive = test.loc[test["Book-Rating"].ge(threshold)]
    users = np.sort(positive["User-ID"].unique())
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    return explicit, users


def evaluate(model, train, test, rating_test, ranking_users, k=10, threshold=7):
    explicit_train = train.loc[train["Book-Rating"].gt(0)]
    result = {
        "test_rows": len(test), "test_explicit_total": int(test["Book-Rating"].gt(0).sum()),
        "test_unknown_item_rows": int((~test["Book-ID"].isin(train["Book-ID"])).sum()),
        "rating_cohort": len(rating_test), "rating_evaluated": 0,
        "rating_coverage": None, "mae": None, "rmse": None,
        "ranking_users_evaluated": len(ranking_users),
        "users_without_relevant_test": int(test["User-ID"].nunique()
                                           - test.loc[test["Book-Rating"].ge(threshold), "User-ID"].nunique()),
    }
    predictions = rating_test[["User-ID", "Book-ID", "Book-Rating"]].copy()
    predictions["cold_user"] = ~predictions["User-ID"].isin(explicit_train["User-ID"])
    predictions["cold_item"] = ~predictions["Book-ID"].isin(explicit_train["Book-ID"])
    if model.supports_ratings:
        predictions["prediction"] = [model.predict(u, i) for u, i in
                                       predictions[["User-ID", "Book-ID"]].itertuples(index=False, name=None)]
        result.update(rating_metrics(predictions["Book-Rating"], predictions["prediction"]))
        result["rating_coverage"] = result["rating_evaluated"] / len(predictions) if len(predictions) else None
        warm = ~predictions["cold_user"] & ~predictions["cold_item"]
        warm_metrics = rating_metrics(predictions.loc[warm, "Book-Rating"], predictions.loc[warm, "prediction"])
        result.update({f"warm_{key}": value for key, value in warm_metrics.items()})
        result["rating_note"] = "All explicit cohort rows including baseline cold-start; warm metrics exclude cold pairs"
    else:
        predictions["prediction"] = np.nan
        result["rating_note"] = "Ranking scores are not explicit rating predictions"
    result["rating_cold_users"] = int(predictions["cold_user"].sum())
    result["rating_cold_items"] = int(predictions["cold_item"].sum())
    relevant = test.loc[test["Book-Rating"].ge(threshold)].groupby("User-ID")["Book-ID"].agg(set).to_dict()
    rows = []
    for user in ranking_users:
        recs = model.recommend(user, k)
        precision, recall = precision_recall_at_k(recs["Book-ID"], relevant[user], k)
        rows.append({"User-ID": int(user), "precision_at_k": precision, "recall_at_k": recall,
                     "relevant_count": len(relevant[user]), "returned_count": len(recs),
                     "unknown_train_relevant": len(set(relevant[user]) - model.trained_items)})
    ranking = pd.DataFrame(rows)
    result["precision_at_k"] = float(ranking["precision_at_k"].mean()) if len(ranking) else None
    result["recall_at_k"] = float(ranking["recall_at_k"].mean()) if len(ranking) else None
    result["recommendation_fill_rate"] = float(ranking["returned_count"].sum() / (k * len(ranking))) if len(ranking) else None
    return result, predictions, ranking
