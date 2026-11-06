"""Shared CLI, preparation, evaluation and error isolation for all models."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

from experiments.check_data import source_hashes
from src.data_processing.load_data import load_data
from src.data_processing.preprocessing import preprocess, split_interactions
from src.evaluation.evaluate import evaluate, evaluation_cohorts
from src.utils.config import DATA_DIR, RESULTS_DIR, ExperimentConfig

MODEL_NAMES = ["popularity", "content_based", "user_cf", "item_cf", "svd", "lightfm"]


def create_model(name, config):
    if name == "popularity":
        from src.models.popularity import PopularityRecommender
        return PopularityRecommender()
    if name == "content_based":
        from src.models.content_based import ContentBasedRecommender
        return ContentBasedRecommender(config.max_features, config.relevance_threshold)
    if name == "user_cf":
        from src.models.user_based_cf import UserBasedCF
        return UserBasedCF(config.neighbors)
    if name == "item_cf":
        from src.models.item_based_cf import ItemBasedCF
        return ItemBasedCF(config.neighbors)
    if name == "svd":
        from src.models.svd import SVDRecommender
        return SVDRecommender(config.factors, config.epochs, random_seed=config.seed, backend=config.svd_backend)
    if name == "lightfm":
        from src.models.lightfm_model import LightFMRecommender
        return LightFMRecommender(config.factors, config.epochs, config.seed,
                                  config.relevance_threshold, config.lightfm_loss)
    raise ValueError(f"Unknown model: {name}")


def prepare(config, data_dir):
    prepared = preprocess(load_data(data_dir))
    train, test = split_interactions(prepared.interactions, random_seed=config.seed)
    original_test_rows = len(test)
    books = prepared.books
    # Chọn quick subset bằng train, không nhìn test để chọn sách/user phổ biến.
    if config.mode == "quick":
        counts = train["User-ID"].value_counts().rename_axis("id").reset_index(name="count")
        counts = counts.sort_values(["count", "id"], ascending=[False, True])
        users = counts["id"].head(config.max_users) if config.max_users else counts["id"]
        train = train.loc[train["User-ID"].isin(users)]
        item_counts = train["Book-ID"].value_counts().rename_axis("id").reset_index(name="count")
        item_counts = item_counts.sort_values(["count", "id"], ascending=[False, True])
        items = item_counts["id"].head(config.max_items) if config.max_items else item_counts["id"]
        books = books.loc[books["Book-ID"].isin(items)]
        train = train.loc[train["Book-ID"].isin(items)]
        test = test.loc[test["User-ID"].isin(train["User-ID"]) & test["Book-ID"].isin(items)]
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Empty train/test after filtering; increase quick limits or use --mode full")
    if len(train.merge(test, on=["User-ID", "Book-ID"])):
        raise ValueError("Train/test leakage detected")
    rating_test, users = evaluation_cohorts(test, config.eval_users, config.eval_ratings,
                                           config.seed, config.relevance_threshold)
    scope = {"original_interactions": len(prepared.interactions), "catalog": len(books),
             "train_rows": len(train), "test_rows": len(test),
             "test_excluded_by_quick_scope": original_test_rows - len(test),
             "test_unknown_item_rows": int((~test["Book-ID"].isin(train["Book-ID"])).sum()),
             "rating_cohort_rows": len(rating_test), "ranking_cohort_users": len(users)}
    return books, train, test, rating_test, users, scope


def environment():
    result = {"python": platform.python_version(), "executable": sys.executable}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "scikit-surprise", "lightfm"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_one(name, config, bundle, output, user_id=None, book_id=None):
    books, train, test, rating_test, users, scope = bundle
    started = time.perf_counter()
    row = {"model": name, "mode": config.mode, "status": "failed", "error": None,
           "seed": config.seed, "k": config.k, "relevance_threshold": config.relevance_threshold}
    try:
        model = create_model(name, config)
        model.fit(books, train)
        row["fit_seconds"] = time.perf_counter() - started
        row["backend"] = getattr(model, "backend_used", name)
        metrics, predictions, ranking = evaluate(model, train, test, rating_test, users,
                                                  config.k, config.relevance_threshold)
        row.update(metrics)
        example_user = user_id if user_id is not None else int(users[0] if len(users) else train["User-ID"].iloc[0])
        recommendations = model.recommend(example_user, config.k)
        recommendations.to_csv(output / f"{name}_recommendations.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
        ranking.to_csv(output / f"{name}_ranking.csv", index=False, encoding="utf-8-sig")
        print(f"\n{name}: user={example_user}", flush=True)
        print(recommendations.to_string(index=False), flush=True)
        if name == "content_based" and book_id is not None:
            similar = model.similar_books(book_id, config.k)
            similar.to_csv(output / "content_based_similar_books.csv", index=False, encoding="utf-8-sig")
            print(similar.to_string(index=False), flush=True)
        row["status"] = "success"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        (output / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"{name}: FAILED: {row['error']}", flush=True)
    row["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / f"{name}_metrics.json", row)
    pd.DataFrame([row]).to_csv(output / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
    return row


def cli(single_model=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Book recommender experiments on real CSV data")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--book-id")
    defaults = ExperimentConfig()
    for field in ["seed", "max_users", "max_items", "eval_users", "eval_ratings", "neighbors",
                  "factors", "epochs", "k", "relevance_threshold", "max_features"]:
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    parser.add_argument("--svd-backend", choices=["auto", "surprise", "numpy"], default="auto")
    parser.add_argument("--lightfm-loss", choices=["logistic", "warp", "bpr"], default="logistic")
    args = parser.parse_args()
    config = ExperimentConfig(**{key: getattr(args, key) for key in asdict(defaults)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.results_dir / f"{config.mode}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": stamp, "config": asdict(config), "environment": environment()}
    try:
        manifest["source_sha256"] = source_hashes(args.data_dir)
        print(f"Preparing real data: mode={config.mode}", flush=True)
        bundle = prepare(config, args.data_dir)
        manifest["scope"] = bundle[-1]
        print(json.dumps(bundle[-1], indent=2), flush=True)
        rows = []
        for name in ([single_model] if single_model else list(dict.fromkeys(args.models))):
            rows.append(run_one(name, config, bundle, output, args.user_id, args.book_id))
            # Save after each model so earlier results survive a later interruption.
            pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
            write_json(output / "model_comparison.json", rows)
        if manifest["source_sha256"] != source_hashes(args.data_dir):
            raise RuntimeError("Source CSV files changed during experiment")
        manifest["source_unchanged"] = True
        manifest["status"] = "completed" if all(r["status"] == "success" for r in rows) else "completed_with_errors"
        # Root files reference this run; timestamped folders preserve previous runs.
        pd.DataFrame(rows).to_csv(args.results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
        write_json(args.results_dir / "model_comparison.json", {"run_dir": str(output.resolve()),
                                                                "manifest": manifest, "models": rows})
        summary = pd.DataFrame(rows).reindex(columns=["model", "status", "mae", "rmse",
                                                       "precision_at_k", "recall_at_k", "elapsed_seconds"])
        print(summary.fillna("N/A").to_string(index=False), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(manifest["error"], flush=True)
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output.resolve()}", flush=True)
    if single_model and rows[0]["status"] != "success":
        raise SystemExit(1)
