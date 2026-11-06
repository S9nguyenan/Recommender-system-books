"""Run a real-data audit: python -m experiments.check_data."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from src.data_processing.load_data import load_data
from src.data_processing.preprocessing import preprocess, split_interactions
from src.utils.config import DATA_DIR, RESULTS_DIR, PreprocessingConfig


def source_hashes(directory: Path) -> dict:
    hashes = {}
    for name in ["Books.csv", "Ratings.csv", "Users.csv"]:
        with (directory / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def main():
    # Windows redirected terminals may otherwise use cp1252 for Vietnamese paths.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    config = PreprocessingConfig()
    before = source_hashes(args.data_dir)
    raw = load_data(args.data_dir)
    prepared = preprocess(raw, config)
    train, test = split_interactions(prepared.interactions, config.test_fraction, config.random_seed)
    train2, test2 = split_interactions(prepared.interactions, config.test_fraction, config.random_seed)
    keys = ["User-ID", "Book-ID"]
    assert not train.merge(test, on=keys).shape[0], "Train/test pair leakage"
    assert len(train) + len(test) == len(prepared.interactions)
    assert set(test["User-ID"]).issubset(set(train["User-ID"]))
    assert train.equals(train2) and test.equals(test2), "Split is not reproducible"
    assert prepared.explicit_ratings["Book-Rating"].between(1, 10).all()
    assert prepared.implicit_interactions["Book-Rating"].eq(0).all()
    assert set(prepared.interactions["Book-ID"]).issubset(set(prepared.books["Book-ID"]))
    assert raw.books["Year-Of-Publication"].dtype == pd.StringDtype()
    assert before == source_hashes(args.data_dir), "Source files changed"
    report = prepared.report
    report["config"] = asdict(config)
    report["source_sha256"] = before
    report["split"] = {"train": len(train), "test": len(test),
                       "train_explicit": int(train["Book-Rating"].gt(0).sum()),
                       "test_explicit": int(test["Book-Rating"].gt(0).sum()),
                       "test_implicit": int(test["Book-Rating"].eq(0).sum()),
                       "test_items_not_in_train": int((~test["Book-ID"].isin(train["Book-ID"])).sum()),
                       "train_users": int(train["User-ID"].nunique()),
                       "test_users": int(test["User-ID"].nunique())}
    report["checks_passed"] = True
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "data_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = pd.json_normalize(report, sep=".")
    flat.to_csv(args.results_dir / "data_validation.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"clean_rows": report["clean_rows"], "split": report["split"],
                      "checks_passed": True}, indent=2))
    print(f"Reports: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
