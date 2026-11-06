"""Strict CSV loading: never silently skip malformed records."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_DIR

SCHEMAS = {
    "Books": ["Book-ID", "Book-Title", "Book-Author", "Year-Of-Publication",
              "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"],
    "Ratings": ["User-ID", "Book-ID", "Book-Rating"],
    "Users": ["User-ID", "Location", "Age"],
}


@dataclass
class BookCrossingData:
    books: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Keep IDs as strings (including leading zeroes); use standard CSV quoting.

    Empty fields are missing. Literal strings such as 'NA' remain strings.
    A decoding/schema/parser error is reported instead of guessing an encoding.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        frame = pd.read_csv(path, sep=",", encoding="utf-8-sig", dtype="string",
                            keep_default_na=False, na_values=[""],
                            doublequote=True, on_bad_lines="error")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return frame


def load_data(data_dir: Path = DATA_DIR) -> BookCrossingData:
    """Read all three source tables without modifying the files."""
    tables = {name: load_csv(Path(data_dir) / f"{name}.csv", columns)
              for name, columns in SCHEMAS.items()}
    return BookCrossingData(tables["Books"], tables["Ratings"], tables["Users"])
