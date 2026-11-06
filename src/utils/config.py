"""Paths are based on this file, independent of the terminal directory."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class PreprocessingConfig:
    min_year: int = 1450
    max_year: int = 2026
    min_age: int = 5
    max_age: int = 100
    test_fraction: float = 0.2
    random_seed: int = 42

    def __post_init__(self):
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.min_year > self.max_year or self.min_age > self.max_age:
            raise ValueError("Invalid year or age bounds")


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "quick"
    seed: int = 42
    max_users: int = 1500
    max_items: int = 5000
    eval_users: int = 50
    eval_ratings: int = 1000
    neighbors: int = 30
    factors: int = 32
    epochs: int = 15
    k: int = 10
    relevance_threshold: int = 7
    max_features: int = 30000
    svd_backend: str = "auto"
    lightfm_loss: str = "logistic"

    def __post_init__(self):
        if self.mode not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        for name in ["max_users", "max_items", "eval_users", "eval_ratings"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; 0 means unlimited")
        for name in ["neighbors", "factors", "epochs", "k", "max_features"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.relevance_threshold <= 10:
            raise ValueError("relevance_threshold must be between 1 and 10")
