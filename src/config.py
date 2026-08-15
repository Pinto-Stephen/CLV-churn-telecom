"""Typed access to config.yaml.

Every tunable number in this project (dates, priors, thresholds, budgets) lives in
config.yaml at the repo root. Modules import from here rather than hardcoding values,
so a single file controls reproducibility and business assumptions.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_xlsx: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class DataConfig:
    sheet_names: list[str]
    non_product_stock_codes: set[str]
    non_product_stock_code_regex: str
    min_duplicate_removed_warning_threshold: int


@dataclass(frozen=True)
class RFMConfig:
    calibration_end: str
    observation_end: str
    new_cohort_window_days: int


@dataclass(frozen=True)
class BTYDConfig:
    sampler: dict[str, Any]
    rhat_threshold: float
    monetary_frequency_correlation_warn_threshold: float
    clv_future_t_months: int
    discount_rate_monthly: float
    gross_margin: float
    time_unit: str


@dataclass(frozen=True)
class FeaturesConfig:
    trend_window_days: int
    top_n_countries: int


@dataclass(frozen=True)
class ChurnConfig:
    test_size: float
    cv_folds: int
    random_search_iterations: int
    calibration_brier_threshold: float
    xgb_param_distributions: dict[str, list[Any]]


@dataclass(frozen=True)
class AllocateConfig:
    total_budget: float
    uplift: float
    cost_per_customer: float
    uplift_sweep: list[float]
    cost_sweep: list[float]
    n_terciles: int


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: DataConfig
    rfm: RFMConfig
    btyd: BTYDConfig
    features: FeaturesConfig
    churn: ChurnConfig
    allocate: AllocateConfig
    raw: dict[str, Any] = field(repr=False)


_cached_config: Config | None = None


def load_config(force: bool = False) -> Config:
    """Load and type-check config.yaml, caching the result for the process lifetime.

    Returns the single source of truth for every tunable number used across the
    pipeline: dates for the calibration/holdout split, MCMC sampler settings,
    the gross-margin assumption, XGBoost search space, and the retention budget.
    """
    global _cached_config
    if _cached_config is not None and not force:
        return _cached_config

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = Paths(
        raw_xlsx=REPO_ROOT / paths_raw["raw_xlsx"],
        interim_dir=REPO_ROOT / paths_raw["interim_dir"],
        processed_dir=REPO_ROOT / paths_raw["processed_dir"],
        reports_dir=REPO_ROOT / paths_raw["reports_dir"],
        figures_dir=REPO_ROOT / paths_raw["figures_dir"],
    )

    data_raw = raw["data"]
    data = DataConfig(
        sheet_names=list(data_raw["sheet_names"]),
        non_product_stock_codes=set(data_raw["non_product_stock_codes"]),
        non_product_stock_code_regex=data_raw["non_product_stock_code_regex"],
        min_duplicate_removed_warning_threshold=data_raw[
            "min_duplicate_removed_warning_threshold"
        ],
    )

    rfm = RFMConfig(**raw["rfm"])
    btyd = BTYDConfig(**raw["btyd"])
    features = FeaturesConfig(**raw["features"])
    churn = ChurnConfig(**raw["churn"])
    allocate = AllocateConfig(**raw["allocate"])

    cfg = Config(
        seed=raw["seed"],
        paths=paths,
        data=data,
        rfm=rfm,
        btyd=btyd,
        features=features,
        churn=churn,
        allocate=allocate,
        raw=raw,
    )

    for d in (paths.interim_dir, paths.processed_dir, paths.reports_dir, paths.figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    _cached_config = cfg
    return cfg


def seed_everything(seed: int | None = None) -> int:
    """Seed numpy and Python's random module so pipeline reruns are reproducible.

    Returns the seed actually applied (config seed if none was passed explicitly).
    Individual modules additionally pass this seed into xgboost and PyMC calls,
    since those libraries do not read the global numpy/random state.
    """
    resolved_seed = seed if seed is not None else load_config().seed
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    logger.info("Seeded numpy and random with seed=%d", resolved_seed)
    return resolved_seed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    seed_everything(cfg.seed)
    print("Loaded config from", CONFIG_PATH)
    print("seed:", cfg.seed)
    print("raw_xlsx:", cfg.paths.raw_xlsx, "exists:", cfg.paths.raw_xlsx.exists())
    print("calibration_end:", cfg.rfm.calibration_end, "observation_end:", cfg.rfm.observation_end)
    print("non_product_stock_codes:", sorted(cfg.data.non_product_stock_codes))
    print("btyd sampler:", cfg.btyd.sampler)
    print("allocate budget:", cfg.allocate.total_budget, "uplift:", cfg.allocate.uplift)
