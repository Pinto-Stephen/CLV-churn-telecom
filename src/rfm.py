"""Calibration/holdout split, RFM summarisation, and proxy churn labels.

The dataset spans 2009-12-01 to 2011-12-09 (~740 days). We split it into a
calibration window (used to fit BTYD models and compute churn features) and a
holdout window (used to validate BTYD predictions and to derive the churn label).
Both windows are read from config.yaml -- calibration_end and observation_end.

IMPORTANT LIMITATION -- proxy churn label: in a non-contractual retail setting,
"churn" is never directly observed. There is no cancellation event; we only see the
absence of a future purchase. Labelling churned = 1 for zero holdout-window purchases
is therefore a proxy, not ground truth. A wholesaler on a nine-month reorder cycle will
be misclassified as churned by a six-month holdout window purely because the window is
shorter than their natural buying cycle. This module logs how many customers have a
historical inter-purchase gap exceeding the holdout length, since those are the
known-suspect labels -- their "churn" may just be a cycle the window is too short to see.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from pymc_marketing.clv.utils import rfm_train_test_split

from src.config import load_config
from src.load import load_transactions

logger = logging.getLogger(__name__)


def _assert_t_geq_recency(rfm_calibration: pd.DataFrame) -> None:
    violations = rfm_calibration.loc[rfm_calibration["T"] < rfm_calibration["recency"]]
    if len(violations) > 0:
        raise AssertionError(
            f"T >= recency violated for {len(violations)} customers -- BG/NBD is undefined "
            f"for these rows. Example customer_ids: {violations['customer_id'].head(5).tolist()}"
        )
    logger.info("Verified T >= recency for all %d calibration customers", len(rfm_calibration))


def compute_rfm_and_churn(transactions: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build calibration-window RFM summary and holdout-derived proxy churn labels.

    Returns (rfm_calibration, churn_labels):

    rfm_calibration -- one row per customer with at least one calibration-window
    purchase: customer_id, frequency (repeat purchases, first excluded), recency
    (age at last calibration purchase), T (age at calibration_end), monetary_value
    (mean revenue of repeat transactions), is_new_cohort (first purchase fell in the
    last `new_cohort_window_days` days of calibration -- these customers have almost
    no observable history and downstream code may want to exclude them, but they are
    not dropped here).

    churn_labels -- customer_id, churned (1 if zero holdout-window purchases, else 0),
    is_new_cohort. See the module docstring for why this is a proxy label, not ground
    truth.
    """
    cfg = load_config()

    split = rfm_train_test_split(
        transactions,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        train_period_end=cfg.rfm.calibration_end,
        test_period_end=cfg.rfm.observation_end,
        time_unit="D",
        monetary_value_col="revenue",
    )
    logger.info("rfm_train_test_split produced %d calibration-window customers", len(split))

    _assert_t_geq_recency(split)

    is_new_cohort = split["T"] <= cfg.rfm.new_cohort_window_days
    logger.info(
        "%d of %d customers (%.1f%%) flagged is_new_cohort (first purchase within %d days of calibration_end)",
        is_new_cohort.sum(),
        len(split),
        100 * is_new_cohort.mean(),
        cfg.rfm.new_cohort_window_days,
    )

    rfm_calibration = split[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    rfm_calibration["is_new_cohort"] = is_new_cohort.values

    churned = (split["test_frequency"] == 0).astype(int)
    churn_labels = pd.DataFrame(
        {
            "customer_id": split["customer_id"].values,
            "churned": churned.values,
            "is_new_cohort": is_new_cohort.values,
        }
    )

    base_rate = churn_labels["churned"].mean()
    logger.info("Proxy churn base rate: %.1f%% (%d of %d customers)", 100 * base_rate, churned.sum(), len(churn_labels))

    holdout_length_days = (
        pd.Timestamp(cfg.rfm.observation_end) - pd.Timestamp(cfg.rfm.calibration_end)
    ).days
    repeat_buyers = split.loc[split["frequency"] > 0]
    avg_interpurchase_days = repeat_buyers["recency"] / repeat_buyers["frequency"]
    suspect_mask = avg_interpurchase_days > holdout_length_days
    logger.info(
        "%d of %d repeat buyers (%.1f%%) have an average inter-purchase gap longer than the "
        "%d-day holdout window -- these churn labels are known-suspect (a real reorder cycle "
        "the window is too short to observe would look identical to genuine churn)",
        suspect_mask.sum(),
        len(repeat_buyers),
        100 * suspect_mask.mean() if len(repeat_buyers) else 0.0,
        holdout_length_days,
    )

    return rfm_calibration, churn_labels


def load_rfm_and_churn(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached wrapper around compute_rfm_and_churn.

    Reads dataset/processed/rfm_calibration.parquet and churn_labels.parquet if present;
    otherwise recomputes from load_transactions() and writes both.
    """
    cfg = load_config()
    rfm_path = cfg.paths.processed_dir / "rfm_calibration.parquet"
    churn_path = cfg.paths.processed_dir / "churn_labels.parquet"

    if not force and rfm_path.exists() and churn_path.exists():
        logger.info("Loading cached RFM/churn tables from %s and %s", rfm_path, churn_path)
        return pd.read_parquet(rfm_path), pd.read_parquet(churn_path)

    transactions = load_transactions(force=force)
    rfm_calibration, churn_labels = compute_rfm_and_churn(transactions, force=force)

    rfm_calibration.to_parquet(rfm_path, index=False)
    churn_labels.to_parquet(churn_path, index=False)
    logger.info("Wrote %s and %s", rfm_path, churn_path)

    return rfm_calibration, churn_labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached parquet exists")
    args = parser.parse_args()

    rfm_calibration, churn_labels = load_rfm_and_churn(force=args.force)

    print("rfm_calibration shape:", rfm_calibration.shape)
    print(rfm_calibration.head())
    print(rfm_calibration.describe())
    print()
    print("churn_labels shape:", churn_labels.shape)
    print(churn_labels.head())
    print("churn base rate:", churn_labels["churned"].mean())


if __name__ == "__main__":
    main()
