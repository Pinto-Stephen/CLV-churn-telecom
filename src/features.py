"""Churn feature engineering, computed strictly from calibration-window data.

Every feature in this module is derived only from transactions on or before
calibration_end. The churn label (src/rfm.py) is derived from what happens *after*
calibration_end, so any feature that peeked past that boundary would leak the label
into the inputs and invalidate every downstream metric. assert_no_leakage() enforces
this at the point every calibration-window subset is built, not just at the end.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.btyd import load_clv
from src.config import load_config
from src.load import load_clean_lines, load_transactions
from src.rfm import load_rfm_and_churn

logger = logging.getLogger(__name__)


def assert_no_leakage(df: pd.DataFrame, calibration_end: str, date_col: str = "invoice_date") -> None:
    """Raise if any row's date falls after calibration_end.

    This is the leakage guard referenced in the module docstring: called on every
    calibration-window subset before it feeds a feature, so a future bug that widens
    a date filter fails loudly instead of silently leaking the churn label.
    """
    max_date = df[date_col].max()
    if max_date > pd.Timestamp(calibration_end):
        violations = (df[date_col] > pd.Timestamp(calibration_end)).sum()
        raise AssertionError(
            f"Leakage guard failed: {violations} rows have {date_col} after calibration_end "
            f"({calibration_end}); max date found was {max_date}."
        )


def _restrict_to_calibration(df: pd.DataFrame, calibration_end: str, date_col: str = "invoice_date") -> pd.DataFrame:
    out = df.loc[df[date_col] <= pd.Timestamp(calibration_end)].copy()
    assert_no_leakage(out, calibration_end, date_col)
    return out


def _derived_rfm_features(rfm_calibration: pd.DataFrame) -> pd.DataFrame:
    """recency_ratio, days_since_last_purchase, purchase_rate, is_one_time_buyer.

    frequency = 0 customers have no repeat-purchase interval to speak of; their
    interpurchase statistics are computed separately (see _interpurchase_features)
    and filled with an explicit sentinel here rather than a mean, so the model can
    tell "no signal" apart from "a genuinely short gap".
    """
    out = rfm_calibration[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    out["recency_ratio"] = out["recency"] / out["T"]
    out["days_since_last_purchase"] = out["T"] - out["recency"]
    out["purchase_rate"] = out["frequency"] / out["T"]
    out["is_one_time_buyer"] = (out["frequency"] == 0).astype(int)
    return out


def _interpurchase_features(clean_lines_cal: pd.DataFrame, sentinel: float = -1.0) -> pd.DataFrame:
    """avg_interpurchase_days, std_interpurchase_days per customer, day-collapsed.

    Undefined for customers with fewer than two distinct purchase days (one-time
    buyers); those rows get the sentinel rather than an imputed mean, paired with
    is_one_time_buyer from _derived_rfm_features so the model can distinguish them.
    """
    days = (
        clean_lines_cal.assign(day=clean_lines_cal["invoice_date"].dt.floor("D"))
        .drop_duplicates(["customer_id", "day"])
        .sort_values(["customer_id", "day"])
    )
    gaps = days.groupby("customer_id")["day"].diff().dt.days

    out = (
        pd.DataFrame({"customer_id": days["customer_id"], "gap": gaps})
        .groupby("customer_id")["gap"]
        .agg(avg_interpurchase_days="mean", std_interpurchase_days="std")
        .reset_index()
    )
    out["avg_interpurchase_days"] = out["avg_interpurchase_days"].fillna(sentinel)
    out["std_interpurchase_days"] = out["std_interpurchase_days"].fillna(sentinel)
    return out


def _basket_features(transactions_cal: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/max order value, mean items and distinct products per order."""
    return (
        transactions_cal.groupby("customer_id")
        .agg(
            mean_order_value=("revenue", "mean"),
            std_order_value=("revenue", "std"),
            max_order_value=("revenue", "max"),
            mean_items_per_order=("quantity", "mean"),
            mean_distinct_products_per_order=("n_distinct_products", "mean"),
        )
        .reset_index()
    )


def _breadth_features(clean_lines_cal: pd.DataFrame) -> pd.DataFrame:
    """Distinct stock codes ever bought, distinct months active, Herfindahl index.

    The Herfindahl index is computed over each customer's spend concentration across
    stock_code (the dataset has no separate product-category column, so stock_code is
    used as the category proxy): sum of squared revenue shares, 1/n_products for an
    evenly-spread basket up to 1.0 for a single-product customer.
    """
    n_stock_codes = clean_lines_cal.groupby("customer_id")["stock_code"].nunique().rename("n_distinct_stock_codes")
    n_months = (
        clean_lines_cal.assign(month=clean_lines_cal["invoice_date"].dt.to_period("M"))
        .groupby("customer_id")["month"]
        .nunique()
        .rename("n_distinct_months_active")
    )

    product_revenue = clean_lines_cal.groupby(["customer_id", "stock_code"])["line_revenue"].sum()
    customer_revenue = product_revenue.groupby("customer_id").transform("sum")
    share = (product_revenue / customer_revenue).fillna(0.0)
    herfindahl = (share**2).groupby("customer_id").sum().rename("herfindahl_index")

    out = pd.concat([n_stock_codes, n_months, herfindahl], axis=1).reset_index()
    return out


def _trend_features(clean_lines_cal: pd.DataFrame, calibration_end: str, trend_window_days: int) -> pd.DataFrame:
    """revenue_trend = last-90-day revenue / this customer's own historical 90-day average.

    A decay signal: values below 1 mean the customer's most recent spend is below
    their own historical pace, above 1 means they are accelerating. Customers whose
    entire tenure is shorter than the trend window have their single window compared
    to itself and are excluded from the ratio (set to 1.0, i.e. "no evidence of decay
    yet") rather than divided by a near-zero denominator.
    """
    cal_end_ts = pd.Timestamp(calibration_end)
    window_start = cal_end_ts - pd.Timedelta(days=trend_window_days)

    total_revenue = clean_lines_cal.groupby("customer_id")["line_revenue"].sum()
    first_purchase = clean_lines_cal.groupby("customer_id")["invoice_date"].min()
    tenure_days = (cal_end_ts - first_purchase).dt.days.clip(lower=1)

    recent_revenue = (
        clean_lines_cal.loc[clean_lines_cal["invoice_date"] > window_start]
        .groupby("customer_id")["line_revenue"]
        .sum()
    )
    recent_revenue = recent_revenue.reindex(total_revenue.index, fill_value=0.0)

    historical_90d_avg_revenue = total_revenue / tenure_days * trend_window_days
    revenue_trend = recent_revenue / historical_90d_avg_revenue.replace(0, np.nan)
    revenue_trend = revenue_trend.where(tenure_days > trend_window_days, other=1.0)
    revenue_trend = revenue_trend.fillna(1.0)

    return pd.DataFrame({"customer_id": total_revenue.index, "revenue_trend_90d": revenue_trend.values})


def _returns_features(cancellations: pd.DataFrame, calibration_end: str, gross_revenue: pd.Series) -> pd.DataFrame:
    """Cancelled-order count and cancelled value as a share of calibration-window gross revenue."""
    cancellations_cal = cancellations.loc[
        cancellations["customer_id"].notna() & (cancellations["invoice_date"] <= pd.Timestamp(calibration_end))
    ].copy()
    cancellations_cal["customer_id"] = cancellations_cal["customer_id"].astype(int)
    cancellations_cal["cancelled_value"] = -(cancellations_cal["quantity"] * cancellations_cal["price"])

    agg = (
        cancellations_cal.groupby("customer_id")
        .agg(n_cancelled_orders=("invoice", "nunique"), cancelled_value=("cancelled_value", "sum"))
        .reindex(gross_revenue.index, fill_value=0.0)
    )
    agg["cancelled_value_share"] = (agg["cancelled_value"] / gross_revenue.replace(0, np.nan)).fillna(0.0)
    return agg.reset_index()


def _country_features(transactions_cal: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One customer -> most frequent calibration-window country, grouped into top-N + Other.

    Grouping (rather than target-encoding) avoids any risk of encoding the churn
    label into a feature via fold leakage, at the cost of losing some granularity --
    an acceptable trade for a dataset that is ~90% UK.
    """
    top_countries = transactions_cal["country"].value_counts().head(top_n).index
    primary_country = transactions_cal.groupby("customer_id")["country"].agg(lambda s: s.value_counts().idxmax())
    grouped = primary_country.where(primary_country.isin(top_countries), other="Other")
    dummies = pd.get_dummies(grouped, prefix="country").astype(int)
    dummies.insert(0, "customer_id", grouped.index)
    return dummies.reset_index(drop=True)


def build_features(force: bool = False) -> pd.DataFrame:
    """Assemble the full calibration-only feature matrix used to train the churn classifier.

    Joins RFM primitives/derived features, basket stats, breadth (stock-code and
    month diversity, Herfindahl concentration), a 90-day revenue trend, cancellation
    features, country encoding, and p_alive from BG/NBD. One row per customer with a
    calibration-window purchase (same population as rfm_calibration/churn_labels).
    """
    cfg = load_config()

    rfm_calibration, churn_labels = load_rfm_and_churn(force=force)
    transactions = load_transactions(force=force)
    clean_lines = load_clean_lines(force=force)
    cancellations = pd.read_parquet(cfg.paths.interim_dir / "cancellations.parquet")
    clv = load_clv(force=force)

    transactions_cal = _restrict_to_calibration(transactions, cfg.rfm.calibration_end)
    clean_lines_cal = _restrict_to_calibration(clean_lines, cfg.rfm.calibration_end)

    gross_revenue = transactions_cal.groupby("customer_id")["revenue"].sum()

    rfm_feat = _derived_rfm_features(rfm_calibration)
    interpurchase_feat = _interpurchase_features(clean_lines_cal)
    basket_feat = _basket_features(transactions_cal)
    breadth_feat = _breadth_features(clean_lines_cal)
    trend_feat = _trend_features(clean_lines_cal, cfg.rfm.calibration_end, cfg.features.trend_window_days)
    returns_feat = _returns_features(cancellations, cfg.rfm.calibration_end, gross_revenue)
    country_feat = _country_features(transactions_cal, cfg.features.top_n_countries)

    features = rfm_feat
    for part in (interpurchase_feat, basket_feat, breadth_feat, trend_feat, returns_feat, country_feat):
        features = features.merge(part, on="customer_id", how="left")

    features = features.merge(
        clv[["customer_id", "p_alive"]], on="customer_id", how="left"
    )
    features = features.merge(churn_labels[["customer_id", "churned", "is_new_cohort"]], on="customer_id", how="left")

    numeric_cols = features.select_dtypes(include="number").columns
    n_missing = features[numeric_cols].isna().sum().sum()
    if n_missing:
        logger.warning("%d missing numeric feature values after joins; filling with 0", n_missing)
        features[numeric_cols] = features[numeric_cols].fillna(0.0)

    logger.info("Feature matrix built: %d customers, %d columns", *features.shape)
    return features


def load_features(force: bool = False) -> pd.DataFrame:
    """Cached wrapper around build_features. Reads/writes dataset/processed/features.parquet."""
    cfg = load_config()
    features_path = cfg.paths.processed_dir / "features.parquet"

    if not force and features_path.exists():
        logger.info("Loading cached features from %s", features_path)
        return pd.read_parquet(features_path)

    features = build_features(force=force)
    features.to_parquet(features_path, index=False)
    logger.info("Wrote %s", features_path)
    return features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached parquet exists")
    args = parser.parse_args()

    features = load_features(force=args.force)
    print("features shape:", features.shape)
    print(features.head())
    print(features.dtypes)


if __name__ == "__main__":
    main()
