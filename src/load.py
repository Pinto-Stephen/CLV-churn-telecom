"""Ingest and clean the UCI Online Retail II workbook into a transaction-level table.

Reads both sheets ("Year 2009-2010", "Year 2010-2011"), removes the known overlap
between them, strips non-merchandise rows (cancellations, postage, discounts, manual
adjustments), and aggregates order lines up to one row per (customer_id, invoice) --
the purchase-event grain that BTYD models require.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

_COLUMN_RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def _fail_if_raw_missing(raw_xlsx: Path) -> None:
    if not raw_xlsx.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_xlsx}. This project does not download data "
            "automatically. Download 'Online Retail II' from the UCI Machine Learning "
            "Repository (https://archive.ics.uci.edu/dataset/502/online+retail+ii) and "
            f"place the workbook at {raw_xlsx}."
        )


def read_raw_sheets(raw_xlsx: Path, sheet_names: list[str]) -> pd.DataFrame:
    """Read and concatenate both workbook sheets, normalising columns to snake_case.

    Returns the raw union of both sheets (overlap not yet removed): one row per
    order line, columns invoice/stock_code/description/quantity/invoice_date/price/
    customer_id/country.
    """
    _fail_if_raw_missing(raw_xlsx)
    frames = []
    for sheet in sheet_names:
        logger.info("Reading sheet %r from %s", sheet, raw_xlsx)
        df = pd.read_excel(raw_xlsx, sheet_name=sheet, engine="openpyxl")
        df = df.rename(columns=_COLUMN_RENAME)
        missing = set(_COLUMN_RENAME.values()) - set(df.columns)
        if missing:
            raise ValueError(f"Sheet {sheet!r} is missing expected columns: {missing}")
        # Some stock codes are purely numeric and some are alphanumeric, so pandas can
        # infer different dtypes for this column per sheet. Force str consistently here
        # (rather than downstream) so every later step, including parquet writes, sees
        # a single dtype instead of a mix of Python int/str objects.
        df["invoice"] = df["invoice"].astype(str)
        df["stock_code"] = df["stock_code"].astype(str)
        logger.info("Sheet %r: %d rows", sheet, len(df))
        frames.append(df[list(_COLUMN_RENAME.values())])

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw rows across sheets: %d", len(combined))
    return combined


def deduplicate_overlap(df: pd.DataFrame, min_removed_warning_threshold: int) -> pd.DataFrame:
    """Drop exact duplicate rows created by the known sheet overlap.

    Sheet 1 ("Year 2009-2010") ends 2010-12-09 and sheet 2 ("Year 2010-2011") begins
    2010-12-01, so the concatenated frame contains a genuine block of duplicated order
    lines. Returns the deduplicated frame; warns if the removed count looks implausibly
    small, since that would indicate the overlap was not actually present in this read.
    """
    before = len(df)
    deduped = df.drop_duplicates(keep="first")
    removed = before - len(deduped)
    logger.info("Deduplication: %d rows before, %d after, %d exact duplicates removed", before, len(deduped), removed)
    if removed < min_removed_warning_threshold:
        logger.warning(
            "Only %d duplicate rows removed (expected at least %d from the known sheet "
            "overlap). The read may be wrong -- verify sheet names and columns.",
            removed,
            min_removed_warning_threshold,
        )
    return deduped.reset_index(drop=True)


def split_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off cancellation rows (invoice starting with 'C') from the main stream.

    Cancellations are not merchandise sales; they are persisted separately because a
    customer's cancellation/return rate is used later as a churn-risk feature. Returns
    (main_stream, cancellations).
    """
    is_cancellation = df["invoice"].astype(str).str.startswith("C")
    cancellations = df.loc[is_cancellation].reset_index(drop=True)
    main = df.loc[~is_cancellation].reset_index(drop=True)
    logger.info(
        "Cancellations split: %d cancellation rows removed, %d rows remain in main stream",
        len(cancellations),
        len(main),
    )
    return main, cancellations


def drop_null_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no customer_id -- required for customer-level modelling.

    This is expected to remove roughly 20-25% of rows. It is unavoidable: BTYD and
    churn models operate per customer, so anonymous transactions cannot contribute.
    """
    before = len(df)
    out = df.loc[df["customer_id"].notna()].reset_index(drop=True)
    removed = before - len(out)
    pct = 100 * removed / before if before else 0.0
    logger.info(
        "Null customer_id drop: %d before, %d after, %d removed (%.1f%%)",
        before,
        len(out),
        removed,
        pct,
    )
    return out


def drop_non_product_rows(df: pd.DataFrame, non_product_codes: set[str], regex: str) -> pd.DataFrame:
    """Drop rows whose stock_code denotes postage, discounts, fees or manual adjustments.

    These codes do not represent merchandise and would distort both revenue and
    per-customer purchase-event counts if left in.
    """
    before = len(df)
    code_str = df["stock_code"].astype(str)
    is_listed = code_str.isin(non_product_codes)
    is_gift_regex = code_str.str.match(regex, case=False)
    out = df.loc[~(is_listed | is_gift_regex)].reset_index(drop=True)
    logger.info(
        "Non-product stock_code drop: %d before, %d after, %d removed",
        before,
        len(out),
        before - len(out),
    )
    return out


def drop_non_positive_quantity_price(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-positive quantity or price.

    These are residual data-entry artefacts (or adjustment rows not caught by the
    stock_code filter) that cannot represent a genuine merchandise sale.
    """
    before = len(df)
    out = df.loc[(df["quantity"] > 0) & (df["price"] > 0)].reset_index(drop=True)
    logger.info(
        "Non-positive quantity/price drop: %d before, %d after, %d removed",
        before,
        len(out),
        before - len(out),
    )
    return out


def _clean_line_items(raw_xlsx: Path, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full line-item cleaning pipeline. Returns (clean_lines, cancellations)."""
    combined = read_raw_sheets(raw_xlsx, cfg.data.sheet_names)
    deduped = deduplicate_overlap(combined, cfg.data.min_duplicate_removed_warning_threshold)
    main, cancellations = split_cancellations(deduped)
    main = drop_null_customer_id(main)
    main = drop_non_product_rows(main, cfg.data.non_product_stock_codes, cfg.data.non_product_stock_code_regex)
    main = drop_non_positive_quantity_price(main)

    main = main.copy()
    main["customer_id"] = main["customer_id"].astype(int)
    main["invoice_date"] = pd.to_datetime(main["invoice_date"])
    main["line_revenue"] = main["quantity"] * main["price"]
    logger.info("Clean line items ready: %d rows, %d customers", len(main), main["customer_id"].nunique())

    return main, cancellations


def aggregate_to_transactions(clean_lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse cleaned order lines to one row per (customer_id, invoice).

    BTYD models consume purchase events, not order lines. Returns one row per
    transaction with summed revenue, summed quantity, distinct stock-code count
    (basket breadth), and the invoice date.
    """
    grouped = clean_lines.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_date=("invoice_date", "min"),
        revenue=("line_revenue", "sum"),
        quantity=("quantity", "sum"),
        n_distinct_products=("stock_code", "nunique"),
        country=("country", "first"),
    )
    logger.info(
        "Aggregated to transactions: %d transactions across %d customers",
        len(grouped),
        grouped["customer_id"].nunique(),
    )
    return grouped


def load_clean_lines(force: bool = False) -> pd.DataFrame:
    """Return cleaned order lines (pre-aggregation), for feature engineering that needs line detail.

    One row per surviving order line: customer_id, invoice, stock_code, description,
    quantity, invoice_date, price, country, line_revenue. Cached to
    dataset/interim/clean_lines.parquet alongside cancellations.parquet. Most
    downstream code should prefer load_transactions() (invoice-level); this is for
    features that need per-line detail such as distinct stock codes or category
    concentration.
    """
    cfg = load_config()
    clean_lines_path = cfg.paths.interim_dir / "clean_lines.parquet"
    cancellations_path = cfg.paths.interim_dir / "cancellations.parquet"

    if not force and clean_lines_path.exists() and cancellations_path.exists():
        logger.info("Loading cached clean line items from %s", clean_lines_path)
        return pd.read_parquet(clean_lines_path)

    clean_lines, cancellations = _clean_line_items(cfg.paths.raw_xlsx, cfg)

    cancellations.to_parquet(cancellations_path, index=False)
    logger.info("Wrote %d cancellation rows to %s", len(cancellations), cancellations_path)
    clean_lines.to_parquet(clean_lines_path, index=False)
    logger.info("Wrote %d clean line items to %s", len(clean_lines), clean_lines_path)

    return clean_lines


def load_transactions(force: bool = False) -> pd.DataFrame:
    """Return the cleaned, transaction-level table used by every downstream module.

    One row per (customer_id, invoice): customer_id, invoice, invoice_date, revenue,
    quantity, n_distinct_products, country. Cached to dataset/processed/transactions.parquet.
    Skips recomputation if the cache exists unless force=True.
    """
    cfg = load_config()
    transactions_path = cfg.paths.processed_dir / "transactions.parquet"

    if not force and transactions_path.exists():
        logger.info("Loading cached transactions from %s", transactions_path)
        return pd.read_parquet(transactions_path)

    clean_lines = load_clean_lines(force=force)
    transactions = aggregate_to_transactions(clean_lines)

    transactions.to_parquet(transactions_path, index=False)
    logger.info("Wrote %d transactions to %s", len(transactions), transactions_path)

    return transactions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached parquet exists")
    args = parser.parse_args()

    try:
        transactions = load_transactions(force=args.force)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print("transactions shape:", transactions.shape)
    print(transactions.head())
    print(transactions.dtypes)


if __name__ == "__main__":
    main()
