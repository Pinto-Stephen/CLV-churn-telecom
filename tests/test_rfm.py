"""Hand-verified RFM computations for synthetic customers, plus a real-data invariant check.

These synthetic cases were verified by hand against the transaction dates before being
encoded as assertions (see the session notes: customer 12346 in the real dataset was
cross-checked the same way). They exercise the two conventions most likely to be
implemented wrong: a one-time buyer has frequency=0 (not 1), and multiple invoices by
the same customer on the same calendar day collapse to a single purchase event.
"""

import pandas as pd
from pymc_marketing.clv.utils import rfm_train_test_split

from src.rfm import load_rfm_and_churn

CALIBRATION_END = "2011-04-01"
OBSERVATION_END = "2011-05-01"


def _synthetic_transactions() -> pd.DataFrame:
    rows = [
        (1, "2011-01-01", 50.0),  # customer 1: single purchase, ever -> one-time buyer
        (2, "2011-02-01", 30.0),  # customer 2: two invoices same day...
        (2, "2011-02-01", 20.0),  # ...should collapse to one purchase event...
        (2, "2011-03-01", 40.0),  # ...then a genuine second, later purchase
        (3, "2011-01-10", 10.0),  # customer 3: three purchases on three distinct days
        (3, "2011-02-10", 20.0),
        (3, "2011-03-10", 30.0),
        (99, "2011-04-15", 5.0),  # unrelated holdout-window purchase, purely so the
        # holdout window is non-empty (rfm_train_test_split requires at least one
        # transaction after train_period_end); customer 99 is not asserted on.
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "invoice_date", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def _split() -> pd.DataFrame:
    return rfm_train_test_split(
        _synthetic_transactions(),
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        train_period_end=CALIBRATION_END,
        test_period_end=OBSERVATION_END,
        time_unit="D",
        monetary_value_col="revenue",
    )


def test_one_time_buyer_has_zero_frequency_and_zero_recency():
    row = _split().set_index("customer_id").loc[1]
    assert row["frequency"] == 0
    assert row["recency"] == 0
    assert row["T"] == (pd.Timestamp(CALIBRATION_END) - pd.Timestamp("2011-01-01")).days
    assert row["monetary_value"] == 0  # no repeat transactions to average


def test_same_day_invoices_collapse_to_one_purchase_event():
    row = _split().set_index("customer_id").loc[2]
    # Two invoices on 2011-02-01 collapse to a single purchase event; the later,
    # distinct-day purchase on 2011-03-01 is therefore the *only* repeat purchase.
    assert row["frequency"] == 1
    assert row["recency"] == (pd.Timestamp("2011-03-01") - pd.Timestamp("2011-02-01")).days
    assert row["monetary_value"] == 40.0  # mean of the single repeat transaction's revenue


def test_repeat_buyer_on_distinct_days_hand_computed():
    row = _split().set_index("customer_id").loc[3]
    assert row["frequency"] == 2
    assert row["recency"] == (pd.Timestamp("2011-03-10") - pd.Timestamp("2011-01-10")).days
    assert row["T"] == (pd.Timestamp(CALIBRATION_END) - pd.Timestamp("2011-01-10")).days
    assert row["monetary_value"] == 25.0  # mean of the two repeat transactions (20, 30)


def test_t_geq_recency_holds_across_real_pipeline_output():
    rfm_calibration, _ = load_rfm_and_churn(force=False)
    assert (rfm_calibration["T"] >= rfm_calibration["recency"]).all()
