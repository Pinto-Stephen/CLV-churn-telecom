import pandas as pd
import pytest

from src.features import assert_no_leakage


def test_leakage_guard_rejects_a_post_calibration_date():
    df = pd.DataFrame({"invoice_date": pd.to_datetime(["2011-06-01", "2011-06-10"])})
    with pytest.raises(AssertionError):
        assert_no_leakage(df, "2011-06-09")


def test_leakage_guard_passes_when_every_date_is_on_or_before_calibration_end():
    df = pd.DataFrame({"invoice_date": pd.to_datetime(["2011-06-01", "2011-06-09"])})
    assert_no_leakage(df, "2011-06-09")  # should not raise
