import pandas as pd

from src.load import deduplicate_overlap, split_cancellations


def _line_items(invoices, customer_ids, quantities):
    n = len(invoices)
    return pd.DataFrame(
        {
            "invoice": invoices,
            "stock_code": ["A"] * n,
            "description": ["widget"] * n,
            "quantity": quantities,
            "invoice_date": pd.to_datetime(["2010-01-01"] * n),
            "price": [1.0] * n,
            "customer_id": customer_ids,
            "country": ["United Kingdom"] * n,
        }
    )


def test_deduplicate_overlap_removes_known_duplicates():
    base = _line_items(["100", "101", "102"], [1, 2, 3], [1, 2, 3])
    overlap = base.iloc[[0, 1]]  # simulates the sheet-1/sheet-2 overlap window
    combined = pd.concat([base, overlap], ignore_index=True)

    deduped = deduplicate_overlap(combined, min_removed_warning_threshold=1)

    assert len(deduped) == 3
    assert deduped.duplicated().sum() == 0


def test_deduplicate_overlap_is_a_noop_when_no_duplicates_present():
    base = _line_items(["100", "101", "102"], [1, 2, 3], [1, 2, 3])
    deduped = deduplicate_overlap(base, min_removed_warning_threshold=1)
    assert len(deduped) == len(base)


def test_split_cancellations_routes_c_prefixed_invoices_out_of_main_stream():
    df = _line_items(["500", "C501", "502", "C503"], [1, 1, 2, 2], [1, -1, 2, -2])

    main, cancellations = split_cancellations(df)

    assert set(main["invoice"]) == {"500", "502"}
    assert set(cancellations["invoice"]) == {"C501", "C503"}
    assert len(main) + len(cancellations) == len(df)


def test_split_cancellations_handles_no_cancellations_present():
    df = _line_items(["500", "502"], [1, 2], [1, 2])
    main, cancellations = split_cancellations(df)
    assert len(main) == len(df)
    assert len(cancellations) == 0
