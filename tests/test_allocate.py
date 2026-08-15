import pandas as pd

from src.allocate import allocate_uniform_cost

UPLIFT = 0.2
COST = 5.0


def _table() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "customer_id": range(10),
            "clv_12m": [100, 200, 50, 0, 300, 10, 400, 5, 150, 0],
            "p_churn": [0.5, 0.1, 0.9, 0.5, 0.05, 0.9, 0.2, 0.9, 0.5, 0.9],
        }
    )
    df["expected_save"] = df["clv_12m"] * df["p_churn"] * UPLIFT - COST
    return df


def test_allocation_never_exceeds_budget():
    df = _table()
    budget = 30.0  # affords 6 customers at cost=5 each
    selected = allocate_uniform_cost(df, budget, COST)
    assert len(selected) * COST <= budget


def test_allocation_never_selects_negative_expected_save():
    df = _table()
    selected = allocate_uniform_cost(df, budget=1000.0, cost_per_customer=COST)
    assert (selected["expected_save"] > 0).all()


def test_allocation_returns_fewer_than_affordable_when_too_few_customers_are_worth_saving():
    df = pd.DataFrame({"customer_id": range(5), "clv_12m": [10.0] * 5, "p_churn": [0.01] * 5})
    df["expected_save"] = df["clv_12m"] * df["p_churn"] * UPLIFT - COST  # all negative
    selected = allocate_uniform_cost(df, budget=1000.0, cost_per_customer=COST)
    assert len(selected) == 0
