from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.allocate import allocate_uniform_cost, simulate_strategies

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
    budget = 30.0
    selected = allocate_uniform_cost(df, budget, COST)
    assert len(selected) * COST <= budget


def test_allocation_never_selects_negative_expected_save():
    df = _table()
    selected = allocate_uniform_cost(df, budget=1000.0, cost_per_customer=COST)
    assert (selected["expected_save"] > 0).all()


def test_allocation_returns_fewer_than_affordable_when_too_few_customers_are_worth_saving():
    df = pd.DataFrame({"customer_id": range(5), "clv_12m": [10.0] * 5, "p_churn": [0.01] * 5})
    df["expected_save"] = df["clv_12m"] * df["p_churn"] * UPLIFT - COST
    selected = allocate_uniform_cost(df, budget=1000.0, cost_per_customer=COST)
    assert len(selected) == 0


def _fake_cfg(total_budget: float, cost_per_customer: float, uplift: float, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        allocate=SimpleNamespace(total_budget=total_budget, cost_per_customer=cost_per_customer, uplift=uplift),
        seed=seed,
    )


def test_strategy_table_includes_mean_selected_clv_and_p_churn():
    rng = np.random.default_rng(5)
    n = 40
    uplift, cost = 0.2, 5.0
    df = pd.DataFrame(
        {
            "customer_id": range(n),
            "clv_12m": rng.uniform(10, 500, size=n),
            "p_churn": rng.uniform(0.05, 0.95, size=n),
        }
    )
    df["expected_save"] = df["clv_12m"] * df["p_churn"] * uplift - cost
    cfg = _fake_cfg(total_budget=50.0, cost_per_customer=cost, uplift=uplift, seed=5)

    table = simulate_strategies(df, cfg)

    assert "mean_selected_clv" in table.columns
    assert "mean_selected_p_churn" in table.columns
    assert table["mean_selected_clv"].notna().all()
    assert table["mean_selected_p_churn"].notna().all()

    random_row = table.loc[table["strategy"] == "Random selection"].iloc[0]
    assert random_row["net_value_retained_std"] >= 0
    other_rows = table.loc[table["strategy"] != "Random selection"]
    assert other_rows["net_value_retained_std"].isna().all()
