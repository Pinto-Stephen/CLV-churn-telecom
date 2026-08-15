"""Risk-weighted retention budget allocation.

This is the module that turns two separately-fit models (BTYD CLV, XGBoost churn
probability) into a spending decision. Everything upstream exists to produce
clv_12m and p_churn per customer; this module is where those numbers become a
ranked list of who to spend a fixed retention budget on, and why that ranking beats
the two obvious naive alternatives (spend on your most valuable customers, or spend
on your highest-risk customers).

Core quantities, per customer:
  value_at_risk  = clv_12m * p_churn                       -- expected value lost if nothing is done
  expected_save  = clv_12m * p_churn * uplift - cost        -- expected net gain from intervening

uplift (the assumed relative reduction in churn probability from an intervention) is
an assumption, not something estimable from this dataset -- there was no randomised
holdout test of any retention offer. The sensitivity analysis in this module exists
specifically to bracket the answer instead of hiding that assumption.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.btyd import load_clv
from src.churn import load_churn_scores
from src.config import load_config

logger = logging.getLogger(__name__)


def build_allocation_table(force: bool = False) -> pd.DataFrame:
    """Join CLV and churn scores; compute value_at_risk and expected_save per customer.

    Returns one row per customer with clv_12m, p_churn, value_at_risk, and
    expected_save (using the uniform cost_per_customer and uplift from config).
    """
    cfg = load_config()
    clv = load_clv(force=force)
    churn_scores = load_churn_scores(force=force)
    df = clv.merge(churn_scores, on="customer_id", how="inner")
    logger.info(
        "Allocation table: %d customers (clv=%d rows, churn_scores=%d rows)", len(df), len(clv), len(churn_scores)
    )

    df["value_at_risk"] = df["clv_12m"] * df["p_churn"]
    df["expected_save"] = df["clv_12m"] * df["p_churn"] * cfg.allocate.uplift - cfg.allocate.cost_per_customer
    return df


def allocate_uniform_cost(df: pd.DataFrame, budget: float, cost_per_customer: float) -> pd.DataFrame:
    """Select customers to fund under a fixed budget, uniform per-customer cost.

    This is a sort, not an optimiser: with every customer costing the same amount,
    ranking by expected_save and taking the top floor(budget/cost) is provably optimal
    for maximizing total expected_save under the budget constraint -- it is the trivial
    case of the knapsack problem where all item weights are equal. Only customers with
    expected_save > 0 are eligible; if the budget affords more than that, the smaller
    set is returned and the leftover budget is logged, since spending the full budget
    on customers who are not worth saving is not the objective.
    """
    eligible = df.loc[df["expected_save"] > 0].sort_values("expected_save", ascending=False)
    n_affordable = int(budget // cost_per_customer)
    selected = eligible.head(n_affordable)
    spent = len(selected) * cost_per_customer
    leftover = budget - spent

    if len(eligible) < n_affordable:
        logger.info(
            "Only %d customers have positive expected_save (budget affords %d) -- spending %.2f of %.2f "
            "budget, %.2f left over. Spending the full budget is not the objective.",
            len(eligible),
            n_affordable,
            spent,
            budget,
            leftover,
        )
    else:
        logger.info("Uniform-cost allocation: selected %d customers, spent %.2f of %.2f budget", len(selected), spent, budget)

    return selected


def allocate_variable_cost_knapsack(df: pd.DataFrame, budget: float, cost_col: str, expected_save_col: str) -> pd.DataFrame:
    """0/1 knapsack via greedy expected_save/cost ratio ordering (approximate, not exact).

    Exact 0/1 knapsack is NP-hard; ranking by the expected_save/cost ratio and greedily
    filling the budget is the standard approximation (exact for the fractional
    relaxation of the problem). It is acceptable here specifically because individual
    customer costs are tiny relative to the total budget -- with thousands of eligible
    customers each costing a few pounds against a budget in the thousands, the "last
    item doesn't quite fit" rounding error that makes greedy provably suboptimal in the
    worst case is negligible in relative terms.
    """
    d = df.copy()
    d["ratio"] = d[expected_save_col] / d[cost_col]
    eligible = d.loc[d[expected_save_col] > 0].sort_values("ratio", ascending=False)
    cum_cost = eligible[cost_col].cumsum()
    selected = eligible.loc[cum_cost <= budget]
    spent = float(selected[cost_col].sum())
    leftover = budget - spent
    logger.info(
        "Variable-cost knapsack (greedy ratio): selected %d of %d eligible customers, spent %.2f of %.2f "
        "budget, %.2f left over",
        len(selected),
        len(eligible),
        spent,
        budget,
        leftover,
    )
    return selected


def add_variable_cost(df: pd.DataFrame, cost_per_customer: float) -> pd.DataFrame:
    """Illustrative variable per-customer outreach cost, for demonstrating the knapsack path.

    There is no real per-customer retention-outreach cost in this dataset. This scales
    the uniform base cost mildly by predicted_purchases_holdout as a proxy for account
    complexity (a customer expected to transact more often plausibly needs more
    touchpoints to retain) -- explicitly a demonstration assumption, not a fitted cost
    model, and should be labelled as such wherever it is reported.
    """
    d = df.copy()
    d["variable_cost"] = cost_per_customer * (1.0 + 0.2 * np.log1p(d["predicted_purchases_holdout"].clip(lower=0)))
    return d


def simulate_strategies(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compare total net expected value retained under the same budget across four strategies.

    Random selection, top-CLV-only, top-churn-risk-only, and risk-weighted
    expected_save ranking all select the same number of customers (budget /
    cost_per_customer) but rank them differently. The gap between the naive strategies
    and risk-weighted ranking is the business case for this whole project: top-CLV
    wastes budget on loyal customers who were never going to leave, and top-churn
    wastes budget on cheap-to-flag-as-risky but low-value customers.
    """
    budget = cfg.allocate.total_budget
    cost = cfg.allocate.cost_per_customer
    uplift = cfg.allocate.uplift
    n_afford = min(int(budget // cost), len(df))

    def net_value(selected: pd.DataFrame) -> float:
        gross = float((selected["clv_12m"] * selected["p_churn"] * uplift).sum())
        spent = len(selected) * cost
        return gross - spent

    rng = np.random.default_rng(cfg.seed)
    random_draws = [net_value(df.sample(n=n_afford, random_state=int(rng.integers(0, 1_000_000)))) for _ in range(100)]
    random_value = float(np.mean(random_draws))

    top_clv = df.sort_values("clv_12m", ascending=False).head(n_afford)
    top_churn = df.sort_values("p_churn", ascending=False).head(n_afford)
    risk_weighted = allocate_uniform_cost(df, budget, cost)

    table = pd.DataFrame(
        [
            {"strategy": "Random selection", "net_value_retained": random_value},
            {"strategy": "Top-CLV only (ignores risk)", "net_value_retained": net_value(top_clv)},
            {"strategy": "Top-churn-risk only (ignores value)", "net_value_retained": net_value(top_churn)},
            {"strategy": "Risk-weighted expected_save", "net_value_retained": net_value(risk_weighted)},
        ]
    )
    logger.info("Strategy comparison (budget=%.0f, cost/customer=%.2f):\n%s", budget, cost, table.to_string(index=False))
    return table


def plot_strategy_comparison(table: pd.DataFrame, cfg) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table["strategy"], table["net_value_retained"])
    ax.set_ylabel("Net expected value retained")
    ax.set_title("Retention budget allocation: strategy comparison")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "allocation_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def sensitivity_analysis(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sweep uplift x cost_per_customer and tabulate net value retained under the fixed budget.

    uplift cannot be estimated from this data -- there was no randomised holdout test
    of a retention offer. This sweep exists to bracket the plausible range of outcomes
    given that uncertainty, rather than presenting a single point estimate as if it
    were known. See the README for why a real experiment is the only rigorous fix.
    """
    budget = cfg.allocate.total_budget
    uplifts = cfg.allocate.uplift_sweep
    costs = cfg.allocate.cost_sweep

    grid = np.zeros((len(uplifts), len(costs)))
    for i, uplift in enumerate(uplifts):
        for j, cost in enumerate(costs):
            d = df[["customer_id", "clv_12m", "p_churn"]].copy()
            d["expected_save"] = d["clv_12m"] * d["p_churn"] * uplift - cost
            selected = allocate_uniform_cost(d, budget, cost)
            gross = float((selected["clv_12m"] * selected["p_churn"] * uplift).sum())
            spent = len(selected) * cost
            grid[i, j] = gross - spent

    table = pd.DataFrame(grid, index=[f"uplift={u:.2f}" for u in uplifts], columns=[f"cost={c:.1f}" for c in costs])
    return table


def plot_sensitivity_heatmap(table: pd.DataFrame, cfg) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(table.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:,.0f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Net value retained: sensitivity to uplift and cost assumptions")
    fig.colorbar(im, ax=ax, label="Net expected value retained")
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "allocation_sensitivity_heatmap.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


_ACTION_MAP = {
    ("High", "High"): "Priority retention outreach -- high value, high risk",
    ("High", "Medium"): "Proactive check-in -- protect a valuable relationship showing some risk",
    ("High", "Low"): "Loyalty rewards / monitor -- valuable and stable, low urgency",
    ("Medium", "High"): "Targeted save offer -- moderate value genuinely at risk",
    ("Medium", "Medium"): "Standard lifecycle nurture",
    ("Medium", "Low"): "Light-touch engagement",
    ("Low", "High"): "Low-cost automated save offer, or deprioritize",
    ("Low", "Medium"): "Monitor only",
    ("Low", "Low"): "No action needed",
}


def build_segment_summary(df: pd.DataFrame, n_terciles: int) -> pd.DataFrame:
    """Cross CLV tercile x churn-risk tercile: counts, mean CLV, mean p_churn, recommended action."""
    labels = ["Low", "Medium", "High"][:n_terciles]
    d = df.copy()
    d["clv_tercile"] = pd.qcut(d["clv_12m"], n_terciles, labels=labels, duplicates="drop")
    d["risk_tercile"] = pd.qcut(d["p_churn"], n_terciles, labels=labels, duplicates="drop")

    summary = (
        d.groupby(["clv_tercile", "risk_tercile"], observed=True)
        .agg(n_customers=("customer_id", "count"), mean_clv=("clv_12m", "mean"), mean_p_churn=("p_churn", "mean"))
        .reset_index()
    )
    summary["recommended_action"] = summary.apply(
        lambda r: _ACTION_MAP.get((str(r["clv_tercile"]), str(r["risk_tercile"])), "Review manually"), axis=1
    )
    return summary


def run_allocation(force: bool = False) -> dict[str, pd.DataFrame]:
    """End-to-end allocation pipeline: build table, allocate both ways, compare strategies, write reports.

    Writes reports/allocation.csv (full per-customer table with both allocation flags),
    reports/segment_summary.csv, reports/allocation_strategy_comparison.csv,
    reports/allocation_sensitivity.csv, and three figures under reports/figures/.
    """
    cfg = load_config()
    df = build_allocation_table(force=force)

    uniform_selected = allocate_uniform_cost(df, cfg.allocate.total_budget, cfg.allocate.cost_per_customer)
    df["selected_uniform_cost"] = df["customer_id"].isin(uniform_selected["customer_id"])

    df_variable = add_variable_cost(df, cfg.allocate.cost_per_customer)
    df_variable["expected_save_variable_cost"] = (
        df_variable["clv_12m"] * df_variable["p_churn"] * cfg.allocate.uplift - df_variable["variable_cost"]
    )
    knapsack_selected = allocate_variable_cost_knapsack(
        df_variable, cfg.allocate.total_budget, "variable_cost", "expected_save_variable_cost"
    )
    df["variable_cost"] = df_variable["variable_cost"]
    df["expected_save_variable_cost"] = df_variable["expected_save_variable_cost"]
    df["selected_variable_cost_knapsack"] = df["customer_id"].isin(knapsack_selected["customer_id"])

    strategy_table = simulate_strategies(df, cfg)
    plot_strategy_comparison(strategy_table, cfg)

    sensitivity_table = sensitivity_analysis(df, cfg)
    plot_sensitivity_heatmap(sensitivity_table, cfg)

    segment_summary = build_segment_summary(df, cfg.allocate.n_terciles)
    logger.info("Segment summary:\n%s", segment_summary.to_string(index=False))

    allocation_path = cfg.paths.reports_dir / "allocation.csv"
    df.to_csv(allocation_path, index=False)
    logger.info("Wrote %s", allocation_path)

    segment_path = cfg.paths.reports_dir / "segment_summary.csv"
    segment_summary.to_csv(segment_path, index=False)
    logger.info("Wrote %s", segment_path)

    strategy_path = cfg.paths.reports_dir / "allocation_strategy_comparison.csv"
    strategy_table.to_csv(strategy_path, index=False)
    logger.info("Wrote %s", strategy_path)

    sensitivity_path = cfg.paths.reports_dir / "allocation_sensitivity.csv"
    sensitivity_table.to_csv(sensitivity_path)
    logger.info("Wrote %s", sensitivity_path)

    return {
        "allocation": df,
        "segment_summary": segment_summary,
        "strategy_comparison": strategy_table,
        "sensitivity": sensitivity_table,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute upstream caches even if they exist")
    args = parser.parse_args()

    results = run_allocation(force=args.force)
    df = results["allocation"]
    print("allocation shape:", df.shape)
    print(df[["customer_id", "clv_12m", "p_churn", "value_at_risk", "expected_save", "selected_uniform_cost"]].head())
    print()
    print("strategy comparison:")
    print(results["strategy_comparison"])
    print()
    print("segment summary:")
    print(results["segment_summary"])


if __name__ == "__main__":
    main()
