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
from scipy.stats import chi2_contingency, spearmanr

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
    """Compare what four rankings select under the same budget -- NOT a measured-lift comparison.

    Random selection, top-CLV-only, top-churn-risk-only, and risk-weighted
    expected_save ranking all select the same number of customers (budget /
    cost_per_customer) but rank them differently.

    IMPORTANT (see README): net_value_retained is computed from clv_12m * p_churn *
    uplift, the exact quantity the risk-weighted strategy is built to maximise. A
    strategy cannot lose a contest scored on its own objective function, so
    risk-weighted "winning" here is arithmetic, not empirical evidence of realised
    lift -- that can only come from the randomised holdout test described in the
    caveats.

    What this table legitimately shows is what each ranking *selects*, and
    mean_selected_clv/mean_selected_p_churn together make both naive strategies'
    failure modes symmetric and explained rather than one quantified and one merely
    asserted:
      - Top-churn-risk-only selects low-CLV customers (little value to protect in the
        first place -- see the CLV/risk collapse in clv_risk_dependence.txt), so a
        flat per-customer cost consumes most of their small expected return.
      - Top-CLV-only selects low-p_churn customers (at rho=-0.885 between clv_12m and
        p_churn, the highest-CLV customers are almost by construction the
        lowest-risk), so there is little expected loss left to prevent even though
        the customers themselves are valuable.
    Both are the same underlying collapse, viewed from opposite ends of the ranking.

    net_value_retained for Random selection is a mean over 100 resamples; its std is
    also reported so "risk-weighted beats random" can be read against the spread of
    the baseline, not just its point estimate.
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
    random_net_draws = []
    random_clv_draws = []
    random_p_churn_draws = []
    for _ in range(100):
        sample = df.sample(n=n_afford, random_state=int(rng.integers(0, 1_000_000)))
        random_net_draws.append(net_value(sample))
        random_clv_draws.append(float(sample["clv_12m"].mean()))
        random_p_churn_draws.append(float(sample["p_churn"].mean()))
    random_value = float(np.mean(random_net_draws))
    random_value_std = float(np.std(random_net_draws, ddof=1))
    random_mean_clv = float(np.mean(random_clv_draws))
    random_mean_p_churn = float(np.mean(random_p_churn_draws))

    top_clv = df.sort_values("clv_12m", ascending=False).head(n_afford)
    top_churn = df.sort_values("p_churn", ascending=False).head(n_afford)
    risk_weighted = allocate_uniform_cost(df, budget, cost)

    table = pd.DataFrame(
        [
            {
                "strategy": "Random selection",
                "net_value_retained": random_value,
                "net_value_retained_std": random_value_std,
                "mean_selected_clv": random_mean_clv,
                "mean_selected_p_churn": random_mean_p_churn,
            },
            {
                "strategy": "Top-CLV only (ignores risk)",
                "net_value_retained": net_value(top_clv),
                "net_value_retained_std": np.nan,
                "mean_selected_clv": float(top_clv["clv_12m"].mean()),
                "mean_selected_p_churn": float(top_clv["p_churn"].mean()),
            },
            {
                "strategy": "Top-churn-risk only (ignores value)",
                "net_value_retained": net_value(top_churn),
                "net_value_retained_std": np.nan,
                "mean_selected_clv": float(top_churn["clv_12m"].mean()),
                "mean_selected_p_churn": float(top_churn["p_churn"].mean()),
            },
            {
                "strategy": "Risk-weighted expected_save",
                "net_value_retained": net_value(risk_weighted),
                "net_value_retained_std": np.nan,
                "mean_selected_clv": float(risk_weighted["clv_12m"].mean()),
                "mean_selected_p_churn": float(risk_weighted["p_churn"].mean()),
            },
        ]
    )
    logger.info(
        "What each ranking selects (budget=%.0f, cost/customer=%.2f) -- net_value_retained is the "
        "project's own objective, NOT a measured lift; see docstring:\n%s",
        budget,
        cost,
        table.to_string(index=False),
    )
    return table


def write_strategy_comparison_notes(table: pd.DataFrame, cfg) -> None:
    """Write reports/allocation_strategy_notes.txt: the circularity caveat and both naive-strategy failure modes.

    This is the prose the README's strategy-comparison section is required to carry
    (see Fix 3) persisted as a pipeline artifact rather than living only in a
    docstring, a log line, or a figure caption -- so it survives independently of
    which of those a reader (or a later README rewrite) happens to consult.
    """
    random_row = table.loc[table["strategy"] == "Random selection"].iloc[0]
    top_clv_row = table.loc[table["strategy"] == "Top-CLV only (ignores risk)"].iloc[0]
    top_churn_row = table.loc[table["strategy"] == "Top-churn-risk only (ignores value)"].iloc[0]
    risk_weighted_row = table.loc[table["strategy"] == "Risk-weighted expected_save"].iloc[0]

    lines = [
        "Strategy comparison: what this table is, and is not, evidence of",
        "===================================================================",
        "",
        "NOT evidence of realised lift. net_value_retained is computed from "
        "clv_12m * p_churn * uplift, the exact objective the risk-weighted strategy is "
        "built to maximise. A strategy cannot lose a contest scored on its own objective "
        "function, so risk-weighted 'winning' this comparison is arithmetic, not an "
        "empirical finding. The only rigorous way to measure real lift is the randomised "
        "holdout test described in the README caveats (offer the intervention to a random "
        "half of the customers this ranking would select, withhold it from the other half, "
        "measure the actual difference in realised churn).",
        "",
        f"Random selection baseline: mean net_value_retained = {random_row['net_value_retained']:,.2f} "
        f"+/- {random_row['net_value_retained_std']:,.2f} (std over 100 resamples). Read the "
        f"risk-weighted vs. random gap against this spread, not just the point estimate.",
        "",
        "What this table IS legitimate evidence of: what each ranking selects. The two naive "
        "strategies fail in symmetric, and now quantified, ways:",
        "",
        f"  Top-churn-risk-only selects low-value customers: mean_selected_clv = "
        f"£{top_churn_row['mean_selected_clv']:,.2f}, vs. £{random_row['mean_selected_clv']:,.2f} "
        f"for random and £{risk_weighted_row['mean_selected_clv']:,.2f} for risk-weighted. There is "
        "little value to protect in the first place (see the CLV/risk collapse in "
        "clv_risk_dependence.txt), so the flat per-customer cost consumes most of their small "
        "expected return -- which is why this strategy nets BELOW random selection "
        f"({top_churn_row['net_value_retained']:,.2f} vs. {random_row['net_value_retained']:,.2f}).",
        "",
        f"  Top-CLV-only selects low-risk customers: mean_selected_p_churn = "
        f"{top_clv_row['mean_selected_p_churn']:.4f}, vs. {random_row['mean_selected_p_churn']:.4f} "
        f"for random and {risk_weighted_row['mean_selected_p_churn']:.4f} for risk-weighted. At "
        "rho=-0.885 between clv_12m and p_churn (Fix 2), the highest-CLV customers are almost by "
        "construction the lowest-risk ones, so despite selecting customers worth "
        f"£{top_clv_row['mean_selected_clv']:,.2f} on average (nearly 1.8x the risk-weighted "
        f"strategy's £{risk_weighted_row['mean_selected_clv']:,.2f}), there is little expected "
        "loss left to prevent.",
        "",
        "Both failure modes are the same underlying CLV/risk collapse, viewed from opposite ends "
        "of the ranking -- not two unrelated phenomena.",
    ]
    report_path = cfg.paths.reports_dir / "allocation_strategy_notes.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


def plot_strategy_comparison(table: pd.DataFrame, cfg) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(table["strategy"], table["net_value_retained"])
    ax.set_ylabel("Net value by the risk-weighted objective (not a measured lift)")
    ax.set_title("What each ranking selects, under the same budget")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.text(
        0.5,
        0.01,
        "Scored on the risk-weighted strategy's own objective (clv_12m x p_churn x uplift) -- "
        "its 'win' here is arithmetic, not measured lift. See README.",
        ha="center",
        fontsize=8,
        style="italic",
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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


def compute_clv_risk_dependence(df: pd.DataFrame, segment_summary: pd.DataFrame, n_terciles: int) -> dict:
    """Test whether clv_12m and p_churn are independent axes, or largely the same signal.

    The segment table's cell counts (10 in High-CLV/High-risk, 1,359 and 1,261 on the
    anti-diagonal) only look wrong if you expected independence -- under independence
    every cell of a 3x3 tercile cross-tab would hold roughly n/9. This makes that
    intuition explicit: Spearman rank correlation between clv_12m and p_churn directly
    (sign and magnitude of the association, tercile-boundary-independent), plus a
    chi-square test of independence on the observed 3x3 tercile counts against what
    independence would predict. Both are expected to reject independence hard, because
    clv_12m and p_churn are both downstream of the same recency/frequency signal.
    """
    labels = ["Low", "Medium", "High"][:n_terciles]

    corr, corr_p = spearmanr(df["clv_12m"], df["p_churn"])

    contingency = (
        segment_summary.pivot(index="clv_tercile", columns="risk_tercile", values="n_customers")
        .reindex(index=labels, columns=labels, fill_value=0)
    )
    chi2, chi2_p, dof, expected = chi2_contingency(contingency.to_numpy())
    expected_df = pd.DataFrame(expected, index=labels, columns=labels)

    return {
        "spearman_r": float(corr),
        "spearman_p": float(corr_p),
        "chi2": float(chi2),
        "chi2_p": float(chi2_p),
        "chi2_dof": int(dof),
        "observed": contingency,
        "expected_under_independence": expected_df,
    }


def _format_p(p: float) -> str:
    """Format a p-value as a bound rather than a spurious floating-point-underflow figure.

    At n~5000 both tests here always reject; the exact tiny p is a statement about
    float64 underflow, not about the data. '< 0.001' carries the real information
    (rejects at any conventional threshold) without implying false precision.
    """
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def compute_mechanism_correlations(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Spearman correlation of clv_12m and p_churn against the RFM primitives that likely drive both.

    Establishing that clv_12m and p_churn collapse onto one axis (rho=-0.885) shows
    THAT they are not independent. This is the mechanism check for WHY: if both load
    heavily on the same underlying primitives (recency_ratio, purchase_rate), the
    collapse is structural -- both quantities are functions of the same
    recency/frequency signal by construction, not a coincidence specific to this
    sample -- which is the stronger and more useful claim to make in the README.
    """
    from src.features import load_features

    primitives = load_features(force=force)[["customer_id", "recency_ratio", "purchase_rate"]]
    merged = df.merge(primitives, on="customer_id", how="inner")

    rows = []
    for target in ("clv_12m", "p_churn"):
        for primitive in ("recency_ratio", "purchase_rate"):
            r, p = spearmanr(merged[target], merged[primitive])
            rows.append({"target": target, "primitive": primitive, "spearman_r": float(r), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def write_clv_risk_dependence_report(dependence: dict, mechanism: pd.DataFrame, cfg) -> None:
    """Write reports/clv_risk_dependence.txt: dependence stats, mechanism correlations, observed-vs-expected."""
    mechanism_lines = [
        f"  {row.target} vs {row.primitive}: rho = {row.spearman_r:+.4f} (p {_format_p(row.spearman_p)})"
        for row in mechanism.itertuples()
    ]

    lines = [
        "CLV vs. churn-risk dependence check",
        "====================================",
        "",
        f"Spearman rank correlation (clv_12m, p_churn): rho = {dependence['spearman_r']:.4f} "
        f"(p {_format_p(dependence['spearman_p'])})",
        f"Chi-square test of independence on the {dependence['observed'].shape[0]}x{dependence['observed'].shape[1]} "
        f"CLV-tercile x risk-tercile contingency table: "
        f"chi2 = {dependence['chi2']:.2f}, dof = {dependence['chi2_dof']} (p {_format_p(dependence['chi2_p'])})",
        "",
        "The coefficient is the finding, not the p-value: at n~5000 both tests reject independence "
        "at any conventional threshold regardless of effect size, so p only confirms the association "
        "is not sampling noise. rho = -0.885 is what says clv_12m and p_churn are, for practical "
        "purposes, close to the same variable measured twice -- far beyond ordinary correlation.",
        "",
        "Mechanism: both targets against the RFM primitives that plausibly drive both",
        "-----------------------------------------------------------------------------",
        *mechanism_lines,
        "",
        "If both clv_12m and p_churn load heavily on recency_ratio and purchase_rate, the CLV/risk "
        "collapse above is structural -- both are functions of the same recency/frequency signal by "
        "construction (BG/NBD's p_alive and clv_12m are literally fit on frequency/recency/T; the churn "
        "features are engineered from the same calibration-window purchase history) -- not a "
        "coincidence specific to this sample. This is expected behaviour for non-contractual "
        "transactional retail data, not a data quality problem.",
        "",
        "Observed customer counts (CLV tercile x risk tercile):",
        dependence["observed"].to_string(),
        "",
        "Expected counts under independence (same marginals, no association):",
        dependence["expected_under_independence"].round(1).to_string(),
    ]
    report_path = cfg.paths.reports_dir / "clv_risk_dependence.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


def plot_clv_vs_risk(df: pd.DataFrame, cfg) -> None:
    """Hexbin of clv_12m (log scale) against p_churn -- makes the CLV/risk collapse visible.

    A scatter of ~4,900 points would overplot badly; hexbin shows point density instead
    and makes the same pattern the segment-table cell counts imply (few customers are
    simultaneously high-CLV and high-risk) visible directly, rather than something a
    reader has to infer from a 3x3 table.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    clv_floor = df["clv_12m"].clip(lower=1.0)
    hb = ax.hexbin(df["p_churn"], clv_floor, yscale="log", gridsize=40, cmap="viridis", mincnt=1)
    ax.set_xlabel("p_churn (predicted churn probability)")
    ax.set_ylabel("clv_12m (log scale, £)")
    ax.set_title("CLV vs. churn risk: not independent axes in this dataset")
    fig.colorbar(hb, ax=ax, label="Customer count")
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "clv_vs_risk.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


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
    write_strategy_comparison_notes(strategy_table, cfg)
    plot_strategy_comparison(strategy_table, cfg)

    sensitivity_table = sensitivity_analysis(df, cfg)
    plot_sensitivity_heatmap(sensitivity_table, cfg)

    segment_summary = build_segment_summary(df, cfg.allocate.n_terciles)
    logger.info("Segment summary:\n%s", segment_summary.to_string(index=False))

    dependence = compute_clv_risk_dependence(df, segment_summary, cfg.allocate.n_terciles)
    mechanism = compute_mechanism_correlations(df, force=force)
    logger.info(
        "FINDING: CLV and churn risk collapse onto one axis in this dataset -- rho=%.4f between clv_12m "
        "and p_churn directly (chi-square=%.2f, dof=%d, both tests reject independence at p %s). The "
        "high-CLV/high-risk quadrant that motivates risk-weighted targeting holds only %d customers.",
        dependence["spearman_r"],
        dependence["chi2"],
        dependence["chi2_dof"],
        _format_p(max(dependence["spearman_p"], dependence["chi2_p"])),
        int(segment_summary.loc[
            (segment_summary["clv_tercile"] == "High") & (segment_summary["risk_tercile"] == "High"),
            "n_customers",
        ].iloc[0]),
    )
    write_clv_risk_dependence_report(dependence, mechanism, cfg)
    plot_clv_vs_risk(df, cfg)

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
