"""BG/NBD purchase-timing model and Gamma-Gamma spend model, fit on calibration RFM.

BG/NBD suits this dataset because Online Retail II is non-contractual: customers do
not "cancel" a subscription, they simply stop ordering. BG/NBD models exactly that --
a Beta-distributed per-customer dropout probability combined with a Gamma-distributed
purchase rate -- and is the standard tool for exactly this kind of non-contractual,
repeat-purchase retail data.

Gamma-Gamma then estimates each customer's expected average transaction value,
conditioned on their observed repeat-purchase spend, independent of purchase
frequency. Combining the two via clv.utils.customer_lifetime_value gives a
probabilistic, discounted 12-month CLV per customer.

The module closes with the single most important artifact in this repo: predicted vs.
actual holdout-window purchases, bucketed by calibration frequency. If the model is
wrong, this plot shows it; if it isn't, this plot is the evidence.
"""

from __future__ import annotations

import argparse
import logging

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
from pymc_marketing.clv.utils import rfm_train_test_split

from src.config import load_config
from src.load import load_transactions
from src.rfm import load_rfm_and_churn

logger = logging.getLogger(__name__)


def fit_bgnbd(rfm_calibration: pd.DataFrame, cfg) -> BetaGeoModel:
    """Fit BG/NBD on calibration-window (frequency, recency, T).

    Uses pymc-marketing's default model_config: alpha, r ~ Prior("Weibull", ...) for
    the purchase-rate process, and a and b derived from phi_dropout ~ Uniform(0, 1)
    and kappa_dropout ~ Pareto(1, 1) for the dropout process. These are weakly
    informative, not HalfFlat -- HalfFlat priors are too diffuse to be reliable with a
    small customer base, but this dataset's ~4.9K calibration-window customers are
    comfortably large enough that the choice barely matters; we keep the informative
    defaults regardless since they are the safer choice in general.
    """
    model = BetaGeoModel(sampler_config=dict(cfg.btyd.sampler))
    model.fit(
        data=rfm_calibration[["customer_id", "frequency", "recency", "T"]],
        random_seed=cfg.seed,
    )
    logger.info("BG/NBD fit complete on %d customers", len(rfm_calibration))
    return model


def check_convergence(model: BetaGeoModel, rhat_threshold: float) -> None:
    """Log MCMC convergence diagnostics; warn loudly (without halting the run) on failure.

    Computes max R-hat across all sampled parameters and total divergence count from
    the fitted model's posterior. A failed check does not raise, because a single
    borderline diagnostic on an otherwise-converged run should not abort a multi-stage
    pipeline -- but it is never allowed to pass silently.
    """
    summary = az.summary(model.idata)
    max_rhat = float(summary["r_hat"].max())
    n_divergent = int(model.idata["sample_stats"]["diverging"].sum())

    logger.info("BG/NBD convergence: max R-hat=%.4f, divergences=%d", max_rhat, n_divergent)

    if max_rhat >= rhat_threshold:
        logger.warning(
            "*** CONVERGENCE WARNING *** max R-hat=%.4f exceeds threshold %.4f -- "
            "posterior samples may not be trustworthy. Consider more draws/tuning.",
            max_rhat,
            rhat_threshold,
        )
    if n_divergent > 0:
        logger.warning(
            "*** CONVERGENCE WARNING *** %d divergent transitions during sampling -- "
            "posterior may not have explored the full parameter space.",
            n_divergent,
        )


def fit_gamma_gamma(rfm_calibration: pd.DataFrame, cfg) -> tuple[GammaGammaModel, float]:
    """Fit Gamma-Gamma on repeat buyers with positive monetary value.

    Gamma-Gamma assumes purchase frequency and monetary value are independent given
    the customer's latent spend parameters. This is checked directly: the Pearson
    correlation between frequency and monetary_value on the training subset is logged,
    and a warning is raised if |r| > threshold, since a strong correlation would
    violate the model's core assumption.
    """
    train = rfm_calibration.loc[
        (rfm_calibration["frequency"] > 0) & (rfm_calibration["monetary_value"] > 0),
        ["customer_id", "frequency", "monetary_value"],
    ]
    corr = float(train["frequency"].corr(train["monetary_value"]))
    logger.info(
        "Gamma-Gamma independence check: Pearson r(frequency, monetary_value) = %.4f on %d repeat buyers",
        corr,
        len(train),
    )
    if abs(corr) > cfg.btyd.monetary_frequency_correlation_warn_threshold:
        logger.warning(
            "*** ASSUMPTION WARNING *** |r|=%.4f exceeds threshold %.4f -- Gamma-Gamma's "
            "independence assumption is violated; treat expected_avg_value/clv_12m as approximate.",
            abs(corr),
            cfg.btyd.monetary_frequency_correlation_warn_threshold,
        )

    model = GammaGammaModel(sampler_config=dict(cfg.btyd.sampler))
    model.fit(data=train, random_seed=cfg.seed)
    logger.info("Gamma-Gamma fit complete on %d repeat buyers", len(train))
    return model, corr


def predict_customer_metrics(
    bg_model: BetaGeoModel,
    gg_model: GammaGammaModel,
    rfm_calibration: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Produce per-customer BTYD outputs used by feature engineering and allocation.

    Returns customer_id, predicted_purchases_holdout (expected transactions over the
    holdout window), p_alive (probability still active at calibration_end),
    expected_avg_value (Gamma-Gamma conditional expected transaction value), and
    clv_12m (12-month discounted CLV, revenue scaled by the gross_margin assumption
    from config -- revenue is not profit).
    """
    holdout_length_days = (
        pd.Timestamp(cfg.rfm.observation_end) - pd.Timestamp(cfg.rfm.calibration_end)
    ).days

    bg_input = rfm_calibration[["customer_id", "frequency", "recency", "T"]]
    predicted_purchases = bg_model.expected_purchases(data=bg_input, future_t=holdout_length_days)
    p_alive = bg_model.expected_probability_alive(data=bg_input)

    gg_input = rfm_calibration[["customer_id", "frequency", "monetary_value"]]
    expected_avg_value = gg_model.expected_customer_spend(data=gg_input)

    clv_input = rfm_calibration[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    clv_revenue = gg_model.expected_customer_lifetime_value(
        transaction_model=bg_model,
        data=clv_input,
        future_t=cfg.btyd.clv_future_t_months,
        discount_rate=cfg.btyd.discount_rate_monthly,
        time_unit=cfg.btyd.time_unit,
    )

    out = pd.DataFrame(
        {
            "customer_id": rfm_calibration["customer_id"].values,
            "predicted_purchases_holdout": predicted_purchases.mean(("chain", "draw")).values,
            "p_alive": p_alive.mean(("chain", "draw")).values,
            "expected_avg_value": expected_avg_value.mean(("chain", "draw")).values,
            "clv_12m": clv_revenue.mean(("chain", "draw")).values * cfg.btyd.gross_margin,
        }
    )
    logger.info(
        "Per-customer BTYD predictions ready for %d customers (gross_margin=%.2f applied to clv_12m)",
        len(out),
        cfg.btyd.gross_margin,
    )
    return out


def _actual_holdout_purchases(cfg) -> pd.DataFrame:
    """Recompute actual holdout-window transaction counts per customer, for validation only.

    This is a cheap re-application of the same calibration/holdout split used in
    src/rfm.py (not a second source of truth) -- it exists here purely to extract the
    test_frequency column that rfm.py's public contract (rfm_calibration, churn_labels)
    deliberately does not expose downstream.
    """
    transactions = load_transactions()
    split = rfm_train_test_split(
        transactions,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        train_period_end=cfg.rfm.calibration_end,
        test_period_end=cfg.rfm.observation_end,
        time_unit="D",
    )
    return split[["customer_id", "test_frequency"]].rename(
        columns={"test_frequency": "actual_holdout_purchases"}
    )


def validate_holdout(predictions: pd.DataFrame, rfm_calibration: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compare predicted vs. actual holdout purchases, bucketed by calibration frequency.

    Writes the bucketed comparison table to reports/holdout_calibration.csv and a bar
    chart to reports/figures/holdout_calibration.png -- the plot that demonstrates the
    model was validated against held-out data, not just fitted. Also logs MAE/RMSE of
    predicted vs. actual holdout transaction counts across all customers.
    """
    actuals = _actual_holdout_purchases(cfg)
    merged = predictions.merge(rfm_calibration[["customer_id", "frequency"]], on="customer_id")
    merged = merged.merge(actuals, on="customer_id", how="left")
    merged["actual_holdout_purchases"] = merged["actual_holdout_purchases"].fillna(0)

    mae = float((merged["predicted_purchases_holdout"] - merged["actual_holdout_purchases"]).abs().mean())
    rmse = float(np.sqrt(((merged["predicted_purchases_holdout"] - merged["actual_holdout_purchases"]) ** 2).mean()))
    logger.info("Holdout validation: MAE=%.4f, RMSE=%.4f (n=%d customers)", mae, rmse, len(merged))

    bucket_edges = [0, 1, 2, 3, 4, 5, 6, 7, np.inf]
    bucket_labels = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    merged["frequency_bucket"] = pd.cut(
        merged["frequency"], bins=bucket_edges, labels=bucket_labels, right=False, include_lowest=True
    )

    table = (
        merged.groupby("frequency_bucket", observed=True)
        .agg(
            n_customers=("customer_id", "count"),
            mean_predicted=("predicted_purchases_holdout", "mean"),
            mean_actual=("actual_holdout_purchases", "mean"),
        )
        .reset_index()
    )
    table["abs_error"] = (table["mean_predicted"] - table["mean_actual"]).abs()
    logger.info("Holdout calibration table:\n%s", table.to_string(index=False))

    table_path = cfg.paths.reports_dir / "holdout_calibration.csv"
    table.to_csv(table_path, index=False)
    logger.info("Wrote %s", table_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(table))
    width = 0.35
    ax.bar(x - width / 2, table["mean_predicted"], width, label="Predicted (BG/NBD)")
    ax.bar(x + width / 2, table["mean_actual"], width, label="Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(table["frequency_bucket"])
    ax.set_xlabel("Calibration-period frequency bucket")
    ax.set_ylabel("Mean holdout-window purchases")
    ax.set_title(f"BG/NBD holdout validation: predicted vs. actual (MAE={mae:.3f}, RMSE={rmse:.3f})")
    ax.legend()
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "holdout_calibration.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)

    return table


def plot_cumulative_transactions(bg_model: BetaGeoModel, rfm_calibration: pd.DataFrame, cfg) -> None:
    """Plot cumulative actual vs. BG/NBD-predicted repeat transactions over the holdout window.

    Restricted to the same population the model predicts for (calibration-window
    customers) and day-collapsed to match the BTYD "purchase event" convention used
    everywhere else -- otherwise a customer splitting one day's order into several
    invoices would be double-counted on the actual line but not the predicted one.
    """
    holdout_length_days = (
        pd.Timestamp(cfg.rfm.observation_end) - pd.Timestamp(cfg.rfm.calibration_end)
    ).days
    calibration_customer_ids = set(rfm_calibration["customer_id"])

    transactions = load_transactions()
    holdout_tx = transactions.loc[
        (transactions["invoice_date"] > cfg.rfm.calibration_end)
        & (transactions["invoice_date"] <= cfg.rfm.observation_end)
        & (transactions["customer_id"].isin(calibration_customer_ids))
    ].copy()
    holdout_tx["day"] = holdout_tx["invoice_date"].dt.floor("D")
    holdout_tx = holdout_tx.drop_duplicates(["customer_id", "day"])
    holdout_tx["day_offset"] = (holdout_tx["day"] - pd.Timestamp(cfg.rfm.calibration_end)).dt.days
    days = np.arange(0, holdout_length_days + 1, max(1, holdout_length_days // 30))

    actual_cumulative = [int((holdout_tx["day_offset"] <= d).sum()) for d in days]

    bg_input = rfm_calibration[["customer_id", "frequency", "recency", "T"]]
    thinned_model = bg_model.thin_fit_result(keep_every=10)
    predicted_cumulative = []
    for d in days:
        t = max(d, 1e-6)
        pred = thinned_model.expected_purchases(data=bg_input, future_t=t)
        predicted_cumulative.append(float(pred.mean(("chain", "draw")).sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, actual_cumulative, label="Actual cumulative repeat transactions")
    ax.plot(days, predicted_cumulative, label="Predicted cumulative repeat transactions", linestyle="--")
    ax.set_xlabel("Days into holdout window")
    ax.set_ylabel("Cumulative repeat transactions (all customers)")
    ax.set_title("Cumulative transactions over time: actual vs. BG/NBD prediction")
    ax.legend()
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "cumulative_transactions.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def load_clv(force: bool = False) -> pd.DataFrame:
    """Cached end-to-end BTYD pipeline: fit both models, validate, and return per-customer CLV.

    Reads dataset/processed/clv.parquet if present; otherwise fits BG/NBD and
    Gamma-Gamma on calibration RFM, validates against the holdout window, and writes
    both the fitted BG/NBD model (dataset/processed/bgnbd.nc) and clv.parquet.
    """
    cfg = load_config()
    clv_path = cfg.paths.processed_dir / "clv.parquet"
    bg_path = cfg.paths.processed_dir / "bgnbd.nc"

    if not force and clv_path.exists():
        logger.info("Loading cached CLV table from %s", clv_path)
        return pd.read_parquet(clv_path)

    rfm_calibration, _ = load_rfm_and_churn(force=force)

    bg_model = fit_bgnbd(rfm_calibration, cfg)
    check_convergence(bg_model, cfg.btyd.rhat_threshold)
    bg_model.save(str(bg_path))
    logger.info("Wrote %s", bg_path)

    gg_model, _corr = fit_gamma_gamma(rfm_calibration, cfg)

    predictions = predict_customer_metrics(bg_model, gg_model, rfm_calibration, cfg)
    validate_holdout(predictions, rfm_calibration, cfg)
    plot_cumulative_transactions(bg_model, rfm_calibration, cfg)

    predictions.to_parquet(clv_path, index=False)
    logger.info("Wrote %s", clv_path)
    return predictions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached parquet exists")
    args = parser.parse_args()

    clv = load_clv(force=args.force)
    print("clv shape:", clv.shape)
    print(clv.head())
    print(clv.describe())


if __name__ == "__main__":
    main()
