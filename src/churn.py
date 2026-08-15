"""XGBoost churn classifier, benchmarked against four baselines including BG/NBD's p_alive.

The honest framing this module is built around: BG/NBD's p_alive is already a strong
churn signal on its own (it is fit on exactly the recency/frequency/T information that
drives most churn heuristics). XGBoost's contribution should therefore be judged by its
*incremental* AUC over 1 - p_alive, not reported as if the baseline were zero. That
delta -- not the raw XGBoost AUC -- is the number that justifies training a second
model at all.

allocate.py multiplies CLV by p_churn directly, so a well-ranked but poorly-calibrated
score would silently distort every downstream budget decision. Calibration (Brier
score, reliability curve, isotonic recalibration) is applied unconditionally and
reported -- raw and calibrated -- on every run, treated as a first-class requirement
here, not an afterthought gated behind a threshold.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import load_config
from src.features import load_features

logger = logging.getLogger(__name__)

RFM_PRIMITIVE_COLS = ["frequency", "recency", "T", "monetary_value"]
NON_FEATURE_COLS = ["customer_id", "churned"]


def paired_bootstrap_auc_delta(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI and two-sided p-value for AUC(scores_a) - AUC(scores_b).

    scores_a and scores_b are two scores over the *same* test set, so their AUCs are
    correlated -- an unpaired test would overstate the uncertainty in their
    difference. Each bootstrap resample draws the same customer indices for both
    scores, preserving that pairing, and the delta is recomputed per resample.
    Resamples where the resampled y has only one class (AUC undefined) are dropped
    rather than counted. The p-value is the fraction of the bootstrap distribution on
    the opposite side of zero from the point estimate, doubled for a two-sided test.
    """
    y = np.asarray(y_true)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    n = len(y)
    rng = np.random.default_rng(seed)

    point_delta = roc_auc_score(y, a) - roc_auc_score(y, b)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y[idx]
        if y_boot.min() == y_boot.max():
            continue
        deltas.append(roc_auc_score(y_boot, a[idx]) - roc_auc_score(y_boot, b[idx]))
    deltas = np.array(deltas)

    alpha = 1 - ci
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    n_side = min(int((deltas <= 0).sum()), int((deltas >= 0).sum()))
    p_value = min(1.0, 2 * (n_side + 1) / (len(deltas) + 1))

    return {
        "delta": float(point_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_boot": int(len(deltas)),
    }


def prepare_dataset(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split the feature matrix into model inputs X, target y, and the feature column list.

    Every column in features.parquet except customer_id and the churned label is used
    as a model input, including is_new_cohort (cast to int) -- excluding it would just
    force the model to re-derive "recently acquired" from low frequency/recency anyway,
    while including it explicitly lets it be inspected in SHAP/importance output.
    """
    feature_cols = [c for c in features.columns if c not in NON_FEATURE_COLS]
    X = features[feature_cols].copy()
    if "is_new_cohort" in X.columns:
        X["is_new_cohort"] = X["is_new_cohort"].astype(int)
    y = features["churned"].astype(int)
    return X, y, feature_cols


def compute_baselines(
    y_train: pd.Series,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg,
) -> tuple[pd.DataFrame, float]:
    """Score the four non-tuned-XGBoost baselines: majority class, 1-p_alive, recency, logistic regression.

    Population-agnostic: called once on the pooled population and once on the
    repeat-buyers-only subset (see train_churn_model), each with its own train/test
    split. Returns (results_table, reference_auc) -- reference_auc is the 1-p_alive
    AUC on whichever population was passed in, the number XGBoost's incremental
    contribution is measured against *within that same track*. Pooled and
    repeat-buyers-only reference_auc values are not interchangeable -- see
    train_churn_model for why the pooled one is degenerate.
    """
    rows = []

    majority_score = np.full(len(y_test), y_train.mean())
    rows.append(
        {
            "model": "Majority class",
            "auc": roc_auc_score(y_test, majority_score) if y_test.nunique() > 1 else np.nan,
            "brier": brier_score_loss(y_test, majority_score),
        }
    )

    p_alive_score = 1.0 - X_test["p_alive"].to_numpy()
    reference_auc = roc_auc_score(y_test, p_alive_score)
    rows.append({"model": "1 - p_alive (BG/NBD, no ML)", "auc": reference_auc, "brier": np.nan})

    recency_score = X_test["days_since_last_purchase"].to_numpy()
    rows.append({"model": "Recency alone", "auc": roc_auc_score(y_test, recency_score), "brier": np.nan})

    logreg = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    logreg.fit(X_train[RFM_PRIMITIVE_COLS], y_train)
    logreg_proba = logreg.predict_proba(X_test[RFM_PRIMITIVE_COLS])[:, 1]
    rows.append(
        {
            "model": "Logistic regression (RFM primitives)",
            "auc": roc_auc_score(y_test, logreg_proba),
            "brier": brier_score_loss(y_test, logreg_proba),
        }
    )

    return pd.DataFrame(rows), reference_auc


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, cfg) -> tuple[xgb.XGBClassifier, float, float]:
    """Randomized hyperparameter search for XGBClassifier with 5-fold stratified CV.

    scale_pos_weight is computed from the training base rate so the search self-adjusts
    for class imbalance; on this dataset the base rate is close to 50/50 so it lands
    near 1.0, but the computation is imbalance-agnostic. Returns (best_estimator,
    cv_mean_auc, cv_std_auc) for the winning hyperparameter combination.
    """
    base_rate = y_train.mean()
    scale_pos_weight = (1 - base_rate) / base_rate if base_rate > 0 else 1.0
    logger.info("Training set churn base rate: %.4f, scale_pos_weight=%.4f", base_rate, scale_pos_weight)

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.seed,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=cfg.churn.cv_folds, shuffle=True, random_state=cfg.seed)
    search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=cfg.churn.xgb_param_distributions,
        n_iter=cfg.churn.random_search_iterations,
        scoring="roc_auc",
        cv=cv,
        random_state=cfg.seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
    logger.info(
        "RandomizedSearchCV (%d iters, %d-fold CV): best CV AUC=%.4f +/- %.4f, params=%s",
        cfg.churn.random_search_iterations,
        cfg.churn.cv_folds,
        cv_mean,
        cv_std,
        search.best_params_,
    )
    return search.best_estimator_, cv_mean, cv_std


def _score(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {"auc": roc_auc_score(y, proba), "brier": brier_score_loss(y, proba), "proba": proba}


def calibrate_model(
    tuned_model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg,
) -> tuple[object, dict, dict]:
    """Always wrap the tuned model in isotonic CalibratedClassifierCV; always report raw AND calibrated.

    This used to be conditional on a Brier-score threshold (config.yaml's
    calibration_brier_threshold: 0.05), but that threshold was unreachable at this
    dataset's ~48% base rate -- the majority-class Brier alone is ~0.25, five times the
    threshold, so the conditional fired on every run regardless of the tuned model's
    actual calibration and was decorative. Rather than pick a new threshold value
    (arbitrary either way), calibration is now applied unconditionally -- allocate.py
    multiplies clv_12m by p_churn directly, so a well-ranked but poorly-calibrated
    probability would silently distort the budget regardless of whether some threshold
    happened to fire -- and both raw and calibrated metrics are always computed and
    reported (see build_results_table), so the effect of calibration is visible on
    every run rather than hidden behind a branch that only sometimes executes.
    """
    raw_eval = _score(tuned_model, X_test, y_test)
    logger.info("Tuned XGBoost (raw): test AUC=%.4f, Brier=%.4f", raw_eval["auc"], raw_eval["brier"])

    cv = StratifiedKFold(n_splits=cfg.churn.cv_folds, shuffle=True, random_state=cfg.seed)
    calibrated = CalibratedClassifierCV(estimator=tuned_model, method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    cal_eval = _score(calibrated, X_test, y_test)
    logger.info("Calibrated XGBoost (isotonic): test AUC=%.4f, Brier=%.4f", cal_eval["auc"], cal_eval["brier"])

    if cal_eval["brier"] >= raw_eval["brier"]:
        logger.warning(
            "Isotonic calibration did NOT improve Brier score on this track (%.4f -> %.4f) -- still using "
            "the calibrated model downstream for consistency across tracks, but this track's calibration "
            "gain is negative.",
            raw_eval["brier"],
            cal_eval["brier"],
        )
    return calibrated, raw_eval, cal_eval


def plot_calibration_curve(
    y_test: pd.Series, raw_proba: np.ndarray, calibrated_proba: np.ndarray | None, cfg
) -> None:
    """Reliability diagram: fraction of actual churners vs. predicted probability, in deciles."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Perfect calibration")

    frac_pos, mean_pred = calibration_curve(y_test, raw_proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label="XGBoost (raw)")

    if calibrated_proba is not None:
        frac_pos_c, mean_pred_c = calibration_curve(y_test, calibrated_proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred_c, frac_pos_c, marker="o", label="XGBoost (isotonic calibrated)")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual churners")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "calibration_curve.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def plot_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, cfg) -> None:
    """SHAP summary plot if shap installs cleanly on this model; permutation importance as fallback."""
    try:
        import shap

        base_estimator = model
        if isinstance(base_estimator, CalibratedClassifierCV):
            base_estimator = base_estimator.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_test)
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        fig.tight_layout()
        fig_path = cfg.paths.figures_dir / "shap_summary.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s (SHAP summary)", fig_path)
        return
    except Exception:
        logger.warning("SHAP summary plot failed; falling back to permutation importance", exc_info=True)

    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0)
    importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    importance.tail(20).plot.barh(ax=ax)
    ax.set_xlabel("Permutation importance (mean AUC drop)")
    ax.set_title("Feature importance (permutation, fallback for SHAP)")
    fig.tight_layout()
    fig_path = cfg.paths.figures_dir / "permutation_importance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s (permutation importance fallback)", fig_path)


def build_results_table(
    baselines: pd.DataFrame,
    raw_eval: dict,
    cal_eval: dict,
    reference_auc: float,
    include_increment: bool = True,
) -> pd.DataFrame:
    """Assemble the baselines-vs-XGBoost comparison table for one evaluation track.

    Both the raw and isotonic-calibrated XGBoost are always included as separate rows
    (calibrate_model() always computes both -- see its docstring for why the old
    conditional was removed), so the calibration effect is visible in every run's
    table rather than hidden behind a branch.

    include_increment controls whether an incremental_auc_over_p_alive column is
    added at all: it is meaningful only when reference_auc is a real, above-random
    signal (the repeat-buyers-only track), and actively misleading against the
    degenerate pooled 1-p_alive AUC (0.412), so the pooled track omits the column
    entirely rather than reporting a number computed against a broken reference.

    Even within a track where the column is included, the Majority class row's
    "increment" is left blank: majority class has no discriminative signal by
    construction, so auc - reference_auc for that row is just the reference AUC
    negated -- the same subtract-from-a-reference artifact that caused the original
    pooled-track bug, just inverted. It is not a claim about majority class "losing"
    to the reference by that margin.
    """
    rows = baselines.to_dict("records")
    rows.append({"model": "XGBoost (tuned, raw)", "auc": raw_eval["auc"], "brier": raw_eval["brier"]})
    rows.append(
        {"model": "XGBoost (tuned, isotonic calibrated)", "auc": cal_eval["auc"], "brier": cal_eval["brier"]}
    )
    table = pd.DataFrame(rows)
    if include_increment:
        table["incremental_auc_over_p_alive"] = table["auc"] - reference_auc
        table.loc[table["model"] == "Majority class", "incremental_auc_over_p_alive"] = np.nan
    return table


def _run_track(features: pd.DataFrame, cfg, track: str, include_increment: bool) -> dict:
    """Run the full baselines + tuned/calibrated XGBoost evaluation on one population.

    Used twice by train_churn_model: once on the pooled population, once on the
    repeat-buyers-only subset, with identical split logic, seed, and CV protocol
    (cfg.seed, cfg.churn.test_size, cfg.churn.cv_folds) -- the only thing that
    differs between calls is which rows of `features` are passed in.
    """
    X, y, feature_cols = prepare_dataset(features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.churn.test_size, stratify=y, random_state=cfg.seed
    )
    logger.info(
        "[%s] Train/test split: %d train (churn rate %.3f), %d test (churn rate %.3f)",
        track,
        len(X_train),
        y_train.mean(),
        len(X_test),
        y_test.mean(),
    )

    baselines, reference_auc = compute_baselines(y_train, y_test, X_train, X_test, cfg)
    tuned_model, cv_mean, cv_std = tune_xgboost(X_train, y_train, cfg)
    logger.info("[%s] Tuned XGBoost: %d-fold CV AUC=%.4f +/- %.4f", track, cfg.churn.cv_folds, cv_mean, cv_std)

    final_model, raw_eval, cal_eval = calibrate_model(tuned_model, X_train, y_train, X_test, y_test, cfg)

    table = build_results_table(baselines, raw_eval, cal_eval, reference_auc, include_increment=include_increment)
    table.insert(0, "track", track)

    return {
        "table": table,
        "reference_auc": reference_auc,
        "final_model": final_model,
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "raw_eval": raw_eval,
        "cal_eval": cal_eval,
        "feature_cols": feature_cols,
    }


def train_churn_model(force: bool = False) -> pd.DataFrame:
    """End-to-end churn pipeline: two evaluation tracks, tuned+calibrated XGBoost, full-population scores.

    Runs baselines + tuned XGBoost twice: once pooled (all customers) and once
    restricted to repeat buyers (calibration-window frequency > 0). The pooled
    1-p_alive AUC is degenerate -- BG/NBD assigns p_alive=1.0 identically to every
    one-time buyer, colliding with the proxy churn label -- so incremental AUC is
    only reported for the repeat-buyers-only track, against its own (real) 1-p_alive
    reference. Both tracks' metrics go to reports/churn_metrics.csv with a `track`
    column. The production model, scores, and plots are all built from the pooled
    track since downstream (allocate.py) needs a p_churn for every customer,
    including one-time buyers.

    Returns the churn_scores table (customer_id, p_churn, fold) and writes it to
    dataset/processed/churn_scores.parquet, along with the persisted model
    (dataset/processed/churn_model.joblib).
    """
    cfg = load_config()
    features = load_features(force=force)

    overall_base_rate = float(features["churned"].mean())
    one_time_base_rate = float(features.loc[features["is_one_time_buyer"] == 1, "churned"].mean())
    repeat_base_rate = float(features.loc[features["is_one_time_buyer"] == 0, "churned"].mean())
    n_one_time = int((features["is_one_time_buyer"] == 1).sum())
    n_repeat = int((features["is_one_time_buyer"] == 0).sum())
    logger.info(
        "Proxy-churn base rate: overall=%.4f (n=%d), one-time buyers=%.4f (n=%d), repeat buyers=%.4f (n=%d)",
        overall_base_rate,
        len(features),
        one_time_base_rate,
        n_one_time,
        repeat_base_rate,
        n_repeat,
    )

    pooled = _run_track(features, cfg, track="pooled", include_increment=False)

    repeat_features = features.loc[features["is_one_time_buyer"] == 0].reset_index(drop=True)
    repeat = _run_track(repeat_features, cfg, track="repeat_buyers_only", include_increment=True)

    one_time_churn_rate = float(features.loc[features["is_one_time_buyer"] == 1, "churned"].mean())
    logger.warning(
        "Pooled 1-p_alive AUC=%.4f is DEGENERATE and not a baseline to benchmark anything against: BG/NBD "
        "assigns p_alive=1.0 identically to every zero-frequency (one-time-buyer) customer -- a mathematical "
        "property of the model, not a bug -- while %.1f%% of those same one-time buyers carry a churned=1 "
        "proxy label simply because they have not had time to reorder within the 6-month holdout window. "
        "That collision is a finding about the proxy label, not about model skill. The repeat_buyers_only "
        "track (reference 1-p_alive AUC=%.4f) is the fair comparison for XGBoost's incremental contribution.",
        pooled["reference_auc"],
        one_time_churn_rate * 100,
        repeat["reference_auc"],
    )

    repeat_incremental = repeat["table"].loc[
        repeat["table"]["model"] == "XGBoost (tuned, isotonic calibrated)", "incremental_auc_over_p_alive"
    ].iloc[0]

    xgb_proba = repeat["cal_eval"]["proba"]
    p_alive_score = 1.0 - repeat["X_test"]["p_alive"].to_numpy()
    boot = paired_bootstrap_auc_delta(
        repeat["y_test"].to_numpy(), xgb_proba, p_alive_score, n_boot=2000, seed=cfg.seed
    )
    logger.info(
        "[repeat_buyers_only] Incremental AUC of tuned XGBoost over 1-p_alive baseline (AUC=%.4f): %+.4f "
        "(95%% paired bootstrap CI [%+.4f, %+.4f], n_boot=%d, p=%.4f). p_alive is itself one of the model's "
        "input features, so this delta is the incremental value of the basket/breadth/trend features layered "
        "on top of BG/NBD's signal, not XGBoost re-deriving that signal from scratch.",
        repeat["reference_auc"],
        repeat_incremental,
        boot["ci_low"],
        boot["ci_high"],
        boot["n_boot"],
        boot["p_value"],
    )
    if boot["p_value"] >= 0.05:
        logger.warning(
            "[repeat_buyers_only] The incremental AUC delta (%+.4f) is NOT significant at alpha=0.05 "
            "(p=%.4f) -- report this as 'a modest amount of signal this sample cannot sharply resolve', "
            "not as a confirmed improvement.",
            repeat_incremental,
            boot["p_value"],
        )

    recency_auc = float(repeat["table"].loc[repeat["table"]["model"] == "Recency alone", "auc"].iloc[0])
    if recency_auc < repeat["reference_auc"]:
        logger.info(
            "[repeat_buyers_only] Recency alone (AUC=%.4f) now sits BELOW 1-p_alive (AUC=%.4f) -- the reverse "
            "of the pooled ordering, where raw recency (0.767) beat the degenerate pooled p_alive (0.412). On "
            "the population BG/NBD is actually valid for, its dropout-process structure beats raw recency.",
            recency_auc,
            repeat["reference_auc"],
        )

    metrics_table = pd.concat([pooled["table"], repeat["table"]], ignore_index=True)
    metrics_table["delta_vs_p_alive_ci_low"] = np.nan
    metrics_table["delta_vs_p_alive_ci_high"] = np.nan
    metrics_table["delta_vs_p_alive_p_value"] = np.nan
    xgb_repeat_row = (metrics_table["track"] == "repeat_buyers_only") & (
        metrics_table["model"] == "XGBoost (tuned, isotonic calibrated)"
    )
    metrics_table.loc[xgb_repeat_row, "delta_vs_p_alive_ci_low"] = boot["ci_low"]
    metrics_table.loc[xgb_repeat_row, "delta_vs_p_alive_ci_high"] = boot["ci_high"]
    metrics_table.loc[xgb_repeat_row, "delta_vs_p_alive_p_value"] = boot["p_value"]
    logger.info("Churn model comparison (both tracks):\n%s", metrics_table.to_string(index=False))

    metrics_path = cfg.paths.reports_dir / "churn_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)
    logger.info("Wrote %s", metrics_path)

    plot_calibration_curve(
        pooled["y_test"],
        pooled["raw_eval"]["proba"],
        pooled["cal_eval"]["proba"],
        cfg,
    )
    plot_feature_importance(pooled["final_model"], pooled["X_test"], pooled["y_test"], cfg)

    model_path = cfg.paths.processed_dir / "churn_model.joblib"
    dump(pooled["final_model"], model_path)
    logger.info("Wrote %s", model_path)

    X_pooled = pooled["X"]
    p_churn_all = pooled["final_model"].predict_proba(X_pooled)[:, 1]
    fold = pd.Series("train", index=X_pooled.index)
    fold.loc[pooled["X_test"].index] = "test"
    churn_scores = pd.DataFrame(
        {"customer_id": features["customer_id"].values, "p_churn": p_churn_all, "fold": fold.values}
    )

    scores_path = cfg.paths.processed_dir / "churn_scores.parquet"
    churn_scores.to_parquet(scores_path, index=False)
    logger.info("Wrote %s", scores_path)

    return churn_scores


def load_churn_scores(force: bool = False) -> pd.DataFrame:
    """Cached wrapper around train_churn_model. Reads dataset/processed/churn_scores.parquet if present."""
    cfg = load_config()
    scores_path = cfg.paths.processed_dir / "churn_scores.parquet"
    if not force and scores_path.exists():
        logger.info("Loading cached churn scores from %s", scores_path)
        return pd.read_parquet(scores_path)
    return train_churn_model(force=force)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached parquet exists")
    args = parser.parse_args()

    churn_scores = load_churn_scores(force=args.force)
    print("churn_scores shape:", churn_scores.shape)
    print(churn_scores.head())
    print(churn_scores["p_churn"].describe())


if __name__ == "__main__":
    main()
