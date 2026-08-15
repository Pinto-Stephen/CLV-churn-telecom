"""XGBoost churn classifier, benchmarked against four baselines including BG/NBD's p_alive.

The honest framing this module is built around: BG/NBD's p_alive is already a strong
churn signal on its own (it is fit on exactly the recency/frequency/T information that
drives most churn heuristics). XGBoost's contribution should therefore be judged by its
*incremental* AUC over 1 - p_alive, not reported as if the baseline were zero. That
delta -- not the raw XGBoost AUC -- is the number that justifies training a second
model at all.

allocate.py multiplies CLV by p_churn directly, so a well-ranked but poorly-calibrated
score would silently distort every downstream budget decision. Calibration (Brier
score, reliability curve, and isotonic recalibration if needed) is treated as a
first-class requirement here, not an afterthought.
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

    Returns (results_table, p_alive_auc) -- p_alive_auc is returned separately because
    it is the reference point XGBoost's incremental contribution is measured against.
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
    p_alive_auc = roc_auc_score(y_test, p_alive_score)
    rows.append({"model": "1 - p_alive (BG/NBD, no ML)", "auc": p_alive_auc, "brier": np.nan})

    if "is_one_time_buyer" in X_test.columns and X_test["is_one_time_buyer"].nunique() > 1:
        repeat_mask = X_test["is_one_time_buyer"] == 0
        if repeat_mask.sum() > 0 and y_test[repeat_mask].nunique() > 1:
            repeat_auc = roc_auc_score(y_test[repeat_mask], 1.0 - X_test.loc[repeat_mask, "p_alive"])
            logger.info(
                "1-p_alive AUC=%.4f pooled across all test customers, but %.4f restricted to repeat "
                "buyers only -- BG/NBD assigns p_alive=1.0 identically to every zero-frequency customer "
                "(a mathematical property of the model, not a bug), and most one-time buyers are labelled "
                "'churned' by the 6-month proxy simply because they have not had time to reorder yet. "
                "That collision is the clearest empirical illustration of the proxy-label caveat in this project.",
                p_alive_auc,
                repeat_auc,
            )

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

    return pd.DataFrame(rows), p_alive_auc


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
        n_jobs=1,  # process-based parallelism (loky) is unreliable in this sandboxed
        # Windows environment; the search is small enough (~150 fits on <4K rows) that
        # sequential execution finishes in well under a minute anyway.
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


def calibrate_if_needed(
    tuned_model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg,
) -> tuple[object, dict, dict | None]:
    """Evaluate the tuned model's calibration; wrap in isotonic CalibratedClassifierCV if Brier score is poor.

    allocate.py multiplies clv_12m by p_churn directly, so a well-ranked but
    poorly-calibrated probability would still distort the retention budget. Returns
    (final_model, raw_eval, calibrated_eval_or_None).
    """
    raw_eval = _score(tuned_model, X_test, y_test)
    logger.info("Tuned XGBoost (raw): test AUC=%.4f, Brier=%.4f", raw_eval["auc"], raw_eval["brier"])

    if raw_eval["brier"] <= cfg.churn.calibration_brier_threshold:
        logger.info(
            "Brier score %.4f is within threshold %.4f -- no calibration wrapper needed",
            raw_eval["brier"],
            cfg.churn.calibration_brier_threshold,
        )
        return tuned_model, raw_eval, None

    logger.warning(
        "Brier score %.4f exceeds threshold %.4f -- wrapping in CalibratedClassifierCV(isotonic)",
        raw_eval["brier"],
        cfg.churn.calibration_brier_threshold,
    )
    cv = StratifiedKFold(n_splits=cfg.churn.cv_folds, shuffle=True, random_state=cfg.seed)
    calibrated = CalibratedClassifierCV(estimator=tuned_model, method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    cal_eval = _score(calibrated, X_test, y_test)
    logger.info("Calibrated XGBoost (isotonic): test AUC=%.4f, Brier=%.4f", cal_eval["auc"], cal_eval["brier"])
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


def build_results_table(baselines: pd.DataFrame, raw_eval: dict, cal_eval: dict | None, p_alive_auc: float) -> pd.DataFrame:
    """Assemble the final baselines-vs-XGBoost comparison table with the incremental-AUC framing."""
    xgb_label = "XGBoost (tuned, isotonic calibrated)" if cal_eval is not None else "XGBoost (tuned)"
    final_eval = cal_eval if cal_eval is not None else raw_eval
    rows = baselines.to_dict("records")
    rows.append({"model": xgb_label, "auc": final_eval["auc"], "brier": final_eval["brier"]})
    table = pd.DataFrame(rows)
    table["incremental_auc_over_p_alive"] = table["auc"] - p_alive_auc
    return table


def train_churn_model(force: bool = False) -> pd.DataFrame:
    """End-to-end churn pipeline: baselines, tuned+calibrated XGBoost, and full-population scores.

    Returns the churn_scores table (customer_id, p_churn, fold) and writes it to
    dataset/processed/churn_scores.parquet, along with the persisted model
    (dataset/processed/churn_model.joblib) and reports/churn_baselines.csv.
    """
    cfg = load_config()
    features = load_features(force=force)
    X, y, feature_cols = prepare_dataset(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.churn.test_size, stratify=y, random_state=cfg.seed
    )
    logger.info(
        "Train/test split: %d train (churn rate %.3f), %d test (churn rate %.3f)",
        len(X_train),
        y_train.mean(),
        len(X_test),
        y_test.mean(),
    )

    baselines, p_alive_auc = compute_baselines(y_train, y_test, X_train, X_test, cfg)
    tuned_model, cv_mean, cv_std = tune_xgboost(X_train, y_train, cfg)
    logger.info("Tuned XGBoost: %d-fold CV AUC=%.4f +/- %.4f", cfg.churn.cv_folds, cv_mean, cv_std)

    final_model, raw_eval, cal_eval = calibrate_if_needed(tuned_model, X_train, y_train, X_test, y_test, cfg)

    results_table = build_results_table(baselines, raw_eval, cal_eval, p_alive_auc)
    logger.info("Churn model comparison:\n%s", results_table.to_string(index=False))
    incremental = results_table.loc[results_table["model"].str.startswith("XGBoost"), "incremental_auc_over_p_alive"].iloc[0]
    logger.info(
        "Incremental AUC of tuned XGBoost over 1-p_alive baseline (AUC=%.4f): %+.4f",
        p_alive_auc,
        incremental,
    )

    results_path = cfg.paths.reports_dir / "churn_baselines.csv"
    results_table.to_csv(results_path, index=False)
    logger.info("Wrote %s", results_path)

    plot_calibration_curve(y_test, raw_eval["proba"], cal_eval["proba"] if cal_eval else None, cfg)
    plot_feature_importance(final_model, X_test, y_test, cfg)

    model_path = cfg.paths.processed_dir / "churn_model.joblib"
    dump(final_model, model_path)
    logger.info("Wrote %s", model_path)

    p_churn_all = final_model.predict_proba(X)[:, 1]
    fold = pd.Series("train", index=X.index)
    fold.loc[X_test.index] = "test"
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
