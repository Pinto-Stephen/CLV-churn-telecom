import numpy as np
import pandas as pd
import pytest

from src.churn import _run_track, compute_baselines, paired_bootstrap_auc_delta, prepare_dataset
from src.config import load_config


def test_majority_class_baseline_predicts_training_base_rate_not_half():
    rng = np.random.default_rng(7)
    n_train, n_test = 100, 40
    y_train = pd.Series([1] * 20 + [0] * 80)
    y_test = pd.Series(rng.integers(0, 2, size=n_test))

    def _frame(n):
        return pd.DataFrame(
            {
                "frequency": rng.integers(0, 5, size=n),
                "recency": rng.uniform(0, 50, size=n),
                "T": rng.uniform(50, 100, size=n),
                "monetary_value": rng.uniform(10, 100, size=n),
                "p_alive": rng.uniform(0, 1, size=n),
                "days_since_last_purchase": rng.uniform(0, 50, size=n),
            }
        )

    X_train, X_test = _frame(n_train), _frame(n_test)
    cfg = load_config()

    baselines, _ = compute_baselines(y_train, y_test, X_train, X_test, cfg)
    majority_brier = baselines.loc[baselines["model"] == "Majority class", "brier"].iloc[0]

    base_rate = y_train.mean()
    assert base_rate == pytest.approx(0.20)
    expected_brier = float(((y_test - base_rate) ** 2).mean())
    hardcoded_half_brier = float(((y_test - 0.5) ** 2).mean())

    assert majority_brier == pytest.approx(expected_brier)
    assert majority_brier != pytest.approx(hardcoded_half_brier)


def _synthetic_repeat_buyer_features(n: int = 220, seed: int = 2) -> pd.DataFrame:
    """Synthetic repeat-buyer-only feature matrix, all frequency > 0, exactly balanced classes."""
    rng = np.random.default_rng(seed)
    frequency = rng.integers(1, 8, size=n)
    recency = rng.uniform(1, 100, size=n)
    T = recency + rng.uniform(1, 50, size=n)
    monetary_value = rng.uniform(10, 200, size=n)
    p_alive = rng.uniform(0, 1, size=n)
    risk_score = (1 - p_alive) + rng.normal(0, 0.05, size=n)
    churned = (risk_score > np.median(risk_score)).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "frequency": frequency,
            "recency": recency,
            "T": T,
            "monetary_value": monetary_value,
            "recency_ratio": recency / T,
            "days_since_last_purchase": T - recency,
            "purchase_rate": frequency / T,
            "is_one_time_buyer": 0,
            "p_alive": p_alive,
            "churned": churned,
        }
    )


def test_repeat_buyers_only_track_returns_metrics_for_all_baseline_models_plus_xgboost():
    repeat_features = _synthetic_repeat_buyer_features()
    cfg = load_config()

    result = _run_track(repeat_features, cfg, track="repeat_buyers_only", include_increment=True)
    models = set(result["table"]["model"])

    expected = {
        "Majority class",
        "1 - p_alive (BG/NBD, no ML)",
        "Recency alone",
        "Logistic regression (RFM primitives)",
        "XGBoost (tuned, raw)",
        "XGBoost (tuned, isotonic calibrated)",
    }
    assert expected <= models
    assert result["table"]["track"].eq("repeat_buyers_only").all()
    assert result["table"]["incremental_auc_over_p_alive"].notna().sum() == len(expected) - 1


def test_prepare_dataset_excludes_only_customer_id_and_churned():
    features = _synthetic_repeat_buyer_features(n=10)
    X, y, feature_cols = prepare_dataset(features)
    assert "customer_id" not in X.columns
    assert "churned" not in X.columns
    assert set(feature_cols) == set(X.columns)
    assert y.tolist() == features["churned"].tolist()


def test_paired_bootstrap_auc_delta_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, size=n)
    scores_a = y * 0.6 + rng.normal(0, 0.3, size=n)
    scores_b = rng.normal(0, 0.3, size=n)

    result = paired_bootstrap_auc_delta(y, scores_a, scores_b, n_boot=300, seed=3)

    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["delta"] > 0
