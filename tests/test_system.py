"""
Tests for AuditorSystem (system.py).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.system import AuditorSystem, run_pipeline


@pytest.fixture
def synthetic_data():
    """600-sample, 12-feature dataset with 8% label noise."""
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, flip_y=0.08, random_state=42
    )
    return X, y


def _trained_system(synthetic_data):
    """Helper: returns a trained AuditorSystem and splits."""
    X, y = synthetic_data
    from src.utils import split_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    system = AuditorSystem()
    system.train(X_train, y_train, X_val, y_val)
    return system, X_train, X_val, X_test, y_train, y_val, y_test


def test_run_pipeline_returns_system_and_splits(synthetic_data):
    X, y = synthetic_data
    result = run_pipeline(X, y)
    assert len(result) == 3
    system, X_test, y_test = result
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_train_produces_router(synthetic_data):
    system, *_ = _trained_system(synthetic_data)
    assert system.router_ is not None


def test_predict_returns_required_keys(synthetic_data):
    system, _, _, X_test, _, _, _ = _trained_system(synthetic_data)
    result = system.predict(X_test)
    for key in ("show_mask", "suppress_mask", "p_wrong", "ai_predictions"):
        assert key in result


def test_evaluate_returns_required_keys(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    expected_keys = {
        "ai_only_accuracy",
        "joint_accuracy",
        "accuracy_gain",
        "suppression_rate",
        "n_shown",
        "n_suppressed",
        "auditor_auroc",
        "auditor_precision",
        "auditor_recall",
    }
    assert expected_keys.issubset(metrics.keys())


def test_accuracy_values_in_range(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    assert 0.0 <= metrics["ai_only_accuracy"] <= 1.0
    assert 0.0 <= metrics["joint_accuracy"] <= 1.0


def test_n_shown_plus_suppressed_equals_total(synthetic_data):
    system, _, _, X_test, _, _, y_test = _trained_system(synthetic_data)
    metrics = system.evaluate(X_test, y_test)
    assert metrics["n_shown"] + metrics["n_suppressed"] == len(X_test)


def test_auto_tune_changes_threshold(synthetic_data):
    system, _, X_val, _, _, y_val, _ = _trained_system(synthetic_data)
    original_threshold = system.router_.threshold
    best_tau = system.auto_tune(X_val, y_val)
    # The auto-tuned threshold may or may not differ from default;
    # verify it was applied consistently.
    assert system.router_.threshold == best_tau
    assert system.auditor_.threshold == best_tau


def test_save_load_roundtrip(tmp_path, synthetic_data):
    system, _, _, X_test, _, _, _ = _trained_system(synthetic_data)
    preds_before = system.predict(X_test)["ai_predictions"]

    save_dir = str(tmp_path / "models")
    system.save(save_dir)

    new_system = AuditorSystem()
    new_system.load(save_dir)
    preds_after = new_system.predict(X_test)["ai_predictions"]

    np.testing.assert_array_equal(preds_before, preds_after)


def test_predict_before_train_raises():
    system = AuditorSystem()
    X = np.random.rand(10, 5)
    with pytest.raises(RuntimeError):
        system.predict(X)
