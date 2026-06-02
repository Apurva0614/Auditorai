"""
Tests for AuditorDriftDetector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from auditorai.core.auditor import AuditorModel
from auditorai.adapters.sklearn_adapter import SklearnAdapter
from auditorai.monitor import AuditorDriftDetector


@pytest.fixture
def trained_setup():
    # Generate data
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, flip_y=0.1, random_state=42
    )
    # Train primary
    primary = RandomForestClassifier(n_estimators=50, random_state=42)
    primary.fit(X[:100], y[:100])
    adapter = SklearnAdapter(primary)

    # Train auditor
    auditor = AuditorModel(threshold=0.4)
    auditor.fit(X[100:], y[100:], adapter)

    # Reference data
    ref_X = X[100:]
    ref_y_true = y[100:]
    ref_y_pred = adapter.predict(ref_X)
    ref_data = (ref_X, ref_y_true, ref_y_pred)

    return auditor, ref_data, adapter


def test_drift_detector_no_drift(trained_setup):
    auditor, ref_data, adapter = trained_setup
    detector = AuditorDriftDetector(
        auditor=auditor, reference_data=ref_data, primary_model=adapter, drift_threshold=0.15
    )

    # Similar new data
    new_X, new_y = make_classification(
        n_samples=100, n_features=5, n_informative=3, flip_y=0.1, random_state=42
    )
    new_y_pred = adapter.predict(new_X)

    results = detector.check(new_X, new_y_pred)
    assert "flag_rate_reference" in results
    assert "flag_rate_new" in results
    assert "drift_detected" in results
    assert results["drift_detected"] is False


def test_drift_detector_with_drift(trained_setup):
    auditor, ref_data, adapter = trained_setup
    # Set a small threshold to trigger drift
    detector = AuditorDriftDetector(
        auditor=auditor, reference_data=ref_data, primary_model=adapter, drift_threshold=0.01
    )

    # Different distribution data (noise flip_y=0.9 will result in high error rate, hence higher suppression rate)
    new_X, new_y = make_classification(
        n_samples=100, n_features=5, n_informative=3, flip_y=0.9, random_state=100
    )
    new_y_pred = adapter.predict(new_X)

    results = detector.check(new_X, new_y_pred)
    # Since drift_threshold is extremely small (0.01), drift should be detected
    assert results["drift_detected"] is True


def test_drift_detector_configurable_threshold(trained_setup):
    auditor, ref_data, adapter = trained_setup
    detector = AuditorDriftDetector(
        auditor=auditor, reference_data=ref_data, primary_model=adapter, drift_threshold=0.5
    )

    # Generate drifted data
    new_X, new_y = make_classification(
        n_samples=100, n_features=5, n_informative=3, flip_y=0.9, random_state=100
    )
    new_y_pred = adapter.predict(new_X)

    results = detector.check(new_X, new_y_pred)
    # With a high threshold of 0.5, drift is not detected unless difference exceeds 0.5
    assert results["drift_detected"] is False
