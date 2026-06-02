"""
Drift detection utility for AuditorAI.
"""

import numpy as np
from auditorai.core.auditor import AuditorModel


class AuditorDriftDetector:
    """
    Detects drift in the auditor's flag (suppression) rate between a reference
    dataset and new incoming production data.
    """

    def __init__(
        self,
        auditor: AuditorModel,
        reference_data: tuple,
        primary_model: object,
        drift_threshold: float = 0.15,
    ):
        """
        Args:
            auditor: A trained AuditorModel.
            reference_data: A tuple (X, y_true, y_pred) representing the reference set.
            primary_model: The ModelAdapter or compatible model object.
            drift_threshold: Configurable threshold for detecting drift. Defaults to 0.15.
        """
        self.auditor = auditor
        self.primary_model = primary_model
        self.drift_threshold = drift_threshold

        X_ref, y_true_ref, y_pred_ref = reference_data
        ref_suppressed = self.auditor.predict_suppression(X_ref, self.primary_model)
        self.flag_rate_reference = float(np.mean(ref_suppressed))

    def check(self, new_X: np.ndarray, new_y_pred: np.ndarray) -> dict:
        """
        Computes the auditor's flag rate on the new data and compares it to the reference rate.

        Args:
            new_X: New feature array.
            new_y_pred: Predicted labels for the new data.

        Returns:
            A dict with keys:
                - "flag_rate_reference" (float)
                - "flag_rate_new" (float)
                - "drift_detected" (bool)
        """
        new_suppressed = self.auditor.predict_suppression(new_X, self.primary_model)
        flag_rate_new = float(np.mean(new_suppressed))
        drift_detected = bool(
            abs(flag_rate_new - self.flag_rate_reference) > self.drift_threshold
        )

        return {
            "flag_rate_reference": self.flag_rate_reference,
            "flag_rate_new": flag_rate_new,
            "drift_detected": drift_detected,
        }
