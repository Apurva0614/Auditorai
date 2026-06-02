"""
AuditorSystem: end-to-end orchestration of primary model, auditor, and router.
"""

import os

import numpy as np

from src.auditor_model import AuditorModel
from src.primary_model import PrimaryModel
from src.router import Router
from src.utils import get_logger, set_seed, split_data

logger = get_logger(__name__)


class AuditorSystem:
    """Orchestrates training, prediction, tuning, and evaluation of the full system.

    The system comprises three components:
        - PrimaryModel: makes the base classification predictions.
        - AuditorModel: predicts when the primary model is wrong.
        - Router: decides per-sample whether to show the AI prediction or
          suppress it (deferring to a human).

    Args:
        primary_model_type: Model type for PrimaryModel.
            One of "random_forest", "gradient_boosting", "logistic".
            Defaults to "random_forest".
        auditor_threshold: Suppression threshold for the auditor.
            Defaults to 0.5.
    """

    def __init__(
        self,
        primary_model_type: str = "random_forest",
        auditor_threshold: float = 0.5,
    ) -> None:
        self.primary_ = PrimaryModel(model_type=primary_model_type)
        self.auditor_ = AuditorModel(threshold=auditor_threshold)
        self.router_: Router | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train primary and auditor models, then instantiate the router.

        CRITICAL: The auditor is fitted on X_val/y_val (the validation set),
        never on X_train/y_train. The primary model memorises its training
        examples, so its error pattern on training data is unrepresentative
        of generalisation behaviour. Using the held-out validation set
        ensures the auditor sees authentic primary-model failure modes.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training labels (n_train,).
            X_val: Validation features (n_val, n_features).
            y_val: Validation labels (n_val,).
        """
        logger.info("Training primary model...")
        self.primary_.fit(X_train, y_train)

        logger.info(
            "Training auditor on held-out validation set (%d samples)...", len(X_val)
        )
        # NEVER pass X_train to auditor.fit() — the primary has memorised
        # its training data; errors there do not reflect real error modes.
        self.auditor_.fit(X_val, y_val, self.primary_)

        self.router_ = Router(
            self.primary_,
            self.auditor_,
            self.auditor_.threshold,
        )
        logger.info("AuditorSystem training complete.")

    def predict(self, X: np.ndarray) -> dict:
        """Route samples through the trained system.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            The routing dict from Router.route() with keys:
            show_mask, suppress_mask, p_wrong, ai_predictions.

        Raises:
            RuntimeError: If train() has not been called yet.
        """
        if self.router_ is None:
            raise RuntimeError(
                "AuditorSystem has not been trained. Call train() first."
            )
        return self.router_.route(X)

    def auto_tune(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        human_accuracy: float = 0.72,
    ) -> float:
        """Find and apply the threshold that maximises joint accuracy gain.

        Args:
            X_val: Validation features (n_val, n_features).
            y_val: Validation labels (n_val,).
            human_accuracy: Simulated human accuracy. Defaults to 0.72.

        Returns:
            The best tau that was selected and applied.
        """
        best_tau = self.router_.best_threshold(X_val, y_val, human_accuracy)
        self.router_.set_threshold(best_tau)
        self.auditor_.threshold = best_tau
        logger.info("Auto-tuned threshold set to %.4f", best_tau)
        return best_tau

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        human_accuracy: float = 0.72,
    ) -> dict:
        """Compute comprehensive evaluation metrics.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).
            human_accuracy: Simulated human accuracy on suppressed cases.
                Defaults to 0.72.

        Returns:
            A dict with keys:
                - "ai_only_accuracy" (float)
                - "joint_accuracy" (float)
                - "accuracy_gain" (float)
                - "suppression_rate" (float)
                - "n_shown" (int)
                - "n_suppressed" (int)
                - "auditor_auroc" (float)
                - "auditor_precision" (float)
                - "auditor_recall" (float)

        Raises:
            RuntimeError: If train() has not been called yet.
        """
        if self.router_ is None:
            raise RuntimeError(
                "AuditorSystem has not been trained. Call train() first."
            )
        result = self.router_.route(X)
        ai_predictions = result["ai_predictions"]
        suppress_mask = result["suppress_mask"]
        show_mask = result["show_mask"]

        ai_only_accuracy = float(np.mean(ai_predictions == y_true))

        n_shown = int(np.sum(show_mask))
        n_suppressed = int(np.sum(suppress_mask))
        suppression_rate = n_suppressed / len(y_true)

        if n_shown > 0:
            ai_acc_on_shown = float(
                np.mean(ai_predictions[show_mask] == y_true[show_mask])
            )
        else:
            ai_acc_on_shown = 0.0

        joint_accuracy = (
            ai_acc_on_shown * (1 - suppression_rate)
            + human_accuracy * suppression_rate
        )
        accuracy_gain = joint_accuracy - ai_only_accuracy

        auditor_metrics = self.auditor_.evaluate(X, y_true, self.primary_)

        return {
            "ai_only_accuracy": ai_only_accuracy,
            "joint_accuracy": joint_accuracy,
            "accuracy_gain": accuracy_gain,
            "suppression_rate": suppression_rate,
            "n_shown": n_shown,
            "n_suppressed": n_suppressed,
            "auditor_auroc": auditor_metrics["auroc"],
            "auditor_precision": auditor_metrics["precision"],
            "auditor_recall": auditor_metrics["recall"],
        }

    def save(self, directory: str) -> None:
        """Save primary and auditor models to directory.

        Args:
            directory: Directory path where models will be saved.
                primary.joblib and auditor.joblib will be created there.
        """
        os.makedirs(directory, exist_ok=True)
        self.primary_.save(os.path.join(directory, "primary.joblib"))
        self.auditor_.save(os.path.join(directory, "auditor.joblib"))
        logger.info("AuditorSystem saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load primary and auditor models from directory and re-instantiate router.

        Args:
            directory: Directory path containing primary.joblib and auditor.joblib.
        """
        self.primary_.load(os.path.join(directory, "primary.joblib"))
        self.auditor_.load(os.path.join(directory, "auditor.joblib"))
        self.router_ = Router(
            self.primary_,
            self.auditor_,
            self.auditor_.threshold,
        )
        logger.info("AuditorSystem loaded from %s", directory)


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    primary_model_type: str = "random_forest",
    human_accuracy: float = 0.72,
    auto_tune: bool = True,
) -> tuple:
    """Convenience function to run the complete auditor pipeline.

    Splits data, trains the system, optionally auto-tunes the threshold,
    and returns the trained system along with test splits.

    Args:
        X: Full feature array.
        y: Full label array.
        primary_model_type: Classifier type for PrimaryModel.
            Defaults to "random_forest".
        human_accuracy: Simulated human accuracy for threshold tuning.
            Defaults to 0.72.
        auto_tune: Whether to run auto-tuning on the validation set.
            Defaults to True.

    Returns:
        A tuple (system, X_test, y_test) where system is the trained
        AuditorSystem instance.
    """
    set_seed(42)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    system = AuditorSystem(primary_model_type=primary_model_type)
    system.train(X_train, y_train, X_val, y_val)
    if auto_tune:
        system.auto_tune(X_val, y_val, human_accuracy=human_accuracy)
    return system, X_test, y_test
