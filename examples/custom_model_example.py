"""
AuditorAI — Custom model example
Shows how to wrap ANY model by subclassing ModelAdapter directly.
Run: python examples/custom_model_example.py

This is the escape hatch for any model not covered by the built-in
adapters (sklearn, PyTorch, HuggingFace, OpenAI, Anthropic).

Pattern:
  1. Subclass ModelAdapter
  2. Implement predict(X) -> class labels
  3. Implement predict_proba(X) -> probability matrix
  4. Pass your adapter to AuditorSystem
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from auditorai import AuditorSystem, ModelAdapter, run_full_evaluation


# ── Step 1: Your custom model ────────────────────────────────────
# This can be ANYTHING: a REST API, a custom C++ model, a lookup table,
# an ensemble of models, etc. Here we use a simple random model
# to demonstrate the pattern without any external dependency.
class MyCustomModel:
    """A fake model that makes predictions based on simple thresholds."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.weights = None

    def fit(self, X, y):
        """Fit: just learn the mean per feature."""
        self.weights = X.mean(axis=0)
        return self

    def raw_predict(self, X):
        """Return a raw score per sample."""
        # Simple dot product with learned weights
        scores = X @ self.weights
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores


# ── Step 2: Write a custom adapter ───────────────────────────────
class MyCustomAdapter(ModelAdapter):
    """
    Adapter wrapping MyCustomModel for use with AuditorAI.

    This is the minimal interface you need to implement:
      - predict(X) -> np.ndarray of class labels, shape (n,)
      - predict_proba(X) -> np.ndarray of probabilities, shape (n, n_classes)

    The adapter is responsible for converting your model's raw output
    into calibrated probabilities.
    """

    def __init__(self, model: MyCustomModel):
        self.model = model

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels."""
        scores = self.model.raw_predict(X)
        return (scores > 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probabilities, shape (n, 2).
        We convert the raw score to a 2-class probability vector.
        """
        scores = self.model.raw_predict(X)
        # Build probability matrix: [P(class=0), P(class=1)]
        probas = np.column_stack([1 - scores, scores])
        # Validate (optional but recommended)
        self.validate_probas(probas)
        return probas


# ── Step 3: Use it with AuditorAI ────────────────────────────────
if __name__ == "__main__":
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Fit your custom model
    my_model = MyCustomModel(seed=42)
    my_model.fit(X_train, y_train)

    # Wrap it in your custom adapter
    adapter = MyCustomAdapter(my_model)

    # Create the auditor system — works exactly like sklearn/pytorch/etc
    system = AuditorSystem(adapter)
    system.train(X_val, y_val)
    system.auto_tune(X_val, y_val)

    # Run full evaluation
    run_full_evaluation(system, X_test, y_test, output_dir="outputs/custom")
    print("\nCustom model example complete! Check outputs/custom/ for plots.")
