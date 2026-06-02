"""
AuditorAI — sklearn example
Works with any sklearn-compatible model.
Run: python examples/sklearn_example.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from auditorai import AuditorSystem, wrap, run_full_evaluation

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Train your model
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Wrap and audit — 3 lines
adapter = wrap(model)
system = AuditorSystem(adapter)
system.train(X_val, y_val)

# Auto-tune the suppression threshold
system.auto_tune(X_val, y_val)

# Run full evaluation with report + plots
run_full_evaluation(system, X_test, y_test, output_dir="outputs/sklearn")
print("\nSklearn example complete! Check outputs/sklearn/ for plots.")
