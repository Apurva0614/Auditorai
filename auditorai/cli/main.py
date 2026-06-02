"""
AuditorAI CLI — command-line interface for the universal auditor system.

Usage:
    auditorai run   --data breast_cancer --report
    auditorai sweep --data breast_cancer --steps 10
    auditorai validate --adapter-path outputs/models --data breast_cancer
"""

import argparse
import sys
import time

import numpy as np

import auditorai


BANNER = """
+======================================+
|  AuditorAI v{version:<23s}|
|  Universal AI Prediction Auditor     |
+======================================+
""".format(version=auditorai.__version__)


def _timestamp() -> str:
    """Return current time as a formatted string."""
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    """Print a timestamped log message."""
    print(f"  [{_timestamp()}] {msg}")


def _load_data(data_arg: str):
    """Load data from a path or dataset name."""
    from auditorai.utils.data import load_any
    return load_any(data_arg)


def _build_sklearn_model(model_type: str, X_train, y_train):
    """Build, fit, and wrap an sklearn model."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_map = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "logistic": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "svm": lambda: SVC(probability=True, random_state=42),
    }

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model_map["xgboost"] = lambda: XGBClassifier(
                n_estimators=100, random_state=42, use_label_encoder=False,
                eval_metric="logloss"
            )
        except ImportError:
            print("ERROR: XGBoost not installed. Install with: pip install xgboost",
                  file=sys.stderr)
            sys.exit(1)

    if model_type not in model_map:
        print(
            f"ERROR: Unknown model type '{model_type}'. "
            f"Choose from: {list(model_map.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    base_clf = model_map[model_type]()
    calibrated = CalibratedClassifierCV(base_clf, cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    pipeline.fit(X_train, y_train)
    return pipeline


def cmd_run(args) -> None:
    """Execute the 'run' subcommand."""
    print(BANNER)
    _log("Starting AuditorAI run...")

    from auditorai.utils.data import set_seed, split_data
    from auditorai import AuditorSystem, wrap
    from auditorai.core.evaluate import run_full_evaluation

    # Step 1: Set seed
    set_seed(42)
    _log("Random seed set to 42")

    # Step 2: Load data
    data_source = args.data or "breast_cancer"
    _log(f"Loading data: {data_source}")
    X, y = _load_data(data_source)
    _log(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Step 3: Split data
    _log("Splitting into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    _log(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Step 4: Build and fit primary model
    _log(f"Training primary model ({args.model_type})...")
    primary = _build_sklearn_model(args.model_type, X_train, y_train)
    adapter = wrap(primary)
    _log("Primary model trained and wrapped")

    # Step 5: Create auditor system
    _log(f"Creating AuditorSystem (threshold={args.threshold:.2f})...")
    system = AuditorSystem(adapter, auditor_threshold=args.threshold)

    # Step 6: Train auditor
    _log("Training auditor on validation set...")
    system.train(X_val, y_val)
    _log("Auditor training complete")

    # Step 7: Auto-tune
    if not args.no_tune:
        _log("Auto-tuning threshold...")
        best_tau = system.auto_tune(X_val, y_val, human_accuracy=args.human_acc)
        _log(f"Optimal threshold: tau={best_tau:.4f}")
    else:
        _log("Skipping auto-tune (--no-tune flag)")

    # Step 8: Save
    _log(f"Saving models to {args.save_dir}...")
    system.save(args.save_dir)

    # Step 9: Evaluate
    if args.report:
        _log("Running full evaluation on test set...")
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        run_full_evaluation(
            system, X_test, y_test,
            human_accuracy=args.human_acc,
            output_dir=args.output_dir,
        )
    else:
        _log("Evaluating on test set...")
        metrics = system.evaluate(X_test, y_test, human_accuracy=args.human_acc)
        print()
        print(f"  AI-only accuracy:   {metrics['ai_only_accuracy']*100:.1f}%")
        print(f"  Joint accuracy:     {metrics['joint_accuracy']*100:.1f}%")
        print(f"  Accuracy gain:      {metrics['accuracy_gain']*100:+.1f}%")
        print(f"  Suppression rate:   {metrics['suppression_rate']*100:.1f}%")
        print()

    _log("Done!")


def cmd_sweep(args) -> None:
    """Execute the 'sweep' subcommand."""
    print(BANNER)
    _log("Starting AuditorAI threshold sweep...")

    from auditorai.utils.data import set_seed, split_data
    from auditorai import AuditorSystem, wrap

    set_seed(42)

    data_source = args.data or "breast_cancer"
    _log(f"Loading data: {data_source}")
    X, y = _load_data(data_source)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    _log(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    _log(f"Training primary model ({args.model_type})...")
    primary = _build_sklearn_model(args.model_type, X_train, y_train)
    adapter = wrap(primary)

    system = AuditorSystem(adapter)
    system.train(X_val, y_val)

    taus = np.linspace(args.min_tau, args.max_tau, args.steps).tolist()
    _log(f"Sweeping {args.steps} thresholds from {args.min_tau:.2f} to {args.max_tau:.2f}...")

    df = system.router_.sweep_thresholds(
        X_test, y_test,
        human_accuracy=args.human_acc,
        taus=taus,
    )

    # Print results
    print()
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print()

    best_row = df.loc[df["accuracy_gain"].idxmax()]
    print(f"  Best: tau={best_row['tau']:.4f}, "
          f"gain={best_row['accuracy_gain']*100:+.2f}%, "
          f"suppression={best_row['suppression_rate']*100:.1f}%")

    if args.output:
        df.to_csv(args.output, index=False)
        _log(f"Sweep table saved to {args.output}")

    _log("Done!")


def cmd_validate(args) -> None:
    """Execute the 'validate' subcommand."""
    print(BANNER)
    _log("Starting AuditorAI validation...")

    from auditorai.utils.data import set_seed, split_data
    from auditorai import AuditorSystem, wrap

    set_seed(42)

    data_source = args.data
    if data_source is None:
        print("ERROR: --data is required for validate", file=sys.stderr)
        sys.exit(1)

    _log(f"Loading data: {data_source}")
    X, y = _load_data(data_source)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    _log(f"Training primary model (random_forest)...")
    primary = _build_sklearn_model("random_forest", X_train, y_train)
    adapter = wrap(primary)

    _log(f"Loading auditor from {args.adapter_path}...")
    system = AuditorSystem(adapter)
    system.load(args.adapter_path)

    _log("Evaluating on test set...")
    metrics = system.evaluate(X_test, y_test, human_accuracy=args.human_acc)

    print()
    sep = "=" * 50
    print(sep)
    print("  VALIDATION REPORT")
    print(sep)
    print(f"  AI-only accuracy:   {metrics['ai_only_accuracy']*100:.1f}%")
    print(f"  Joint accuracy:     {metrics['joint_accuracy']*100:.1f}%")
    print(f"  Accuracy gain:      {metrics['accuracy_gain']*100:+.1f}%")
    print(f"  Suppression rate:   {metrics['suppression_rate']*100:.1f}%")
    print(f"  Auditor AUROC:      {metrics['auditor_auroc']:.3f}")
    print(f"  Precision:          {metrics['auditor_precision']*100:.1f}%")
    print(f"  Recall:             {metrics['auditor_recall']*100:.1f}%")
    print(sep)

    _log("Done!")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="auditorai",
        description="AuditorAI — Universal AI Prediction Auditor",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"auditorai {auditorai.__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Train and evaluate an auditor system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--data", type=str, default=None,
        help="CSV path OR sklearn dataset name (breast_cancer, iris, digits, wine, adult)",
    )
    run_parser.add_argument(
        "--model-type", type=str, default="random_forest",
        dest="model_type",
        choices=["random_forest", "gradient_boosting", "logistic", "svm", "xgboost"],
        help="sklearn model type",
    )
    run_parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Starting suppression threshold",
    )
    run_parser.add_argument(
        "--human-acc", type=float, default=0.72,
        dest="human_acc",
        help="Human accuracy assumption",
    )
    run_parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip auto-tuning",
    )
    run_parser.add_argument(
        "--save-dir", type=str, default="outputs/models",
        dest="save_dir",
        help="Save auditor to this directory",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default="outputs",
        dest="output_dir",
        help="Save plots here",
    )
    run_parser.add_argument(
        "--report", action="store_true",
        help="Print full evaluation report with plots",
    )
    run_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress all output except the final report",
    )
    run_parser.set_defaults(func=cmd_run)

    # --- sweep ---
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Sweep suppression thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sweep_parser.add_argument(
        "--data", type=str, default=None,
        help="CSV path OR sklearn dataset name",
    )
    sweep_parser.add_argument(
        "--model-type", type=str, default="random_forest",
        dest="model_type",
        choices=["random_forest", "gradient_boosting", "logistic", "svm", "xgboost"],
        help="sklearn model type",
    )
    sweep_parser.add_argument(
        "--min-tau", type=float, default=0.1,
        dest="min_tau",
        help="Min threshold to sweep",
    )
    sweep_parser.add_argument(
        "--max-tau", type=float, default=0.9,
        dest="max_tau",
        help="Max threshold to sweep",
    )
    sweep_parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of threshold steps",
    )
    sweep_parser.add_argument(
        "--human-acc", type=float, default=0.72,
        dest="human_acc",
        help="Human accuracy assumption",
    )
    sweep_parser.add_argument(
        "--output", type=str, default=None,
        help="Save sweep table to this CSV. Default: stdout",
    )
    sweep_parser.set_defaults(func=cmd_sweep)

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a saved auditor on new data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    validate_parser.add_argument(
        "--adapter-path", type=str, default="outputs/models",
        dest="adapter_path",
        help="Path to directory containing auditor.joblib",
    )
    validate_parser.add_argument(
        "--data", type=str, default=None,
        help="CSV path OR sklearn dataset name",
    )
    validate_parser.add_argument(
        "--human-acc", type=float, default=0.72,
        dest="human_acc",
        help="Human accuracy assumption",
    )
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
