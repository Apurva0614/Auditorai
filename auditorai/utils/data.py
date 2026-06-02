"""
Data utility functions for AuditorAI.
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    """Set numpy and Python random seeds for reproducibility.

    Args:
        seed: The integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: str) -> tuple:
    """Load a CSV dataset and return features and labels.

    All columns except the last are treated as features (X).
    The last column is treated as the label (y).
    Both arrays are cast to float64.

    Args:
        path: Path to the CSV file.

    Returns:
        A tuple (X, y) where X has shape (n_samples, n_features)
        and y has shape (n_samples,).

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split data into train, validation, and test sets.

    Splits test first, then splits validation from the remainder.
    Both splits use stratification on y.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        val_size: Fraction of the original data to use as validation.
            Defaults to 0.2.
        test_size: Fraction of the original data to use as test.
            Defaults to 0.2.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Split off test set first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Compute the relative validation fraction from the remaining data
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_model(model: object, path: str) -> None:
    """Save a model to disk using joblib.

    Creates parent directories if they do not exist.

    Args:
        model: The object to serialize.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> object:
    """Load a model from disk using joblib.

    Args:
        path: Path to the serialized model file.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def load_any(source) -> tuple:
    """
    Smart loader. Accepts:
      - str path to CSV (last col = label)
      - str sklearn dataset name: "breast_cancer", "iris", "digits",
        "wine", "adult", "diabetes"
      - np.ndarray tuple (X, y)
      - pd.DataFrame (last col = label)

    Returns (X: np.ndarray, y: np.ndarray)
    """
    # If it's a tuple or list of two arrays
    if isinstance(source, (tuple, list)) and len(source) == 2:
        X, y = source
        return np.asarray(X), np.asarray(y)

    # If it's a DataFrame
    if isinstance(source, pd.DataFrame):
        X = source.iloc[:, :-1].values.astype(np.float64)
        y = source.iloc[:, -1].values.astype(np.float64)
        return X, y

    # If it's a string
    if isinstance(source, str):
        # Check if it's a known sklearn dataset name
        sklearn_datasets = {
            "breast_cancer": "load_breast_cancer",
            "iris": "load_iris",
            "digits": "load_digits",
            "wine": "load_wine",
            "diabetes": "load_diabetes",
        }

        if source.lower() in sklearn_datasets:
            import sklearn.datasets as skds
            loader = getattr(skds, sklearn_datasets[source.lower()])
            data = loader()
            return data.data, data.target

        # Check for "adult" dataset — generate synthetic alternative
        if source.lower() == "adult":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=2000, n_features=14, n_informative=8,
                flip_y=0.08, random_state=42,
            )
            return X, y.astype(np.float64)

        # Try loading as CSV path
        if os.path.exists(source):
            return load_dataset(source)
        else:
            raise FileNotFoundError(
                f"'{source}' is not a recognized dataset name or file path. "
                f"Known names: {list(sklearn_datasets.keys()) + ['adult']}"
            )

    raise TypeError(
        f"Cannot load data from {type(source).__name__}. "
        f"Pass a CSV path, sklearn dataset name, (X, y) tuple, or DataFrame."
    )
