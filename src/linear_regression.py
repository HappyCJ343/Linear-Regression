"""Train and evaluate a linear regression model on the power plant dataset.

This module follows the workflow described in the experiment guide:
1. Load the ``Folds5x2_pp.csv`` dataset that contains weather measurements and
   the corresponding electrical power output.
2. Split the dataset into training and testing subsets.
3. Train a :class:`~sklearn.linear_model.LinearRegression` model using the
   training data.
4. Evaluate the model with common regression metrics.
5. Visualise the predicted versus actual power output values.

The script exposes a simple command line interface so the experiment can be
reproduced from the terminal. Use ``python -m src.linear_regression --help`` to
inspect the available options.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATASET_FILENAME = "Folds5x2_pp.csv"


@dataclass
class ModelResult:
    """Container for model evaluation outputs."""

    model: LinearRegression
    y_test: pd.Series
    predictions: np.ndarray
    mse: float
    rmse: float
    mae: float
    r2: float


def default_dataset_path() -> Path:
    """Return the default location of the dataset within the repository."""

    return Path(__file__).resolve().parent.parent / "data" / DATASET_FILENAME


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the power plant dataset.

    Parameters
    ----------
    path:
        Path to the CSV file containing the dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with the expected feature columns (AT, V, AP, RH) and the
        target column (PE).
    """

    data = pd.read_csv(path)
    expected_columns = {"AT", "V", "AP", "RH", "PE"}
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset at {path} is missing the following required columns: {missing}"
        )

    return data


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features (X) and target (y)."""

    feature_columns: list[str] = ["AT", "V", "AP", "RH"]
    target_column = "PE"
    X = data[feature_columns]
    y = data[target_column]
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> ModelResult:
    """Train the linear regression model and evaluate it on a test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return ModelResult(
        model=model,
        y_test=y_test,
        predictions=predictions,
        mse=float(mse),
        rmse=rmse,
        mae=float(mae),
        r2=float(r2),
    )


def plot_predictions(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    save_path: Path | None = None,
    show_plot: bool = False,
) -> None:
    """Generate a scatter plot comparing true and predicted values."""

    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, color="#1f77b4", edgecolor="black", alpha=0.8)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="tab:red", label="Ideal fit")

    ax.set_title("Linear Regression: Actual vs. Predicted Power Output")
    ax.set_xlabel("Actual Power Output (PE)")
    ax.set_ylabel("Predicted Power Output (PE)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_dataset_path(),
        help=(
            "Path to the Folds5x2_pp.csv dataset. Defaults to the copy included in the "
            "repository."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of the dataset to use for testing (between 0 and 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=Path("plots/actual_vs_predicted.png"),
        help="Destination file for the generated scatter plot.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the scatter plot in a window after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running the experiment from the command line."""

    args = parse_args()

    data = load_dataset(args.data)
    X, y = split_features_and_target(data)
    result = train_model(X, y, test_size=args.test_size, random_state=args.random_state)

    print("Model coefficients:")
    for feature_name, coefficient in zip(X.columns, result.model.coef_):
        print(f"  {feature_name:>3}: {coefficient: .4f}")
    print(f"Intercept: {result.model.intercept_: .4f}")

    print("\nEvaluation metrics on the test set:")
    print(f"  Mean Squared Error (MSE): {result.mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {result.rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {result.mae:.4f}")
    print(f"  R^2 Score: {result.r2:.4f}")

    plot_predictions(
        result.y_test,
        result.predictions,
        save_path=args.save_plot,
        show_plot=args.show_plot,
    )
    print(f"\nScatter plot saved to: {args.save_plot.resolve()}")


if __name__ == "__main__":
    main()
