"""Reproduce and optionally validate the Strategy C submission."""

import argparse
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import tensorflow as tf
from jaxtyping import Float64

from src.dataio import load_test, load_train, save_submission
from src.model import PolynomialModel
from src.optim import SGD, Adam
from src.typedefs import DataVec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "strategy_c" / "submission.csv"
EDGES = np.linspace(0.0, 10.0, 21)
Shape: TypeAlias = Literal["linear", "quadratic"]
EdgeVec: TypeAlias = Float64[np.ndarray, " n_edges"]


def fit_cubics(x: DataVec, y: DataVec, edges: EdgeVec) -> list[PolynomialModel]:
    """Train local cubics with the original full-batch Adam settings."""
    models = []
    grams = []
    crosses = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        keep = (left - 0.1 <= x) & (x <= right + 0.1)
        if not np.any(keep):
            raise ValueError(f"No training data for [{left:g}, {right:g}].")
        target = y[keep]
        model = PolynomialModel(3, center=(left + right) / 2, scale=(right - left) / 2)
        basis = model.features(tf.convert_to_tensor(x[keep])).numpy()
        grams.append(basis.T @ basis / target.size)
        crosses.append(basis.T @ target / target.size)
        models.append(model)

    gram = tf.constant(np.stack(grams), dtype=tf.float64)
    cross = tf.constant(np.stack(crosses), dtype=tf.float64)
    weights = tf.Variable(tf.zeros((len(models), 4), dtype=tf.float64))
    optimizer = Adam([weights], lr=0.05, delta=1e-8)

    @tf.function
    def fit() -> None:
        for _ in tf.range(1_000):
            # Full-batch MSE gradient via the precomputed feature products.
            gradient = 2 * (tf.linalg.matvec(gram, weights) - cross)
            optimizer.step([gradient])

    fit()
    for model, coeffs in zip(models, tf.unstack(weights), strict=True):
        model.theta.assign(coeffs)
    return models


def fit_tail(x: DataVec, y: DataVec, cutoff: float, anchor: float, shape: Shape) -> PolynomialModel:
    """Learn one coefficient and keep the prediction continuous at the cutoff."""
    t = (x - cutoff) / 2
    feature = t - t**2 / 2 if shape == "quadratic" else t
    rms = float(np.sqrt(np.mean(feature**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("The extrapolation feature must have positive, finite RMS.")
    scaled = tf.constant(feature / rms, dtype=tf.float64)
    target = tf.constant(y, dtype=tf.float64)
    coefficient = tf.Variable(0.0, dtype=tf.float64)
    optimizer = SGD([coefficient], lr=0.1)

    @tf.function
    def fit() -> None:
        for _ in tf.range(1_200):
            residual = anchor + coefficient * scaled - target
            gradient = 2 * tf.reduce_mean(scaled * residual)
            optimizer.step([gradient])

    fit()
    b = float(coefficient.numpy()) / rms
    model = PolynomialModel(2 if shape == "quadratic" else 1, center=cutoff, scale=2.0)
    coefficients = [anchor, b, -b / 2] if shape == "quadratic" else [anchor, b]
    model.theta.assign(tf.constant(coefficients, dtype=tf.float64))
    return model


def boundary_value(last_cubic: PolynomialModel, cutoff: float) -> float:
    return float(last_cubic(tf.constant([cutoff], dtype=tf.float64))[0].numpy())


def predict(x: DataVec, models: list[PolynomialModel], tail: PolynomialModel) -> DataVec:
    if not np.all((0 <= x) & (x <= 12)):
        raise ValueError("Strategy C requires x in [0, 12].")
    inside = x <= 10
    result = np.empty_like(x, dtype=np.float64)
    indices = np.searchsorted(EDGES[1:-1], x, side="right")
    for index, model in enumerate(models):
        keep = inside & (indices == index)
        result[keep] = model(tf.convert_to_tensor(x[keep])).numpy()
    result[~inside] = tail(tf.convert_to_tensor(x[~inside])).numpy()
    return result


def compare_cutoffs(x: DataVec, y: DataVec) -> None:
    shapes: tuple[Shape, Shape] = ("quadratic", "linear")
    scores = {shape: [] for shape in shapes}
    print("cutoff\tquadratic RMSE\tlinear RMSE")
    for cutoff in (4, 5, 6, 7, 8):
        available = x < cutoff
        edges = np.array([cutoff - 0.5, cutoff])
        last = fit_cubics(x[available], y[available], edges)[0]
        anchor = boundary_value(last, cutoff)
        train = (cutoff - 2 <= x) & available
        valid = (cutoff <= x) & (x < cutoff + 2)
        xt = tf.convert_to_tensor(x[valid])
        for shape in shapes:
            tail = fit_tail(x[train], y[train], cutoff, anchor, shape)
            scores[shape].append(float(np.sqrt(np.mean((tail(xt).numpy() - y[valid]) ** 2))))
        print(f"{cutoff}\t{scores['quadratic'][-1]:.6f}\t\t{scores['linear'][-1]:.6f}")
    print(f"mean\t{np.mean(scores['quadratic']):.6f}\t\t{np.mean(scores['linear']):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="submission CSV path")
    parser.add_argument("--compare", action="store_true", help="run five virtual cutoff checks")
    args = parser.parse_args()
    _, x, y = load_train(ROOT / "data" / "train.csv")
    ids, xt = load_test(ROOT / "data" / "test.csv")
    models = fit_cubics(x, y, EDGES)
    keep = (8 <= x) & (x <= 10)
    anchor = boundary_value(models[-1], 10)
    tail = fit_tail(x[keep], y[keep], 10, anchor, "quadratic")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_submission(ids, predict(xt, models, tail), args.output)
    print(f"saved={args.output}")
    print(f"a={anchor:.10f} b={float(tail.theta[1].numpy()):.10f}")

    if args.compare:
        compare_cutoffs(x, y)


if __name__ == "__main__":
    main()
