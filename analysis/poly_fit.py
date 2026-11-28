from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import MODELS, build, load_config
from src.dataio import load_train

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
CONFIGS: Path = ROOT / "configs" / "exp"
FIGURES: Path = ROOT / "analysis" / "figures"

DEGREES: tuple[int, ...] = (3, 5, 8, 10, 12, 15, 18, 20)
LSTSQ_DEGREES: tuple[int, ...] = (3, 5, 8, 10, 12, 15, 18, 20, 25, 30)
COMPARE_DEGREES: tuple[int, ...] = (8, 10, 12, 15, 18, 20)
X_RANGE: tuple[float, float] = (-1.0, 13.0)
Y_MARGIN: float = 1.5


def plot_lstsq(x: np.ndarray, y: np.ndarray, grid: np.ndarray, path: Path) -> None:
    y_limits = (y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    z = (x - 5.0) / 5.0
    z_grid = (grid - 5.0) / 5.0

    fig, axes = plt.subplots(5, 2, figsize=(10, 15), sharex=True, sharey=True)
    for ax, degree in zip(axes.flat, LSTSQ_DEGREES, strict=True):
        coef = np.linalg.lstsq(np.polynomial.legendre.legvander(z, degree), y, rcond=None)[0]
        y_grid = np.polynomial.legendre.legval(z_grid, coef)
        rmse = np.sqrt(np.mean((np.polynomial.legendre.legval(z, coef) - y) ** 2))

        ax.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
        ax.plot(grid, y_grid, color="C1", label="lstsq")
        ax.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
        ax.set_ylim(*y_limits)
        ax.set_title(f"degree {degree} | rmse {rmse:.3f}")

    axes[0, 0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solve(x: np.ndarray, y: np.ndarray, grid: np.ndarray, path: Path) -> None:
    y_limits = (y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    z = (x - 5.0) / 5.0
    z_grid = (grid - 5.0) / 5.0

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    for ax, degree in zip(axes.flat, DEGREES, strict=True):
        design = np.polynomial.legendre.legvander(z, degree)
        coef = np.linalg.solve(design.T @ design, design.T @ y)
        y_grid = np.polynomial.legendre.legval(z_grid, coef)
        rmse = np.sqrt(np.mean((np.polynomial.legendre.legval(z, coef) - y) ** 2))

        ax.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
        ax.plot(grid, y_grid, color="C1", label="solve")
        ax.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
        ax.set_ylim(*y_limits)
        ax.set_title(f"degree {degree} | rmse {rmse:.3f}")

    axes[0, 0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(x: np.ndarray, y: np.ndarray, grid: np.ndarray, path: Path) -> None:
    y_limits = (y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    z = (x - 5.0) / 5.0
    z_grid = (grid - 5.0) / 5.0

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True, sharey=True)
    for ax, degree in zip(axes.flat, COMPARE_DEGREES, strict=True):
        config = load_config(CONFIGS / f"poly_d{degree}.toml")
        model = build(MODELS, config["model"])
        model.load(ROOT / config["paths"]["out"] / config["paths"]["checkpoint"])
        y_model = model(tf.constant(grid)).numpy()

        design = np.polynomial.legendre.legvander(z, degree)
        coef = np.linalg.solve(design.T @ design, design.T @ y)
        y_solve = np.polynomial.legendre.legval(z_grid, coef)

        ax.scatter(x, y, s=1, alpha=0.1, color="gray")
        ax.plot(grid, y_model, color="C1", label="Adam")
        ax.plot(grid, y_solve, color="C0", label="np.linalg.solve")
        ax.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
        ax.set_xlim(-1.0, 11.0)
        ax.set_ylim(*y_limits)
        ax.set_title(f"Degree {degree}")

    axes[0, 0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    grid = np.linspace(X_RANGE[0], X_RANGE[1], 1000)
    y_limits = (y_train.min() - Y_MARGIN, y_train.max() + Y_MARGIN)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    for ax, degree in zip(axes.flat, DEGREES, strict=True):
        config = load_config(CONFIGS / f"poly_d{degree}.toml")
        model = build(MODELS, config["model"])
        model.load(ROOT / config["paths"]["out"] / config["paths"]["checkpoint"])
        y_grid = model(tf.constant(grid)).numpy()

        ax.scatter(x_train, y_train, s=1, alpha=0.1, color="gray", label="data")
        ax.plot(grid, y_grid, color="C1", label="model")
        ax.axvspan(x_train.max(), 12.0, color="gray", alpha=0.15)
        ax.set_ylim(*y_limits)
        ax.set_title(f"degree {degree}")

    axes[0, 0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout()

    path = FIGURES / "poly_fit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved={path}")

    lstsq_path = FIGURES / "poly_fit_lstsq.png"
    plot_lstsq(x_train, y_train, grid, lstsq_path)
    print(f"saved={lstsq_path}")

    solve_path = FIGURES / "poly_fit_solve.png"
    plot_solve(x_train, y_train, grid, solve_path)
    print(f"saved={solve_path}")

    compare_path = FIGURES / "poly_fit_compare.png"
    plot_compare(x_train, y_train, grid, compare_path)
    print(f"saved={compare_path}")


if __name__ == "__main__":
    main()
