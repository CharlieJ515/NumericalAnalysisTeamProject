import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.layout_engine import ConstrainedLayoutEngine

from analysis.overview import (
    CENTERED_HALF_WIDTH,
    CENTERED_SIZE,
    KDE_GRID_SIZE,
    REFERENCE_SEED,
    center_kde,
    kde_stats,
)
from src.dataio import load_train

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"
CACHE_PATH: Path = ROOT / "outputs" / "overview" / "overview_arrays.npz"

STAT_NAMES: tuple[str, ...] = ("std", "skew", "kurtosis", "hellinger")
X_RANGE: tuple[float, float] = (-0.5, 10.5)


def load_or_compute(x: np.ndarray, y: np.ndarray, recompute: bool) -> dict[str, np.ndarray]:
    if CACHE_PATH.is_file() and not recompute:
        cache = dict(np.load(CACHE_PATH))
        if "plot_x" in cache:
            print(f"loaded={CACHE_PATH}")
            return cache

    grid_x = np.linspace(x.min(), x.max(), KDE_GRID_SIZE[0])
    grid_y = np.linspace(y.min() - 0.5, y.max() + 0.5, KDE_GRID_SIZE[1])
    offsets = np.linspace(-CENTERED_HALF_WIDTH, CENTERED_HALF_WIDTH, CENTERED_SIZE)
    _, _, mean, stats = kde_stats(x, y, grid_x, grid_y)

    n_plot = round(KDE_GRID_SIZE[0] * (X_RANGE[1] - X_RANGE[0]) / (x.max() - x.min()))
    plot_x = np.linspace(X_RANGE[0], X_RANGE[1], n_plot)
    density, plot_conditional, plot_mean, _ = kde_stats(x, y, plot_x, grid_y)
    centered = center_kde(plot_conditional, plot_mean, grid_y, offsets)
    centered[:, (plot_x < x.min()) | (plot_x > x.max())] = 0.0

    sigma = float(np.median(stats["std"]))
    rng = np.random.default_rng(REFERENCE_SEED)
    y_reference = np.interp(x, grid_x, mean) + rng.normal(0.0, sigma, x.shape[0])
    _, _, _, reference = kde_stats(x, y_reference, grid_x, grid_y)

    arrays: dict[str, Any] = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "plot_x": plot_x,
        "offsets": offsets,
        "density": density,
        "centered": centered,
    }
    for name in STAT_NAMES:
        arrays[f"data_{name}"] = stats[name]
        arrays[f"reference_{name}"] = reference[name]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **arrays)
    print(f"saved={CACHE_PATH}")
    return arrays


def main(recompute: bool) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    arrays = load_or_compute(x_train, y_train, recompute)
    grid_x, grid_y, offsets = arrays["grid_x"], arrays["grid_y"], arrays["offsets"]
    plot_x = arrays["plot_x"]

    engine = ConstrainedLayoutEngine(w_pad=0.02, h_pad=0.02, hspace=0.0, wspace=0.0)
    fig = plt.figure(figsize=(14, 10), layout=engine)
    left, right = fig.subfigures(1, 2, squeeze=False, width_ratios=(3, 2))[0]
    scatter_ax, kde_ax, centered_ax = left.subplots(3, 1, sharex=True)
    stat_axes = right.subplots(len(STAT_NAMES), 1, sharex=True)

    scatter_ax.scatter(x_train, y_train, s=1, alpha=0.2)
    scatter_ax.set_xlim(*X_RANGE)
    scatter_ax.set_ylabel("y")
    scatter_ax.set_title("(a) Scatter Plot")
    fig.colorbar(ScalarMappable(), ax=scatter_ax, pad=0.01).ax.set_visible(False)

    image = kde_ax.imshow(
        arrays["density"],
        origin="lower",
        aspect="auto",
        extent=(plot_x[0], plot_x[-1], grid_y[0], grid_y[-1]),
    )
    fig.colorbar(image, ax=kde_ax, pad=0.01)
    kde_ax.set_ylabel("y")
    kde_ax.set_title("(b) 2D KDE")

    image = centered_ax.imshow(
        arrays["centered"],
        origin="lower",
        aspect="auto",
        extent=(plot_x[0], plot_x[-1], offsets[0], offsets[-1]),
    )
    fig.colorbar(image, ax=centered_ax, pad=0.01)
    centered_ax.set_xlim(*X_RANGE)
    centered_ax.set_xlabel("x")
    centered_ax.set_ylabel("y - mean")
    centered_ax.set_title("(c) Centered KDE")

    for ax, name in zip(stat_axes, STAT_NAMES, strict=True):
        ax.plot(grid_x, arrays[f"reference_{name}"], color="gray", label="Gaussian")
        ax.plot(grid_x, arrays[f"data_{name}"], label="Data")
        ax.set_ylabel(name.capitalize())
    stat_axes[0].set_title("(d) Gaussian Fit")
    stat_axes[0].legend()
    stat_axes[-1].set_xlabel("x")

    path = FIGURES / "overview_compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved={path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    main(args.recompute)
