import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes

from analysis.piecewise_cubic import (
    LR,
    N_STEPS,
    NODE_ORDERS,
    SEED,
    SEGMENT_COLORS,
    SEGMENT_DEGREE,
    X_RANGE,
    Y_MARGIN,
    merge_small_segments,
    segment_nodes,
)
from src.dataio import load_train
from src.dataloader import FullBatchLoader
from src.loss import MSELoss
from src.model import PolynomialModel
from src.optim import Adam
from src.trainer import fit

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"
CACHE_PATH: Path = ROOT / "outputs" / "piecewise_cubic" / "compare_curves.npz"

SETTINGS: tuple[tuple[str, str, float, float], ...] = (
    ("Extrema Nodes", "extrema", 0.0, 0.0),
    ("Inflection Nodes", "inflection", 0.0, 0.0),
    ("Inflection Nodes, Min Width 0.2, Overlap 0.1", "inflection", 0.2, 0.1),
)
ZOOM_RANGE: tuple[float, float] = (7.5, 8.5)
ZOOM_MARGIN: float = 0.1


def fit_curves(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, kind: str, min_width: float, overlap: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    nodes = merge_small_segments(segment_nodes(x, y, NODE_ORDERS[kind]), min_width)
    curves = []
    for s in range(nodes.shape[0] - 1):
        left, right = nodes[s], nodes[s + 1]
        train_mask = (x >= left - overlap) & (x <= right + overlap)
        center, scale = (left + right) / 2, (right - left) / 2

        model = PolynomialModel(degree=SEGMENT_DEGREE, center=center, scale=scale)
        loader = FullBatchLoader(x[train_mask], y[train_mask], seed=SEED)
        fit(model, MSELoss(), Adam(model.parameters(), lr=LR), loader, n_epochs=N_STEPS)
        curves.append(model(tf.constant(grid)).numpy())
    return nodes, curves


def load_or_fit(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, refit: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    if CACHE_PATH.is_file() and not refit:
        cache = np.load(CACHE_PATH)
        if np.array_equal(cache["grid"], grid):
            print(f"loaded={CACHE_PATH}")
            return [(cache[f"nodes_{i}"], cache[f"curves_{i}"]) for i in range(len(SETTINGS))]

    results = []
    for _, kind, min_width, overlap in SETTINGS:
        nodes, curves = fit_curves(x, y, grid, kind, min_width, overlap)
        results.append((nodes, np.stack(curves)))

    arrays: dict[str, Any] = {"grid": grid}
    for i, (nodes, curves) in enumerate(results):
        arrays[f"nodes_{i}"] = nodes
        arrays[f"curves_{i}"] = curves
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **arrays)
    print(f"saved={CACHE_PATH}")
    return results


def draw(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    nodes: np.ndarray,
    curves: np.ndarray,
    outside: bool,
    linewidth: float,
) -> None:
    ax.scatter(x, y, s=1, alpha=0.1, color="gray")
    for s, curve in enumerate(curves):
        own = (grid >= nodes[s]) & (grid <= nodes[s + 1])
        color = SEGMENT_COLORS[s % len(SEGMENT_COLORS)]
        if outside:
            ax.plot(grid, np.where(own, np.nan, curve), color=color, linewidth=0.8, linestyle=":")
        ax.plot(
            grid,
            np.where(own, curve, np.nan),
            color=color,
            linewidth=linewidth,
            label=f"Segment {s}",
        )
    for node in nodes:
        ax.axvline(node, color="black", linewidth=0.5, linestyle=":")
    ax.axvspan(x.max(), 12.0, color="gray", alpha=0.15)


def main(refit: bool) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x, y = load_train(TRAIN_PATH)
    grid = np.linspace(X_RANGE[0], X_RANGE[1], 3000)
    zoom = (x >= ZOOM_RANGE[0]) & (x <= ZOOM_RANGE[1])
    results = load_or_fit(x, y, grid, refit)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="row", sharey="row")
    for col, ((title, *_), (nodes, curves)) in enumerate(zip(SETTINGS, results, strict=True)):
        print(f"nodes ({title}): {np.round(nodes, 4).tolist()}")

        top, bottom = axes[0, col], axes[1, col]
        draw(top, x, y, grid, nodes, curves, outside=False, linewidth=1.5)
        top.set_xlim(-0.5, 10.5)
        top.set_ylim(y.min() - Y_MARGIN, y.max() + Y_MARGIN)
        top.set_title(f"({'abc'[col]}) {title}")
        top.legend(loc="upper left", fontsize=6, ncol=2)

        draw(bottom, x, y, grid, nodes, curves, outside=True, linewidth=3.0)
        bottom.set_xlim(*ZOOM_RANGE)
        bottom.set_ylim(y[zoom].min() - ZOOM_MARGIN, y[zoom].max() + ZOOM_MARGIN)
        bottom.set_title(f"({'def'[col]}) Zoom [{ZOOM_RANGE[0]:g}, {ZOOM_RANGE[1]:g}]")

    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)

    path = FIGURES / "piecewise_cubic_compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved={path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    main(args.refit)
