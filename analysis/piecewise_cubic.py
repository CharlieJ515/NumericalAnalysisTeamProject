import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.polynomial import legendre

from analysis.stationary_spline import CENTER, DEGREE, SCALE
from src.dataio import load_test, load_train, save_submission
from src.dataloader import FullBatchLoader
from src.loss import MSELoss
from src.model import PolynomialModel
from src.optim import Adam
from src.trainer import fit

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
PREDICTIONS: Path = ROOT / "outputs" / "piecewise_cubic"
FIGURES: Path = ROOT / "analysis" / "figures"

SEED: int = 42
SEGMENT_DEGREE: int = 3
LR: float = 0.05
N_STEPS: int = 1000
X_RANGE: tuple[float, float] = (-1.0, 13.0)
Y_MARGIN: float = 1.5
SEGMENT_COLORS: tuple[tuple[float, ...], ...] = tuple(
    plt.get_cmap("tab20")(i) for i in (0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19)
)
NODE_ORDERS: dict[str, tuple[int, ...]] = {"extrema": (1,), "inflection": (2,), "both": (1, 2)}


def derivative_roots(coef: np.ndarray, order: int, z_min: float, z_max: float) -> np.ndarray:
    roots = legendre.legroots(legendre.legder(coef, order))
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    return real[(real >= z_min) & (real <= z_max)]


def segment_nodes(x: np.ndarray, y: np.ndarray, orders: tuple[int, ...]) -> np.ndarray:
    z = (x - CENTER) / SCALE
    coef = np.linalg.lstsq(legendre.legvander(z, DEGREE), y, rcond=None)[0]
    roots = np.sort(
        np.concatenate([derivative_roots(coef, order, z.min(), z.max()) for order in orders])
    )
    z_nodes = np.concatenate([[z.min()], roots, [z.max()]])
    return CENTER + SCALE * z_nodes


def merge_small_segments(nodes: np.ndarray, min_width: float) -> np.ndarray:
    nodes = nodes.copy()
    while nodes.shape[0] > 2:
        widths = np.diff(nodes)
        s = int(np.argmin(widths))
        if widths[s] >= min_width:
            break

        if s == 0:
            drop = 1
        elif s == widths.shape[0] - 1:
            drop = s
        else:
            drop = s if widths[s - 1] <= widths[s + 1] else s + 1
        nodes = np.delete(nodes, drop)
    return nodes


def main(kind: str, min_width: float, overlap: float, save_predictions: bool) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    name = kind if min_width <= 0 else f"{kind}_minw{min_width}"
    name = name if overlap <= 0 else f"{name}_ov{overlap}"

    _, x, y = load_train(TRAIN_PATH)
    nodes = merge_small_segments(segment_nodes(x, y, NODE_ORDERS[kind]), min_width)
    print(f"nodes ({name}): {np.round(nodes, 4).tolist()}")
    segment_of = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.shape[0] - 2)

    grid = np.linspace(X_RANGE[0], X_RANGE[1], 3000)
    curves = []
    models = []
    y_fit = np.empty_like(y)
    histories = []

    print(f"{'segment':>16s} {'n_train':>7s} {'own rmse':>8s} {'lstsq rmse':>10s}")
    for s in range(nodes.shape[0] - 1):
        left, right = nodes[s], nodes[s + 1]
        mask = segment_of == s
        train_mask = (x >= left - overlap) & (x <= right + overlap)
        center, scale = (left + right) / 2, (right - left) / 2

        model = PolynomialModel(degree=SEGMENT_DEGREE, center=center, scale=scale)
        loader = FullBatchLoader(x[train_mask], y[train_mask], seed=SEED)
        history = fit(model, MSELoss(), Adam(model.parameters(), lr=LR), loader, n_epochs=N_STEPS)
        histories.append(history)
        models.append(model)

        y_fit[mask] = model(tf.constant(x[mask])).numpy()
        curves.append(model(tf.constant(grid)).numpy())

        z = (x[train_mask] - center) / scale
        b = np.vander(z, SEGMENT_DEGREE + 1, increasing=True)
        c = np.linalg.lstsq(b, y[train_mask], rcond=None)[0]
        lstsq_own = np.vander((x[mask] - center) / scale, SEGMENT_DEGREE + 1, increasing=True) @ c
        lstsq_rmse = np.sqrt(np.mean((lstsq_own - y[mask]) ** 2))
        own_rmse = np.sqrt(np.mean((y_fit[mask] - y[mask]) ** 2))
        print(
            f"[{left:6.3f},{right:6.3f}] {train_mask.sum():7d} {own_rmse:8.4f} {lstsq_rmse:10.4f}"
        )

    rmse = np.sqrt(np.mean((y_fit - y) ** 2))
    print(f"total train rmse={rmse:.4f}")

    jumps = []
    for i in range(1, nodes.shape[0] - 1):
        at = tf.constant([nodes[i]])
        jump = float(np.abs(models[i - 1](at).numpy()[0] - models[i](at).numpy()[0]))
        jumps.append(jump)
        print(f"jump at x={nodes[i]:7.4f}: {jump:.4f}")
    print(f"jump mean={np.mean(jumps):.4f} max={np.max(jumps):.4f}")

    if save_predictions:
        test_ids, x_test = load_test(TEST_PATH)
        test_segment = np.clip(
            np.searchsorted(nodes, x_test, side="right") - 1, 0, nodes.shape[0] - 2
        )
        y_test = np.empty_like(x_test)
        for s, model in enumerate(models):
            mask = test_segment == s
            y_test[mask] = model(tf.constant(x_test[mask])).numpy()

        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        prediction_path = PREDICTIONS / f"{name}_test.csv"
        save_submission(test_ids, y_test, prediction_path)
        print(f"saved={prediction_path}")

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
    for s, curve in enumerate(curves):
        own = (grid >= nodes[s]) & (grid <= nodes[s + 1])
        color = SEGMENT_COLORS[s % len(SEGMENT_COLORS)]
        plt.plot(grid, np.where(own, np.nan, curve), color=color, linewidth=0.8, linestyle=":")
        plt.plot(grid, np.where(own, curve, np.nan), color=color, label=f"segment {s}")
    for node in nodes:
        plt.axvline(node, color="black", linewidth=0.5, linestyle=":")
    plt.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
    plt.ylim(y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"piecewise cubic ({name}) | rmse {rmse:.3f}")
    plt.legend(loc="upper left", fontsize=6, ncol=2)
    plt.tight_layout()
    fit_path = FIGURES / f"piecewise_cubic_{name}.png"
    plt.savefig(fit_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    for s, history in enumerate(histories):
        color = SEGMENT_COLORS[s % len(SEGMENT_COLORS)]
        plt.plot(np.sqrt(history), color=color, linewidth=1, label=f"segment {s}")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("train rmse")
    plt.title("loss")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    loss_path = FIGURES / f"piecewise_cubic_{name}_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"saved={fit_path}")
    print(f"saved={loss_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", choices=list(NODE_ORDERS), default="extrema")
    parser.add_argument("--min-width", type=float, default=0.0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    main(args.nodes, args.min_width, args.overlap, args.save_predictions)
