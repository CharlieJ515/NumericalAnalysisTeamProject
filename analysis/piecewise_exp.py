from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.polynomial import legendre

from analysis.piecewise_cubic import SEGMENT_COLORS, derivative_roots
from analysis.stationary_spline import CENTER, DEGREE, SCALE
from src.dataio import load_train
from src.dataloader import FullBatchLoader
from src.loss import MSELoss
from src.model import Model
from src.optim import Adam
from src.trainer import fit
from src.typedefs import DataTensor, Params

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"

SEED: int = 42
LR: float = 0.05
N_STEPS: int = 2000
X_RANGE: tuple[float, float] = (-1.0, 13.0)
Y_MARGIN: float = 1.5


class OffsetExponentialModel(Model):
    def __init__(self, left: float, width: float, a: float, b: float, c: float) -> None:
        self.left = left
        self.width = width
        self.a = tf.Variable(tf.constant(a, dtype=tf.float64), name="a")
        self.b = tf.Variable(tf.constant(b, dtype=tf.float64), name="b")
        self.c = tf.Variable(tf.constant(c, dtype=tf.float64), name="c")

    def parameters(self) -> Params:
        return [self.a, self.b, self.c]

    def __call__(self, x: DataTensor) -> DataTensor:
        t = (x - self.left) / self.width
        return self.a + self.b * tf.exp(self.c * t)


def initial_parameters(coef: np.ndarray, left: float, right: float) -> tuple[float, float, float]:
    z_left, z_right = (left - CENTER) / SCALE, (right - CENTER) / SCALE
    z_mid = (z_left + z_right) / 2
    slope = legendre.legval(z_mid, legendre.legder(coef))
    curvature = legendre.legval(z_mid, legendre.legder(coef, 2))

    c = float(np.sign(slope) * np.sign(curvature)) or 1.0
    y_left, y_right = legendre.legval(z_left, coef), legendre.legval(z_right, coef)
    b = (y_right - y_left) / np.expm1(c)
    a = y_left - b
    return float(a), float(b), c


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x, y = load_train(TRAIN_PATH)
    z = (x - CENTER) / SCALE
    coef = np.linalg.lstsq(legendre.legvander(z, DEGREE), y, rcond=None)[0]

    roots = np.concatenate(
        [derivative_roots(coef, 1, z.min(), z.max()), derivative_roots(coef, 2, z.min(), z.max())]
    )
    nodes = CENTER + SCALE * np.concatenate([[z.min()], np.sort(roots), [z.max()]])
    n_segments = nodes.shape[0] - 1
    print(f"nodes: {np.round(nodes, 4).tolist()}")

    segment_of = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, n_segments - 1)
    grid = np.linspace(X_RANGE[0], X_RANGE[1], 3000)
    grid_segment = np.clip(np.searchsorted(nodes, grid, side="right") - 1, 0, n_segments - 1)
    y_grid = np.empty_like(grid)
    y_fit = np.empty_like(y)
    histories = []

    print(f"{'segment':>16s} {'n':>6s} {'rmse':>8s} {'a':>9s} {'b':>9s} {'c':>8s}")
    for s in range(n_segments):
        left, right = nodes[s], nodes[s + 1]
        mask = segment_of == s

        a, b, c = initial_parameters(coef, left, right)
        model = OffsetExponentialModel(left, right - left, a, b, c)
        loader = FullBatchLoader(x[mask], y[mask], seed=SEED)
        history = fit(model, MSELoss(), Adam(model.parameters(), lr=LR), loader, n_epochs=N_STEPS)
        histories.append(history)

        y_fit[mask] = model(tf.constant(x[mask])).numpy()
        grid_mask = grid_segment == s
        y_grid[grid_mask] = model(tf.constant(grid[grid_mask])).numpy()

        params = [float(p.numpy()) for p in model.parameters()]
        print(
            f"[{left:6.3f},{right:6.3f}] {mask.sum():6d} {np.sqrt(history[-1]):8.4f} "
            f"{params[0]:9.3f} {params[1]:9.3f} {params[2]:8.3f}"
        )

    rmse = np.sqrt(np.mean((y_fit - y) ** 2))
    print(f"total train rmse={rmse:.4f}")

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
    for s in range(n_segments):
        grid_mask = grid_segment == s
        color = SEGMENT_COLORS[s % len(SEGMENT_COLORS)]
        plt.plot(grid[grid_mask], y_grid[grid_mask], color=color, label=f"segment {s}")
    for node in nodes:
        plt.axvline(node, color="black", linewidth=0.5, linestyle=":")
    plt.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
    plt.ylim(y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"piecewise exponential | rmse {rmse:.3f}")
    plt.legend(loc="upper left", fontsize=6, ncol=2)
    plt.tight_layout()
    fit_path = FIGURES / "piecewise_exp.png"
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
    loss_path = FIGURES / "piecewise_exp_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"saved={fit_path}")
    print(f"saved={loss_path}")


if __name__ == "__main__":
    main()
