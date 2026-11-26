from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre

from src.dataio import load_train

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"

DEGREE: int = 20
CENTER: float = 5.0
SCALE: float = 5.0
X_RANGE: tuple[float, float] = (-1.0, 13.0)
Y_MARGIN: float = 1.5


def stationary_points(coef: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    roots = legendre.legroots(legendre.legder(coef))
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    return real[(real >= z_min) & (real <= z_max)]


def natural_cubic_spline(knots_x: np.ndarray, knots_y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = knots_x.shape[0]
    h = np.diff(knots_x)

    system = np.zeros((n, n))
    rhs = np.zeros(n)
    system[0, 0] = 1.0
    system[-1, -1] = 1.0
    for i in range(1, n - 1):
        system[i, i - 1] = h[i - 1]
        system[i, i] = 2 * (h[i - 1] + h[i])
        system[i, i + 1] = h[i]
        rhs[i] = 6 * (
            (knots_y[i + 1] - knots_y[i]) / h[i] - (knots_y[i] - knots_y[i - 1]) / h[i - 1]
        )
    m = np.linalg.solve(system, rhs)

    i = np.clip(np.searchsorted(knots_x, x) - 1, 0, n - 2)
    left = knots_x[i + 1] - x
    right = x - knots_x[i]
    return (
        m[i] * left**3 / (6 * h[i])
        + m[i + 1] * right**3 / (6 * h[i])
        + (knots_y[i] / h[i] - m[i] * h[i] / 6) * left
        + (knots_y[i + 1] / h[i] - m[i + 1] * h[i] / 6) * right
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x, y = load_train(TRAIN_PATH)
    z = (x - CENTER) / SCALE
    coef = np.linalg.lstsq(legendre.legvander(z, DEGREE), y, rcond=None)[0]

    z_nodes = np.concatenate([[z.min()], stationary_points(coef, z.min(), z.max()), [z.max()]])
    x_nodes = CENTER + SCALE * z_nodes
    y_nodes = legendre.legval(z_nodes, coef)

    grid = np.linspace(X_RANGE[0], X_RANGE[1], 1000)
    y_poly = legendre.legval((grid - CENTER) / SCALE, coef)
    y_spline = natural_cubic_spline(x_nodes, y_nodes, grid)
    rmse = np.sqrt(np.mean((natural_cubic_spline(x_nodes, y_nodes, x) - y) ** 2))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
    plt.plot(grid, y_poly, color="black", linestyle="--", linewidth=1, label=f"degree {DEGREE}")
    plt.plot(grid, y_spline, color="C1", label="spline")
    plt.scatter(x_nodes, y_nodes, color="C3", zorder=3, label="nodes")
    plt.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
    plt.ylim(y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"cubic spline | rmse {rmse:.3f}")
    plt.legend(loc="upper left")
    plt.tight_layout()

    path = FIGURES / "stationary_spline.png"
    plt.savefig(path, dpi=150)
    plt.close()

    for xn, yn in zip(x_nodes, y_nodes, strict=True):
        print(f"node x={xn:8.4f} y={yn:8.4f}")
    print(f"spline train rmse={rmse:.4f}")
    print(f"saved={path}")


if __name__ == "__main__":
    main()
