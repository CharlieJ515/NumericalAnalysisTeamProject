from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre

from analysis.piecewise_cubic import derivative_roots
from analysis.stationary_spline import CENTER, DEGREE, SCALE
from src.dataio import load_train

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"

X_RANGE: tuple[float, float] = (-0.5, 10.5)
Y_MARGIN: float = 1.5


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x, y = load_train(TRAIN_PATH)
    z = (x - CENTER) / SCALE
    coef = np.linalg.lstsq(legendre.legvander(z, DEGREE), y, rcond=None)[0]

    z_extrema = derivative_roots(coef, 1, z.min(), z.max())
    z_inflection = derivative_roots(coef, 2, z.min(), z.max())
    x_extrema = CENTER + SCALE * z_extrema
    x_inflection = CENTER + SCALE * z_inflection

    grid = np.linspace(X_RANGE[0], X_RANGE[1], 1000)
    y_grid = legendre.legval((grid - CENTER) / SCALE, coef)

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=1, alpha=0.1, color="gray", label="data")
    plt.plot(grid, y_grid, color="black", linewidth=1, label=f"degree {DEGREE}")
    for xe in x_extrema:
        plt.axvline(xe, color="C3", linewidth=0.6, linestyle=":")
    for xi in x_inflection:
        plt.axvline(xi, color="C0", linewidth=0.6, linestyle=":")
    plt.scatter(x_extrema, legendre.legval(z_extrema, coef), color="C3", zorder=3, label="extremum")
    plt.scatter(
        x_inflection,
        legendre.legval(z_inflection, coef),
        color="C0",
        marker="s",
        zorder=3,
        label="inflection",
    )
    plt.xlim(*X_RANGE)
    plt.ylim(y.min() - Y_MARGIN, y.max() + Y_MARGIN)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("extrema and inflection points")
    plt.legend(loc="upper left")
    plt.tight_layout()

    path = FIGURES / "critical_points.png"
    plt.savefig(path, dpi=150)
    plt.close()

    for name, xs in (("extremum", x_extrema), ("inflection", x_inflection)):
        print(f"{name:10s} {np.round(xs, 4).tolist()}")
    print(f"saved={path}")


if __name__ == "__main__":
    main()
