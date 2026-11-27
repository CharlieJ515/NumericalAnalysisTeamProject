from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.dataio import load_test, load_train, save_submission

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
FIGURES: Path = ROOT / "analysis" / "figures"
BASE_PREDICTIONS: Path = ROOT / "outputs" / "piecewise_cubic" / "inflection_minw0.2_ov0.1_test.csv"
SUBMISSIONS: Path = ROOT / "submissions"
SUBMISSION_PREFIX: str = "20261126"

EDGE_WINDOW: float = 0.3
Y_MARGIN: float = 1.5


def edge_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = x >= x.max() - EDGE_WINDOW
    design = np.column_stack([np.ones(mask.sum()), x[mask] - 10.0])
    value, slope = np.linalg.lstsq(design, y[mask], rcond=None)[0]
    return float(value), float(slope)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    _, x_test = load_test(TEST_PATH)

    value, slope = edge_line(x_train, y_train)
    print(f"edge line: f(10)={value:.4f} f'(10)={slope:.4f} (window {EDGE_WINDOW})")

    probe_slopes: dict[str, float] = {
        "probe 1: flat": 0.0,
        "probe 2: up": 1.0,
        "probe 3: down": -1.0,
    }

    base = np.genfromtxt(BASE_PREDICTIONS, delimiter=",", names=True)
    test_ids, _ = load_test(TEST_PATH)
    if not np.array_equal(base["id"].astype(np.int64), test_ids):
        raise ValueError(f"{BASE_PREDICTIONS}의 id 순서가 test와 다릅니다.")
    outside = x_test > x_train.max()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for i, (label, probe_slope) in enumerate(probe_slopes.items(), start=1):
        y_pred = base["y_pred"].copy()
        y_pred[outside] = value + probe_slope * (x_test[outside] - 10)
        name = label.split(": ")[1]
        path = SUBMISSIONS / f"{SUBMISSION_PREFIX}_{i}_phase1_{name}.csv"
        save_submission(test_ids, y_pred, path)
        print(f"saved={path}")

    grid = np.linspace(10.0, 12.0, 400)
    y_limits = (y_train.min() - Y_MARGIN - 4, y_train.max() + Y_MARGIN)

    def draw_background(ax: Axes) -> None:
        ax.scatter(x_train, y_train, s=1, alpha=0.1, color="gray", label="train")
        ax.axvspan(x_train.max(), x_test.max(), color="gray", alpha=0.15)
        ax.set_ylim(*y_limits)

    plt.figure(figsize=(8, 5))
    draw_background(plt.gca())
    for label, probe_slope in probe_slopes.items():
        plt.plot(grid, value + probe_slope * (grid - 10), linewidth=2, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("phase 1")
    plt.legend(loc="upper left")
    plt.tight_layout()
    design_path = FIGURES / "probe_phase1_design.png"
    plt.savefig(design_path, dpi=150)
    plt.close()

    print(f"saved={design_path}")


if __name__ == "__main__":
    main()
