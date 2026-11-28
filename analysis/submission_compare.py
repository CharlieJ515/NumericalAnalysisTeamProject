from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from plot_submission import load_submission
from src.dataio import load_test, load_train

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
SUBMISSIONS: Path = ROOT / "submissions"
FIGURES: Path = ROOT / "analysis" / "figures"

PANELS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "(a) Phase 1 Probes",
        {
            "Slope 0": "20261126_1_phase1_flat.csv",
            "Slope +1": "20261126_2_phase1_up.csv",
            "Slope −1": "20261126_3_phase1_down.csv",
        },
    ),
    (
        "(b) Slope −1.7",
        {"Slope −1.7": "20261127_1_slope-1.7.csv"},
    ),
)
X_RANGE: tuple[float, float] = (-0.5, 12.5)
Y_MARGIN: float = 1.5


def draw(
    ax: Axes,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    order = np.argsort(x_test)
    ax.scatter(x_train, y_train, s=1, alpha=0.1, color="gray")
    ax.axvspan(x_train.max(), x_test.max(), color="gray", alpha=0.15)
    for name, y_pred in predictions.items():
        ax.plot(x_test[order], y_pred[order], linewidth=1.5, label=name)
    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("x")
    ax.legend(loc="upper left")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    test_ids, x_test = load_test(TEST_PATH)
    panels = [
        (
            title,
            {name: load_submission(SUBMISSIONS / file, test_ids) for name, file in files.items()},
        )
        for title, files in PANELS
    ]

    values = np.concatenate([y_train, *(y for _, p in panels for y in p.values())])
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 5), sharey=True)
    for ax, (title, predictions) in zip(axes, panels, strict=True):
        draw(ax, x_train, y_train, x_test, predictions)
        ax.set_title(title)
    axes[0].set_ylim(values.min() - Y_MARGIN, values.max() + Y_MARGIN)
    axes[0].set_ylabel("y")
    fig.tight_layout(pad=0.2, w_pad=0.3)

    path = FIGURES / "submission_compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved={path}")


if __name__ == "__main__":
    main()
