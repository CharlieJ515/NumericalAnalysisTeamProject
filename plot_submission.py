import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.dataio import load_test, load_train

ROOT: Path = Path(__file__).resolve().parent
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
FIGURES: Path = ROOT / "submissions" / "figures"

Y_MARGIN: float = 1.5


def load_submission(path: Path, test_ids: np.ndarray) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True)
    if not np.array_equal(table["id"].astype(np.int64), test_ids):
        raise ValueError(f"{path.name}의 id 순서가 test와 다릅니다.")
    return np.asarray(table["y_pred"], dtype=np.float64)


def draw(
    ax: Axes,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    order = np.argsort(x_test)
    ax.scatter(x_train, y_train, s=1, alpha=0.1, color="gray", label="train")
    ax.axvspan(x_train.max(), x_test.max(), color="gray", alpha=0.15)
    for name, y_pred in predictions.items():
        ax.plot(x_test[order], y_pred[order], linewidth=1.5, label=name)

    values = np.concatenate([y_train, *predictions.values()])
    ax.set_ylim(values.min() - Y_MARGIN, values.max() + Y_MARGIN)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=7)


def main(paths: list[Path], overlay: str | None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    test_ids, x_test = load_test(TEST_PATH)
    predictions = {path.stem: load_submission(path, test_ids) for path in paths}

    if overlay is None:
        for name, y_pred in predictions.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            draw(ax, x_train, y_train, x_test, {name: y_pred})
            ax.set_title(name)
            fig.tight_layout()
            figure_path = FIGURES / f"{name}.png"
            fig.savefig(figure_path, dpi=150)
            plt.close(fig)
            print(f"saved={figure_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, x_train, y_train, x_test, predictions)
    ax.set_title(overlay)
    fig.tight_layout()
    figure_path = FIGURES / f"{overlay}.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"saved={figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--overlay", type=str, default=None)
    args = parser.parse_args()

    main(args.paths, args.overlay)
