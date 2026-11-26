import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import OPTIMIZERS
from src.dataio import load_train
from src.dataloader import FullBatchLoader, MiniBatchLoader
from src.loss import MSELoss
from src.model import PolynomialModel
from src.trainer import fit

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
FIGURES: Path = ROOT / "analysis" / "figures"
RESULTS: Path = ROOT / "outputs" / "poly_sweep"

SEED: int = 42
CENTER: float = 5.0
X_RANGE: tuple[float, float] = (-1.0, 13.0)
Y_MARGIN: float = 1.5

SWEEPS: dict[str, list[dict[str, Any]]] = {
    "optimizer": [
        {"degree": 20, "optimizer": name, "lr": lr, "batch": 1024, "epochs": 500, "scale": 5.0}
        for name, lrs in [
            ("SGD", (0.01, 0.05)),
            ("Momentum", (0.005, 0.01)),
            ("Nesterov", (0.005, 0.01)),
            ("Adagrad", (0.1, 0.5)),
            ("RMSProp", (0.001, 0.01)),
            ("Adam", (0.01, 0.05)),
        ]
        for lr in lrs
    ],
    "budget": [
        {
            "degree": 20,
            "optimizer": "Adam",
            "lr": 0.01,
            "batch": batch,
            "epochs": epochs,
            "scale": 5.0,
        }
        for batch, epochs in [
            (256, 200),
            (256, 1000),
            (1024, 200),
            (1024, 2000),
            (4096, 500),
            (4096, 4000),
        ]
    ],
    "degree": [
        {
            "degree": degree,
            "optimizer": "Adam",
            "lr": 0.01,
            "batch": 1024,
            "epochs": 1000,
            "scale": scale,
        }
        for degree in (10, 15, 20, 25)
        for scale in (5.0, 2.887)
    ],
}


def lstsq_curve(x: np.ndarray, y: np.ndarray, degree: int, grid: np.ndarray) -> np.ndarray:
    z = (x - CENTER) / 5.0
    coef = np.linalg.lstsq(np.polynomial.legendre.legvander(z, degree), y, rcond=None)[0]
    return np.polynomial.legendre.legval((grid - CENTER) / 5.0, coef)


def run(
    x: np.ndarray, y: np.ndarray, setting: dict[str, Any]
) -> tuple[PolynomialModel, np.ndarray, float]:
    model = PolynomialModel(degree=setting["degree"], center=CENTER, scale=setting["scale"])
    optimizer = OPTIMIZERS[setting["optimizer"]](model.parameters(), lr=setting["lr"])
    if setting["batch"] >= x.shape[0]:
        loader = FullBatchLoader(x, y, seed=SEED)
    else:
        loader = MiniBatchLoader(x, y, batch_size=setting["batch"], seed=SEED)

    start = time.time()
    history = fit(model, MSELoss(), optimizer, loader, n_epochs=setting["epochs"])
    return model, history, time.time() - start


def label(setting: dict[str, Any]) -> str:
    return (
        f"d{setting['degree']} {setting['optimizer']} lr{setting['lr']} "
        f"b{setting['batch']} e{setting['epochs']} s{setting['scale']}"
    )


def plot_fits(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, rows: list[dict[str, Any]], path: Path
) -> None:
    n_cols = 2
    n_rows = -(-len(rows) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10, 3 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    y_limits = (y.min() - Y_MARGIN, y.max() + Y_MARGIN)

    for ax, row in zip(axes.flat, rows, strict=False):
        ax.scatter(x, y, s=1, alpha=0.1, color="gray")
        ax.plot(grid, row["lstsq"], color="black", linestyle="--", linewidth=1, label="lstsq")
        ax.plot(grid, row["curve"], color="C1", label="model")
        ax.axvspan(x.max(), 12.0, color="gray", alpha=0.15)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{label(row['setting'])} | rmse {row['rmse']:.3f}", fontsize=9)

    for ax in axes.flat[len(rows) :]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_losses(rows: list[dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for row in rows:
        plt.plot(np.sqrt(row["history"]), label=label(row["setting"]), linewidth=1)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train rmse")
    plt.title("loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(sweep: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    _, x, y = load_train(TRAIN_PATH)
    grid = np.linspace(X_RANGE[0], X_RANGE[1], 1000)

    rows = []
    for setting in SWEEPS[sweep]:
        model, history, seconds = run(x, y, setting)
        rmse = float(np.sqrt(history[-1]))
        rows.append(
            {
                "setting": setting,
                "history": history,
                "rmse": rmse,
                "seconds": seconds,
                "curve": model(tf.constant(grid)).numpy(),
                "lstsq": lstsq_curve(x, y, setting["degree"], grid),
            }
        )
        print(f"{label(setting):45s} rmse={rmse:.4f} time={seconds:6.1f}s", flush=True)

    fits_path = FIGURES / f"poly_sweep_{sweep}_fits.png"
    losses_path = FIGURES / f"poly_sweep_{sweep}_losses.png"
    plot_fits(x, y, grid, rows, fits_path)
    plot_losses(rows, losses_path)

    summary = [{**row["setting"], "rmse": row["rmse"], "seconds": row["seconds"]} for row in rows]
    (RESULTS / f"{sweep}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={fits_path}")
    print(f"saved={losses_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=list(SWEEPS), required=True)
    args = parser.parse_args()

    main(args.sweep)
