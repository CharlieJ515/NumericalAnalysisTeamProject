from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float64

from src.dataio import load_test, load_train
from src.typedefs import DataVec

ROOT: Path = Path(__file__).resolve().parents[1]
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
FIGURES: Path = ROOT / "analysis" / "figures"

KDE_BANDWIDTH: tuple[float, float] = (0.02, 0.05)
KDE_GRID_SIZE: tuple[int, int] = (500, 400)
CENTERED_HALF_WIDTH: float = 1.0
CENTERED_SIZE: int = 201
REFERENCE_SEED: int = 0


def gaussian_kernel(
    grid: DataVec, samples: DataVec, bandwidth: float
) -> Float64[np.ndarray, "g n"]:
    u = (grid[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * u**2) / (bandwidth * np.sqrt(2 * np.pi))


def gaussian_kde_2d(
    x: DataVec, y: DataVec, grid_x: DataVec, grid_y: DataVec, bandwidth: tuple[float, float]
) -> Float64[np.ndarray, "gy gx"]:
    density = np.zeros((grid_y.shape[0], grid_x.shape[0]))
    for idx in np.array_split(np.arange(x.shape[0]), 20):
        kx = gaussian_kernel(grid_x, x[idx], bandwidth[0])
        ky = gaussian_kernel(grid_y, y[idx], bandwidth[1])
        density += ky @ kx.T
    return density / x.shape[0]


def plot_scatter(x_train: DataVec, y_train: DataVec, x_test: DataVec, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x_train, y_train, s=1, alpha=0.2, label="train")
    plt.axvspan(x_train.max(), x_test.max(), color="gray", alpha=0.2, label="test only")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_x_hist(x_train: DataVec, x_test: DataVec, path: Path) -> None:
    bins = np.arange(0.0, np.ceil(max(x_train.max(), x_test.max())) + 0.5, 0.5).tolist()

    plt.figure(figsize=(6, 4))
    plt.hist(x_train, bins=bins, density=True, histtype="step", label="train")
    plt.hist(x_test, bins=bins, density=True, histtype="step", label="test")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("x histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def conditional_kde(
    density: Float64[np.ndarray, "gy gx"], grid_y: DataVec
) -> tuple[Float64[np.ndarray, "gy gx"], DataVec]:
    conditional = density / np.trapezoid(density, grid_y, axis=0)
    mean = np.asarray(np.trapezoid(grid_y[:, None] * conditional, grid_y, axis=0))
    return conditional, mean


def center_kde(
    conditional: Float64[np.ndarray, "gy gx"], mean: DataVec, grid_y: DataVec, offsets: DataVec
) -> Float64[np.ndarray, "go gx"]:
    centered = np.empty((offsets.shape[0], conditional.shape[1]))
    for j in range(conditional.shape[1]):
        centered[:, j] = np.interp(mean[j] + offsets, grid_y, conditional[:, j])
    return centered


def gaussian_fit_stats(
    conditional: Float64[np.ndarray, "gy gx"],
    mean: DataVec,
    grid_x: DataVec,
    grid_y: DataVec,
    bandwidth: tuple[float, float],
) -> dict[str, DataVec]:
    dy = grid_y[:, None] - mean[None, :]
    var = np.asarray(np.trapezoid(dy**2 * conditional, grid_y, axis=0))
    skew = np.asarray(np.trapezoid(dy**3 * conditional, grid_y, axis=0)) / var**1.5
    kurt = np.asarray(np.trapezoid(dy**4 * conditional, grid_y, axis=0)) / var**2 - 3

    gaussian = np.exp(-0.5 * dy**2 / var) / np.sqrt(2 * np.pi * var)
    overlap = np.asarray(np.trapezoid(np.sqrt(conditional * gaussian), grid_y, axis=0))
    hellinger = np.sqrt(np.clip(1 - overlap, 0, None))

    slope = np.gradient(mean, grid_x)
    noise_var = var - bandwidth[1] ** 2 - (slope * bandwidth[0]) ** 2
    std = np.sqrt(np.clip(noise_var, 0, None))

    return {"std": std, "skew": skew, "kurtosis": kurt, "hellinger": hellinger}


def kde_stats(
    x: DataVec, y: DataVec, grid_x: DataVec, grid_y: DataVec
) -> tuple[Float64[np.ndarray, "gy gx"], Float64[np.ndarray, "gy gx"], DataVec, dict[str, DataVec]]:
    density = gaussian_kde_2d(x, y, grid_x, grid_y, KDE_BANDWIDTH)
    conditional, mean = conditional_kde(density, grid_y)
    stats = gaussian_fit_stats(conditional, mean, grid_x, grid_y, KDE_BANDWIDTH)
    return density, conditional, mean, stats


def plot_kde_2d(
    density: Float64[np.ndarray, "gy gx"], grid_x: DataVec, grid_y: DataVec, path: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
    )
    plt.colorbar(label="density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2d kde")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_kde_centered(
    centered: Float64[np.ndarray, "go gx"], grid_x: DataVec, offsets: DataVec, path: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.imshow(
        centered,
        origin="lower",
        aspect="auto",
        extent=(grid_x[0], grid_x[-1], offsets[0], offsets[-1]),
    )
    plt.colorbar(label="density")
    plt.xlabel("x")
    plt.ylabel("y - mean")
    plt.title("centered kde")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gaussian_fit(
    grid_x: DataVec, stats: dict[str, DataVec], reference: dict[str, DataVec], path: Path
) -> None:
    fig, axes = plt.subplots(len(stats), 1, figsize=(6, 8), sharex=True)
    for ax, name in zip(axes, stats, strict=True):
        ax.plot(grid_x, reference[name], color="gray", label="gaussian")
        ax.plot(grid_x, stats[name], label="data")
        ax.set_ylabel(name)
    axes[0].set_title("gaussian fit")
    axes[0].legend()
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_train(TRAIN_PATH)
    _, x_test = load_test(TEST_PATH)

    scatter_path = FIGURES / "overview_scatter.png"
    hist_path = FIGURES / "overview_x_hist.png"
    kde_path = FIGURES / "overview_kde_2d.png"
    centered_path = FIGURES / "overview_kde_centered.png"
    fit_path = FIGURES / "overview_gaussian_fit.png"
    plot_scatter(x_train, y_train, x_test, scatter_path)
    plot_x_hist(x_train, x_test, hist_path)

    grid_x = np.linspace(x_train.min(), x_train.max(), KDE_GRID_SIZE[0])
    grid_y = np.linspace(y_train.min() - 0.5, y_train.max() + 0.5, KDE_GRID_SIZE[1])
    offsets = np.linspace(-CENTERED_HALF_WIDTH, CENTERED_HALF_WIDTH, CENTERED_SIZE)
    density, conditional, mean, stats = kde_stats(x_train, y_train, grid_x, grid_y)
    plot_kde_2d(density, grid_x, grid_y, kde_path)
    centered = center_kde(conditional, mean, grid_y, offsets)
    plot_kde_centered(centered, grid_x, offsets, centered_path)

    sigma = float(np.median(stats["std"]))
    rng = np.random.default_rng(REFERENCE_SEED)
    y_reference = np.interp(x_train, grid_x, mean) + rng.normal(0.0, sigma, x_train.shape[0])
    _, _, _, reference = kde_stats(x_train, y_reference, grid_x, grid_y)
    plot_gaussian_fit(grid_x, stats, reference, fit_path)

    outside = x_test > x_train.max()
    print(f"train x: [{x_train.min():.4f}, {x_train.max():.4f}]  n={x_train.shape[0]}")
    print(f"test  x: [{x_test.min():.4f}, {x_test.max():.4f}]  n={x_test.shape[0]}")
    print(f"test outside train range: {outside.sum()} ({outside.mean():.1%})")
    print(f"saved={scatter_path}")
    print(f"saved={hist_path}")
    print(f"saved={kde_path}")
    for name in stats:
        print(
            f"{name:9s} data: {stats[name].mean():+.4f} ± {stats[name].std():.4f}"
            f"   gaussian: {reference[name].mean():+.4f} ± {reference[name].std():.4f}"
        )
    print(f"saved={centered_path}")
    print(f"saved={fit_path}")


if __name__ == "__main__":
    main()
