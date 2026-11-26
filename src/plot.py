from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.typedefs import LossVec


def visualize_l2(loss_history: LossVec, path: Path) -> None:
    """epoch별 L2 loss 곡선을 저장한다."""
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
    plt.xlabel("epoch")
    plt.ylabel("L2 loss")
    plt.title("L2 loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
