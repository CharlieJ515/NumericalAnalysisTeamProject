from pathlib import Path

import numpy as np

from src.dataio import load_test, load_train, save_checkpoint, save_submission
from src.model import make_design_matrix, model
from src.optim import gradient_descent
from src.plot import visualize_l2
from src.typedefs import ParamVec

SEED: int = 42
LR: float = 0.01
BATCH_SIZE: int = 256
N_EPOCHS: int = 100

ROOT: Path = Path(__file__).resolve().parent
TRAIN_PATH: Path = ROOT / "data" / "train.csv"
TEST_PATH: Path = ROOT / "data" / "test.csv"
OUT: Path = ROOT / "outputs"
SUBMISSION_PATH: Path = OUT / "submission.csv"
LOSS_FIGURE_PATH: Path = OUT / "loss_l2.png"
CKPT_PATH: Path = OUT / "baseline_ckpt.npz"


def predict_from_checkpoint(
    ckpt_path: Path = CKPT_PATH,
    test_path: Path = TEST_PATH,
    submission_path: Path = SUBMISSION_PATH,
) -> ParamVec:
    """저장된 theta로 test를 예측하고 submission.csv를 다시 쓴다."""
    ckpt = np.load(ckpt_path)
    theta = np.asarray(ckpt["theta"], dtype=np.float64)
    test_ids, x_test = load_test(test_path)
    y_pred = model(make_design_matrix(x_test), theta)
    save_submission(test_ids, y_pred, submission_path)
    return theta


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    train_ids, x_train, y_train = load_train(TRAIN_PATH)
    test_ids, x_test = load_test(TEST_PATH)
    del train_ids

    X_train = make_design_matrix(x_train)
    theta = np.zeros(X_train.shape[1], dtype=np.float64)
    theta, loss_history = gradient_descent(
        X_train,
        y_train,
        theta,
        lr=LR,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    visualize_l2(loss_history, LOSS_FIGURE_PATH)
    y_pred = model(make_design_matrix(x_test), theta)
    save_submission(test_ids, y_pred, SUBMISSION_PATH)
    save_checkpoint(theta, loss_history, CKPT_PATH)

    print(f"seed={SEED} lr={LR} batch_size={BATCH_SIZE} epochs={N_EPOCHS}")
    print(f"theta={theta}")
    print(f"final_train_loss={loss_history[-1]:.6f}")
    print(f"saved={SUBMISSION_PATH}")
    print(f"saved={LOSS_FIGURE_PATH}")
    print(f"saved={CKPT_PATH}")


if __name__ == "__main__":
    main()
