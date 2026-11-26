import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.config import DATALOADERS, LOSSES, MODELS, OPTIMIZERS, build, load_config
from src.dataio import load_test, load_train, save_submission
from src.kfold import KFold
from src.model import Model
from src.plot import visualize_l2
from src.trainer import fit
from src.typedefs import DataVec, LossVec

ROOT: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "baseline.toml"


def train_model(config: dict[str, Any], x: DataVec, y: DataVec, seed: int) -> tuple[Model, LossVec]:
    model = build(MODELS, config["model"])
    loss_fn = build(LOSSES, config["loss"])
    optimizer = build(OPTIMIZERS, config["optimizer"], params=model.parameters())
    loader = build(DATALOADERS, config["dataloader"], x=x, y=y, seed=seed)

    loss_history = fit(model, loss_fn, optimizer, loader, n_epochs=config["n_epochs"])
    return model, loss_history


def main(config: dict[str, Any]) -> None:
    seed = config["seed"]
    paths = config["paths"]

    out = ROOT / paths["out"]
    submission_path = out / paths["submission"]
    loss_figure_path = out / paths["loss_figure"]
    checkpoint_path = out / paths["checkpoint"]

    np.random.seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    train_ids, x_train, y_train = load_train(ROOT / paths["train"])
    test_ids, x_test = load_test(ROOT / paths["test"])
    del train_ids

    if "kfold" in config:
        kfold = KFold(seed=seed, **config["kfold"])
        for fold, (train_idx, _) in enumerate(kfold.split(x_train.shape[0])):
            _, fold_history = train_model(config, x_train[train_idx], y_train[train_idx], seed)
            print(f"fold={fold} final_train_loss={fold_history[-1]:.6f}")

    model, loss_history = train_model(config, x_train, y_train, seed)

    visualize_l2(loss_history, loss_figure_path)
    y_pred = model(tf.constant(x_test)).numpy()
    save_submission(test_ids, y_pred, submission_path)

    model.save(checkpoint_path)

    print(f"params={[param.numpy() for param in model.parameters()]}")
    print(f"final_train_loss={loss_history[-1]:.6f}")
    print(f"saved={submission_path}")
    print(f"saved={loss_figure_path}")
    print(f"saved={checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    print(f"config={args.config}")
    main(load_config(args.config))
