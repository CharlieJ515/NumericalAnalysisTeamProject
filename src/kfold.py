from collections.abc import Iterator

import numpy as np

from src.typedefs import IndexVec


class KFold:
    def __init__(self, n_splits: int, seed: int) -> None:
        if n_splits < 2:
            raise ValueError("n_splits는 2 이상이어야 합니다.")

        self.n_splits = n_splits
        self.seed = seed

    def __len__(self) -> int:
        return self.n_splits

    def split(self, n_samples: int) -> Iterator[tuple[IndexVec, IndexVec]]:
        if n_samples < self.n_splits:
            raise ValueError(f"샘플 수가 fold 수보다 적습니다: {n_samples} < {self.n_splits}")

        order = np.random.default_rng(self.seed).permutation(n_samples)
        folds = np.array_split(order, self.n_splits)

        for i, val_idx in enumerate(folds):
            train_idx = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
            yield train_idx, val_idx
