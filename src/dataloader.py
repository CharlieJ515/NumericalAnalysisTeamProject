from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from src.typedefs import BatchVec, DataVec


class DataLoader(ABC):
    def __init__(self, x: DataVec, y: DataVec) -> None:
        if x.shape != y.shape:
            raise ValueError(f"x와 y의 길이가 다릅니다: {x.shape} != {y.shape}")

        self.x = x
        self.y = y

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[BatchVec, BatchVec]]: ...


class MiniBatchLoader(DataLoader):
    def __init__(self, x: DataVec, y: DataVec, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size는 양의 정수여야 합니다.")
        super().__init__(x, y)

        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return -(-self.y.shape[0] // self.batch_size)

    def __iter__(self) -> Iterator[tuple[BatchVec, BatchVec]]:
        """데이터를 섞은 뒤 크기 batch_size인 미니배치를 만든다."""
        n = self.y.shape[0]
        order = self.rng.permutation(n)
        for start in range(0, n, self.batch_size):
            batch = order[start : start + self.batch_size]
            yield self.x[batch], self.y[batch]


class FullBatchLoader(DataLoader):
    def __init__(self, x: DataVec, y: DataVec, seed: int) -> None:
        super().__init__(x, y)

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[tuple[BatchVec, BatchVec]]:
        yield self.x, self.y


class StochasticLoader(MiniBatchLoader):
    def __init__(self, x: DataVec, y: DataVec, seed: int) -> None:
        super().__init__(x, y, batch_size=1, seed=seed)
