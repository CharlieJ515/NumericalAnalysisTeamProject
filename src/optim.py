from collections.abc import Iterator

import numpy as np

from src.model import loss, loss_grad
from src.typedefs import BatchMat, BatchVec, DataVec, DesignMat, LossVec, ParamVec


def generate_batches(
    X: DesignMat, y: DataVec, batch_size: int, seed: int
) -> Iterator[tuple[BatchMat, BatchVec]]:
    """데이터를 섞은 뒤 크기 batch_size인 미니배치를 만든다."""
    if batch_size <= 0:
        raise ValueError("batch_size는 양의 정수여야 합니다.")
    n = y.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    for start in range(0, n, batch_size):
        batch = order[start : start + batch_size]
        yield X[batch], y[batch]


def gradient_descent(
    X: DesignMat,
    y: DataVec,
    theta: ParamVec,
    lr: float,
    n_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[ParamVec, LossVec]:
    """미니배치 gradient descent.

    각 배치에서 theta <- theta - lr * gradient 로 갱신한다.
    epoch가 끝날 때마다 전체 학습 데이터의 loss를 기록한다.
    """
    theta = np.asarray(theta, dtype=np.float64).copy()
    loss_history: list[float] = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in generate_batches(X, y, batch_size, seed + epoch):
            grad = loss_grad(x_batch, y_batch, theta)
            theta = theta - lr * grad
        loss_history.append(loss(X, y, theta))
    return theta, np.asarray(loss_history)
