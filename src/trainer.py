import numpy as np
import tensorflow as tf

from src.dataloader import DataLoader
from src.loss import Loss
from src.model import Model
from src.optim import Optimizer
from src.typedefs import LossVec, ScalarTensor

BATCH_SPEC = tf.TensorSpec(shape=[None], dtype=tf.float64)


def fit(
    model: Model,
    loss_fn: Loss,
    optimizer: Optimizer,
    loader: DataLoader,
    n_epochs: int,
) -> LossVec:
    """미니배치 학습 루프.

    각 배치에서 optimizer.step으로 모델 파라미터를 갱신한다.
    epoch가 끝날 때마다 전체 학습 데이터의 loss를 기록한다.
    """

    x_all = tf.constant(loader.x, dtype=tf.float64)
    y_all = tf.constant(loader.y, dtype=tf.float64)

    def closure() -> ScalarTensor:
        return loss_fn(model(x_all), y_all)

    @tf.function(input_signature=[BATCH_SPEC, BATCH_SPEC])
    def train_step(x: tf.Tensor, y: tf.Tensor) -> None:
        with tf.GradientTape() as tape:
            value = loss_fn(model(x), y)
        grads = tape.gradient(value, model.parameters())
        optimizer.step([tf.convert_to_tensor(grad) for grad in grads], closure)

    loss_history: list[float] = []

    for _ in range(n_epochs):
        for x_batch, y_batch in loader:
            train_step(tf.constant(x_batch), tf.constant(y_batch))

        loss_history.append(float(loss_fn(model(x_all), y_all)))

    return np.asarray(loss_history)
