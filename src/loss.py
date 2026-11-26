from abc import ABC, abstractmethod

import tensorflow as tf

from src.typedefs import DataTensor, ScalarTensor


class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred: DataTensor, y: DataTensor) -> ScalarTensor: ...


class MSELoss(Loss):
    def __call__(self, y_pred: DataTensor, y: DataTensor) -> ScalarTensor:
        """평균 제곱오차. L2 손실이다."""
        return tf.reduce_mean(tf.square(y_pred - y))


class MAELoss(Loss):
    def __call__(self, y_pred: DataTensor, y: DataTensor) -> ScalarTensor:
        return tf.reduce_mean(tf.abs(y_pred - y))
