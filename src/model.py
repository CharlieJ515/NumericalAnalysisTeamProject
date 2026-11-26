from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.typedefs import DataTensor, DesignTensor, Params


class Model(ABC):
    @abstractmethod
    def __call__(self, x: DataTensor) -> DataTensor: ...

    @abstractmethod
    def parameters(self) -> Params: ...

    def named_parameters(self) -> dict[str, tf.Variable]:
        named = {param.name.split(":")[0]: param for param in self.parameters()}
        if len(named) != len(self.parameters()):
            raise ValueError(f"파라미터 이름이 중복됩니다: {[p.name for p in self.parameters()]}")
        return named

    def save(self, path: Path) -> None:
        arrays: dict[str, Any] = {
            name: param.numpy() for name, param in self.named_parameters().items()
        }
        np.savez(path, **arrays)

    def load(self, path: Path) -> None:
        named = self.named_parameters()
        with np.load(path) as ckpt:
            if set(ckpt.files) != set(named):
                raise ValueError(
                    f"파라미터 이름이 다릅니다: {sorted(ckpt.files)} != {sorted(named)}"
                )

            for name, param in named.items():
                value = ckpt[name]
                if value.shape != tuple(param.shape):
                    raise ValueError(f"{name}의 shape이 다릅니다: {value.shape} != {param.shape}")
                param.assign(value)


class LinearModel(Model):
    def __init__(self) -> None:
        self.theta = tf.Variable(tf.zeros(2, dtype=tf.float64), name="theta")

    def parameters(self) -> Params:
        return [self.theta]

    def features(self, x: DataTensor) -> DesignTensor:
        return tf.stack([tf.ones_like(x), x], axis=1)

    def __call__(self, x: DataTensor) -> DataTensor:
        return tf.linalg.matvec(self.features(x), self.theta)


class PolynomialModel(Model):
    def __init__(self, degree: int, center: float = 0.0, scale: float = 1.0) -> None:
        if degree < 0:
            raise ValueError("degree는 0 이상이어야 합니다.")
        if scale <= 0:
            raise ValueError("scale은 양수여야 합니다.")

        self.degree = degree
        self.center = center
        self.scale = scale
        self.theta = tf.Variable(tf.zeros(degree + 1, dtype=tf.float64), name="theta")

    def parameters(self) -> Params:
        return [self.theta]

    def features(self, x: DataTensor) -> DesignTensor:
        z = (x - self.center) / self.scale
        return tf.stack([z**k for k in range(self.degree + 1)], axis=1)

    def __call__(self, x: DataTensor) -> DataTensor:
        return tf.linalg.matvec(self.features(x), self.theta)


class ExponentialModel(Model):
    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        self.alpha = tf.Variable(tf.constant(alpha, dtype=tf.float64), name="alpha")
        self.beta = tf.Variable(tf.constant(beta, dtype=tf.float64), name="beta")

    def parameters(self) -> Params:
        return [self.alpha, self.beta]

    def __call__(self, x: DataTensor) -> DataTensor:
        return self.alpha * tf.exp(self.beta * x)


class SaturationModel(Model):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = tf.Variable(tf.constant(alpha, dtype=tf.float64), name="alpha")
        self.beta = tf.Variable(tf.constant(beta, dtype=tf.float64), name="beta")

    def parameters(self) -> Params:
        return [self.alpha, self.beta]

    def __call__(self, x: DataTensor) -> DataTensor:
        return self.alpha * x / (self.beta + x)


class PowerModel(Model):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = tf.Variable(tf.constant(alpha, dtype=tf.float64), name="alpha")
        self.beta = tf.Variable(tf.constant(beta, dtype=tf.float64), name="beta")

    def parameters(self) -> Params:
        return [self.alpha, self.beta]

    def __call__(self, x: DataTensor) -> DataTensor:
        return self.alpha * tf.pow(x, self.beta)
