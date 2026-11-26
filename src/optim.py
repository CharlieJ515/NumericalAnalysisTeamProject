from abc import ABC, abstractmethod

import tensorflow as tf

from src.typedefs import Closure, Grads, Params


def zeros_like_params(params: Params) -> list[tf.Variable]:
    return [tf.Variable(tf.zeros_like(param), trainable=False) for param in params]


class Optimizer(ABC):
    def __init__(self, params: Params, lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    @abstractmethod
    def step(self, grads: Grads, closure: Closure | None = None) -> None: ...


class SGD(Optimizer):
    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        for param, grad in zip(self.params, grads, strict=True):
            param.assign_sub(self.lr * grad)


class Momentum(Optimizer):
    def __init__(self, params: Params, lr: float, alpha: float = 0.9) -> None:
        super().__init__(params, lr)
        self.alpha = alpha
        self.velocity = zeros_like_params(self.params)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        for param, grad, v in zip(self.params, grads, self.velocity, strict=True):
            v.assign(self.alpha * v - self.lr * grad)
            param.assign_add(v)


class Nesterov(Optimizer):
    def __init__(self, params: Params, lr: float, alpha: float = 0.9) -> None:
        super().__init__(params, lr)
        self.alpha = alpha
        self.velocity = zeros_like_params(self.params)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        for param, grad, v in zip(self.params, grads, self.velocity, strict=True):
            v.assign(self.alpha * v - self.lr * grad)
            param.assign_add(self.alpha * v - self.lr * grad)


class Adagrad(Optimizer):
    def __init__(self, params: Params, lr: float, delta: float = 1e-7) -> None:
        super().__init__(params, lr)
        self.delta = delta
        self.r = zeros_like_params(self.params)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        for param, grad, r in zip(self.params, grads, self.r, strict=True):
            r.assign_add(grad * grad)
            param.assign_sub(self.lr / (self.delta + tf.sqrt(r)) * grad)


class RMSProp(Optimizer):
    def __init__(self, params: Params, lr: float, rho: float = 0.9, delta: float = 1e-7) -> None:
        super().__init__(params, lr)
        self.rho = rho
        self.delta = delta
        self.r = zeros_like_params(self.params)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        for param, grad, r in zip(self.params, grads, self.r, strict=True):
            r.assign(self.rho * r + (1 - self.rho) * grad * grad)
            param.assign_sub(self.lr / tf.sqrt(self.delta + r) * grad)


class Adam(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float,
        rho1: float = 0.9,
        rho2: float = 0.999,
        delta: float = 1e-7,
    ) -> None:
        super().__init__(params, lr)
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta
        self.s = zeros_like_params(self.params)
        self.r = zeros_like_params(self.params)
        self.k = tf.Variable(tf.constant(0.0, dtype=tf.float64), trainable=False)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        self.k.assign_add(1.0)
        s_correction = 1 - tf.pow(tf.constant(self.rho1, dtype=tf.float64), self.k)
        r_correction = 1 - tf.pow(tf.constant(self.rho2, dtype=tf.float64), self.k)

        for param, grad, s, r in zip(self.params, grads, self.s, self.r, strict=True):
            s.assign(self.rho1 * s + (1 - self.rho1) * grad)
            r.assign(self.rho2 * r + (1 - self.rho2) * grad * grad)
            s_hat = s / s_correction
            r_hat = r / r_correction
            param.assign_sub(self.lr * s_hat / (self.delta + tf.sqrt(r_hat)))


class AdaptiveSGD(Optimizer):
    def __init__(self, params: Params, lr: float) -> None:
        super().__init__(params, lr)
        self.current_lr = tf.Variable(tf.constant(lr, dtype=tf.float64), trainable=False)

    def step(self, grads: Grads, closure: Closure | None = None) -> None:
        if closure is None:
            raise ValueError("AdaptiveSGD는 전체 loss를 계산하는 closure가 필요합니다.")

        previous = [tf.identity(param) for param in self.params]
        loss_before = closure()

        for param, grad in zip(self.params, grads, strict=True):
            param.assign_sub(self.current_lr * grad)
        loss_after = closure()

        improved = loss_before > loss_after
        for param, old in zip(self.params, previous, strict=True):
            param.assign(tf.where(improved, param, old))
        self.current_lr.assign(tf.where(improved, self.current_lr, self.current_lr / 2))
