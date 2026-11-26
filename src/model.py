import numpy as np

from src.typedefs import DataVec, DesignMat, ParamVec


def make_design_matrix(x: DataVec) -> DesignMat:
    """1차 다항의 설계행렬. 각 행은 [1, x]이다."""
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack([np.ones_like(x), x])


def model(X: DesignMat, theta: ParamVec) -> DataVec:
    return X @ theta


def loss(X: DesignMat, y: DataVec, theta: ParamVec) -> float:
    """평균 제곱오차. L2 손실이다."""
    residual = model(X, theta) - y
    return np.mean(residual**2)


def loss_grad(X: DesignMat, y: DataVec, theta: ParamVec) -> ParamVec:
    """loss = mean((X theta - y)^2) 의 해석적 gradient.

    d(loss)/d(theta) = (2 / n) X^T (X theta - y)
    """
    residual = model(X, theta) - y
    return (2.0 / y.shape[0]) * (X.T @ residual)


def loss_grad_tape(X: DesignMat, y: DataVec, theta: ParamVec) -> ParamVec:
    """같은 gradient를 tf.GradientTape로 계산하는 예제.

    기본 학습 루프는 loss_grad를 사용한다. TensorFlow가 설치된 환경에서
    이 함수로 바꾼 뒤에도 갱신 식은 gradient_descent 안에 직접 둔다.
    """
    import tensorflow as tf

    x_tf = tf.constant(X, dtype=tf.float64)
    y_tf = tf.constant(y, dtype=tf.float64)
    theta_tf = tf.Variable(tf.constant(theta, dtype=tf.float64))
    with tf.GradientTape() as tape:
        residual = tf.linalg.matvec(x_tf, theta_tf) - y_tf
        value = tf.reduce_mean(residual**2)
    grad = tape.gradient(value, theta_tf)
    return tf.convert_to_tensor(grad).numpy()
