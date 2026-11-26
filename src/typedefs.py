from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import tensorflow as tf
from jaxtyping import Float64, Int64

IdVec: TypeAlias = Int64[np.ndarray, " n"]
DataVec: TypeAlias = Float64[np.ndarray, " n"]
LossVec: TypeAlias = Float64[np.ndarray, " n_epochs"]
BatchVec: TypeAlias = Float64[np.ndarray, " b"]
IndexVec: TypeAlias = Int64[np.ndarray, " m"]

DataTensor: TypeAlias = Float64[tf.Tensor, " n"]
DesignTensor: TypeAlias = Float64[tf.Tensor, "n p"]
ScalarTensor: TypeAlias = Float64[tf.Tensor, ""]

Params: TypeAlias = Sequence[tf.Variable]
Grads: TypeAlias = Sequence[tf.Tensor]
Closure: TypeAlias = Callable[[], ScalarTensor]
