from typing import TypeAlias

import numpy as np
from jaxtyping import Float64, Int64

IdVec: TypeAlias = Int64[np.ndarray, " n"]
DataVec: TypeAlias = Float64[np.ndarray, " n"]
DesignMat: TypeAlias = Float64[np.ndarray, "n p"]
ParamVec: TypeAlias = Float64[np.ndarray, " p"]
LossVec: TypeAlias = Float64[np.ndarray, " n_epochs"]
BatchVec: TypeAlias = Float64[np.ndarray, " b"]
BatchMat: TypeAlias = Float64[np.ndarray, "b p"]