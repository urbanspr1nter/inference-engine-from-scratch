import numpy as np
from model.util import gelu


class Mlp:
    def __init__(
        self,
        c_fc_w: np.ndarray,
        c_fc_b: np.ndarray,
        c_proj_w: np.ndarray,
        c_proj_b: np.ndarray,
    ):
        self._c_fc_w = c_fc_w
        self._c_fc_b = c_fc_b
        self._c_proj_w = c_proj_w
        self._c_proj_b = c_proj_b

    def forward(self, x: np.ndarray):
        activation = x @ self._c_fc_w + self._c_fc_b
        out = gelu(activation)

        return out @ self._c_proj_w + self._c_proj_b
