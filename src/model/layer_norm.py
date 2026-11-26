import numpy as np


class LayerNorm:
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self._weights = weights
        self._bias = bias

    def forward(self, x: np.ndarray):
        h = (x - x.mean(-1, keepdims=True)) / (np.sqrt(x.var(-1, keepdims=True) + 1e-5))

        out = h * self._weights + self._bias

        return out
