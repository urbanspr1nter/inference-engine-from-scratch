import numpy as np


class PositionalEmbedding:
    def __init__(self, wpe_weight: np.ndarray):
        self._wpe = wpe_weight

    def forward(self, positions: np.ndarray) -> np.ndarray:
        return self._wpe[positions]
