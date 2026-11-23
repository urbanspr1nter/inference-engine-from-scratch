import numpy as np


class TokenEmbedding:
    def __init__(self, wte_weight: np.ndarray):
        self._wte = wte_weight

    def forward(self, token_ids: np.ndarray):
        return self._wte[token_ids]
