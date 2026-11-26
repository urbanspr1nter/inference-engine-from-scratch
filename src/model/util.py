import math

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))
