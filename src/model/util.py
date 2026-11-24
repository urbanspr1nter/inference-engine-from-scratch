import math

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (tanh approximation), as used by GPT-2."""
    c = math.sqrt(2 / math.pi)  # Python float: keeps float32 inputs float32
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))
