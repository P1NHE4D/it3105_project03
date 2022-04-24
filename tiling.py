import numpy as np


def tile(state: np.ndarray, w, displacements=None, n=None):
    if n is None:
        n = 2 ** np.log2(4 * state.shape[0])
    if displacements is None:
        displacements = [2 * k - 1 for k in range(1, state.shape[0] + 1)]
    offset = w / n
