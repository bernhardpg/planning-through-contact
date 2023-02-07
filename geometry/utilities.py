import numpy as np
import numpy.typing as npt


def cross_2d(v1, v2):
    return (v1[0] * v2[1] - v1[1] * v2[0])[0]


def normalize_vec(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vec / np.linalg.norm(vec)
