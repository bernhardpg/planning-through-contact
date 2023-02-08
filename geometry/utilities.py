import numpy as np
import numpy.typing as npt


def cross_2d(v1, v2):
    return (v1[0] * v2[1] - v1[1] * v2[0])[0]


def normalize_vec(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vec / np.linalg.norm(vec)


def two_d_rotation_matrix_from_angle(theta: float) -> npt.NDArray[np.float64]:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R
