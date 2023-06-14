import numpy as np
import numpy.typing as npt


def unit_vector(i: int, n: int):
    e_i = np.zeros(n)
    e_i[i] = 1
    return e_i


def cross_2d(v1, v2):
    return (v1[0] * v2[1] - v1[1] * v2[0])[0]


def normalize_vec(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vec / np.linalg.norm(vec)


def two_d_rotation_matrix_from_angle(theta: float) -> npt.NDArray[np.float64]:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


def from_so2_to_so3(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a SO(2) rotation matrix and returns a rotation matrix in SO(3), where the original matrix
    is treated as a rotation about the z-axis.
    """
    R_in_SO3 = np.eye(3)
    R_in_SO3[0:2, 0:2] = R
    return R_in_SO3
