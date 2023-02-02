from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Hyperplane(NamedTuple):
    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]


def construct_2d_plane_from_points(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> Hyperplane:
    diff = p2 - p1
    normal_vec = np.array([-diff[1], diff[0]]).reshape((-1, 1))
    a = normal_vec / np.linalg.norm(normal_vec)
    b = a.T.dot(p1)
    return Hyperplane(a, b)
