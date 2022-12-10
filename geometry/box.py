from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


def construct_2d_plane_from_points(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    diff = p2 - p1
    normal_vec = np.array([-diff[1], diff[0]]).reshape((-1, 1))
    a = normal_vec / np.linalg.norm(normal_vec)
    b = a.T.dot(p1)
    return (a, b)


@dataclass
class Box2d:
    width: float = 3
    height: float = 2
    mass: float = 1

    # p1 -- p2
    # |     |
    # p4 -- p3

    @property
    def p1(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [self.height / 2]])

    @property
    def p2(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [self.height / 2]])

    @property
    def p3(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [-self.height / 2]])

    @property
    def p4(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [-self.height / 2]])

    @property
    def corners(self) -> List[npt.NDArray[np.float64]]:
        return [self.p1, self.p2, self.p3, self.p4]

    # p1 - a1 - p2
    # |          |
    # a4         a2
    # |          |
    # p4 --a3--- p3

    @property
    def a1(self) -> npt.NDArray[np.float64]:
        return construct_2d_plane_from_points(self.p1, self.p2)

    @property
    def a2(self) -> npt.NDArray[np.float64]:
        return construct_2d_plane_from_points(self.p2, self.p3)

    @property
    def a3(self) -> npt.NDArray[np.float64]:
        return construct_2d_plane_from_points(self.p3, self.p4)

    @property
    def a4(self) -> npt.NDArray[np.float64]:
        return construct_2d_plane_from_points(self.p4, self.p1)
