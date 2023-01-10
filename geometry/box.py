from dataclasses import dataclass
from typing import Tuple, Union

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


def normalize_vec(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vec / np.linalg.norm(vec)


@dataclass
class Box2d:
    width: float = 3
    height: float = 2

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
    def corners(self) -> npt.NDArray[np.float64]:
        return np.hstack([self.p1, self.p2, self.p3, self.p4])

    # p1 - a1 - p2
    # |          |
    # a4         a2
    # |          |
    # p4 --a3--- p3

    @property
    def a1(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return construct_2d_plane_from_points(self.p1, self.p2)

    @property
    def a2(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return construct_2d_plane_from_points(self.p2, self.p3)

    @property
    def a3(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return construct_2d_plane_from_points(self.p3, self.p4)

    @property
    def a4(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return construct_2d_plane_from_points(self.p4, self.p1)

    # NOTE! All of these arrows are the wrong ways

    #           ^
    #           | n1
    #       ---------
    #       |       |
    # n4 <- |       | -> n2
    #       |       |
    #       ---------
    #           |
    #           V n3

    @property
    def n1(self) -> npt.NDArray[np.float64]:
        return -np.array([0, 1]).reshape((-1, 1))

    @property
    def n2(self) -> npt.NDArray[np.float64]:
        return -np.array([1, 0]).reshape((-1, 1))

    @property
    def n3(self) -> npt.NDArray[np.float64]:
        return -np.array([0, -1]).reshape((-1, 1))

    @property
    def n4(self) -> npt.NDArray[np.float64]:
        return -np.array([-1, 0]).reshape((-1, 1))

    #       <--t1
    #       ---------
    #       |       | ^
    # t4 |  |       | | t2
    #    V  |       |
    #       ---------
    #           --> t3

    @property
    def t1(self) -> npt.NDArray[np.float64]:
        return self.n4

    @property
    def t2(self) -> npt.NDArray[np.float64]:
        return self.n1

    @property
    def t3(self) -> npt.NDArray[np.float64]:
        return self.n2

    @property
    def t4(self) -> npt.NDArray[np.float64]:
        return self.n3

    # Corner normal vectors
    # nc1 -- nc2
    # |       |
    # nc4 -- nc3

    @property
    def nc1(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self.n4 + self.n1)

    @property
    def nc2(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self.n1 + self.n2)

    @property
    def nc3(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self.n2 + self.n3)

    @property
    def nc4(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self.n3 + self.n4)

    @property
    def tc1(self) -> npt.NDArray[np.float64]:
        return self.nc4

    @property
    def tc2(self) -> npt.NDArray[np.float64]:
        return self.nc1

    @property
    def tc3(self) -> npt.NDArray[np.float64]:
        return self.nc2

    @property
    def tc4(self) -> npt.NDArray[np.float64]:
        return self.nc3

    def get_norm_and_tang_vecs_from_location(
        self, location: str
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if location == "face_1":
            return self.n1, self.t1
        elif location == "face_2":
            return self.n2, self.t2
        elif location == "face_3":
            return self.n3, self.t3
        elif location == "face_4":
            return self.n4, self.t4

        elif location == "corner_1":
            return self.nc1, self.tc1
        elif location == "corner_2":
            return self.nc2, self.tc2
        elif location == "corner_3":
            return self.nc3, self.tc3
        elif location == "corner_4":
            return self.nc4, self.tc4
        else:
            raise NotImplementedError(f"Location {location} not yet implemented")

    def get_proximate_vertices_from_location(
        self, location: str
    ) -> Union[
        npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
        if location == "face_1":
            return self.p1, self.p2
        elif location == "face_2":
            return self.p2, self.p3
        elif location == "face_3":
            return self.p3, self.p4
        elif location == "face_4":
            return self.p4, self.p1

        elif location == "corner_1":
            return self.p1
        elif location == "corner_2":
            return self.p2
        elif location == "corner_3":
            return self.p3
        elif location == "corner_4":
            return self.p4
        else:
            raise NotImplementedError(f"Location {location} not yet implemented")
