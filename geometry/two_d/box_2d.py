from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from geometry.hyperplane import Hyperplane, construct_2d_plane_from_points
from geometry.two_d.contact.types import ContactPosition
from geometry.two_d.rigid_body_2d import PolytopeContactLocation, RigidBody2d
from geometry.utilities import normalize_vec


@dataclass
class Box2d(RigidBody2d):
    width: float = 3
    height: float = 2

    # TODO: This class should be simplified similar to EquilateralPolytope2d

    # v1 -- v2
    # |     |
    # v4 -- v3

    @property
    def v1(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [self.height / 2]])

    @property
    def v2(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [self.height / 2]])

    @property
    def v3(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [-self.height / 2]])

    @property
    def v4(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [-self.height / 2]])

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        return np.hstack([self.v1, self.v2, self.v3, self.v4])

    # v1 - f1 - v2
    # |          |
    # f4         f2
    # |          |
    # v4 --f3--- v3

    @property
    def face_1(self) -> Hyperplane:
        return construct_2d_plane_from_points(self.v1, self.v2)

    @property
    def face_2(self) -> Hyperplane:
        return construct_2d_plane_from_points(self.v2, self.v3)

    @property
    def face_3(self) -> Hyperplane:
        return construct_2d_plane_from_points(self.v3, self.v4)

    @property
    def face_4(self) -> Hyperplane:
        return construct_2d_plane_from_points(self.v4, self.v1)

    #  --------------------
    #  |        |         |
    #  |        v n1      |
    #  |                  |
    #  | --> n4    n2 <-- |
    #  |                  |
    #  |        ^         |
    #  |        | n3      |
    #  --------------------

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

    # Right handed coordinate frame with z-axis out of plane and x-axis along normal
    #
    #           t1--->
    #       ---------
    #    ^  |       |
    # t4 |  |       | | t2
    #       |       | v
    #       ---------
    #       <--- t3

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
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        if location.pos == ContactPosition.FACE:
            if location.idx == 1:
                return self.n1, self.t1
            elif location.idx == 2:
                return self.n2, self.t2
            elif location.idx == 3:
                return self.n3, self.t3
            elif location.idx == 4:
                return self.n4, self.t4
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        elif location.pos == ContactPosition.VERTEX:
            if location.idx == 1:
                return self.nc1, self.tc1
            elif location.idx == 2:
                return self.nc2, self.tc2
            elif location.idx == 3:
                return self.nc3, self.tc3
            elif location.idx == 4:
                return self.nc4, self.tc4
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_neighbouring_vertices(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if location.pos == ContactPosition.FACE:
            raise NotImplementedError(
                f"Can't get neighbouring vertices for face contact"
            )
        elif location.pos == ContactPosition.VERTEX:
            if location.idx == 1:
                return self.v4, self.v2
            elif location.idx == 2:
                return self.v1, self.v3
            elif location.idx == 3:
                return self.v2, self.v4
            elif location.idx == 4:
                return self.v3, self.v1
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        if location.pos == ContactPosition.FACE:
            if location.idx == 1:
                return [self.v1, self.v2]
            elif location.idx == 2:
                return [self.v2, self.v3]
            elif location.idx == 3:
                return [self.v3, self.v4]
            elif location.idx == 4:
                return [self.v4, self.v1]
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        elif location.pos == ContactPosition.VERTEX:
            if location.idx == 1:
                return [self.v1]
            elif location.idx == 2:
                return [self.v2]
            elif location.idx == 3:
                return [self.v3]
            elif location.idx == 4:
                return [self.v4]
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        if location.pos == ContactPosition.VERTEX:
            raise NotImplementedError(f"Can't get hyperplane for vertex contact")
        elif location.pos == ContactPosition.FACE:
            if location.idx == 1:
                return self.face_1
            elif location.idx == 2:
                return self.face_2
            elif location.idx == 3:
                return self.face_3
            elif location.idx == 4:
                return self.face_4
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )
