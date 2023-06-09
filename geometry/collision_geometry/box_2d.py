from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Box as DrakeBox

from geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from geometry.hyperplane import Hyperplane, construct_2d_plane_from_points
from geometry.utilities import normalize_vec


@dataclass
class Box2d(CollisionGeometry):
    """
    Implements a two-dimensional box collision geometry.
    """

    width: float
    height: float
    # v0 -- v1
    # |     |
    # v3 -- v2

    @classmethod
    def from_drake(
        cls, drake_box: DrakeBox, axis_mode: Literal["planar"] = "planar"
    ) -> "Box2d":
        """
        Constructs a two-dimensional box from a Drake 3D box.

        By default, it is assumed that the box is intended to be used with planar pushing, and
        hence the two-dimensional box is constructed with the 'depth' and 'width' from the Drake box.
        """
        if axis_mode == "planar":
            width = drake_box.depth()
            height = drake_box.width()
            return cls(width, height)
        else:
            raise NotImplementedError(
                "Only planar conversion from 3D drake box is currently supported."
            )

    @property
    def _v0(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [self.height / 2]])

    @property
    def _v1(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [self.height / 2]])

    @property
    def _v2(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [-self.height / 2]])

    @property
    def _v3(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [-self.height / 2]])

    @property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        return [self._v0, self._v1, self._v2, self._v3]

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        return np.hstack([self._v0, self._v1, self._v2, self._v3])

    # v0 - f1 - v1
    # |          |
    # f4         f2
    # |          |
    # v3 --f3--- v2

    @property
    def _face_1(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v0, self._v1)

    @property
    def _face_2(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v1, self._v2)

    @property
    def _face_3(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v2, self._v3)

    @property
    def _face_4(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v3, self._v0)

    @property
    def faces(self) -> List[Hyperplane]:
        return [self._face_1, self._face_2, self._face_3, self._face_4]

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
        if location.pos == ContactLocation.FACE:
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
        elif location.pos == ContactLocation.VERTEX:
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
        if location.pos == ContactLocation.FACE:
            raise NotImplementedError(
                f"Can't get neighbouring vertices for face contact"
            )
        elif location.pos == ContactLocation.VERTEX:
            if location.idx == 1:
                return self._v3, self._v1
            elif location.idx == 2:
                return self._v0, self._v2
            elif location.idx == 3:
                return self._v1, self._v3
            elif location.idx == 4:
                return self._v2, self._v0
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
        if location.pos == ContactLocation.FACE:
            if location.idx == 1:
                return [self._v0, self._v1]
            elif location.idx == 2:
                return [self._v1, self._v2]
            elif location.idx == 3:
                return [self._v2, self._v3]
            elif location.idx == 4:
                return [self._v3, self._v0]
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        elif location.pos == ContactLocation.VERTEX:
            if location.idx == 1:
                return [self._v0]
            elif location.idx == 2:
                return [self._v1]
            elif location.idx == 3:
                return [self._v2]
            elif location.idx == 4:
                return [self._v3]
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
        if location.pos == ContactLocation.VERTEX:
            raise NotImplementedError(f"Can't get hyperplane for vertex contact")
        elif location.pos == ContactLocation.FACE:
            if location.idx == 1:
                return self._face_1
            elif location.idx == 2:
                return self._face_2
            elif location.idx == 3:
                return self._face_3
            elif location.idx == 4:
                return self._face_4
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_face_length(self, location: PolytopeContactLocation) -> float:
        if not location.pos == ContactLocation.FACE:
            raise ValueError("Can only get face length for a face")

        if location.idx == 1 or location.idx == 3:
            return self.width
        elif location.idx == 2 or location.idx == 4:
            return self.height
        else:
            raise ValueError(f"Can not get length for face {location.idx} for a box")
