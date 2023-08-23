from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Box as DrakeBox

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    construct_2d_plane_from_points,
)
from planning_through_contact.geometry.utilities import normalize_vec


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
    def contact_locations(self) -> List[PolytopeContactLocation]:
        locs = [
            PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)
            for idx in range(len(self.faces))
        ]
        return locs

    @property
    def _com(self) -> npt.NDArray[np.float64]:
        return np.zeros((2, 1))

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

    # v0 - f0 - v1
    # |          |
    # f3         f1
    # |          |
    # v3 --f2--- v2

    @property
    def _face_0(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v0, self._v1)

    @property
    def _face_1(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v1, self._v2)

    @property
    def _face_2(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v2, self._v3)

    @property
    def _face_3(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v3, self._v0)

    @property
    def faces(self) -> List[Hyperplane]:
        return [self._face_0, self._face_1, self._face_2, self._face_3]

    #  --------------------
    #  |        |         |
    #  |        v n0      |
    #  |                  |
    #  | --> n3    n1 <-- |
    #  |                  |
    #  |        ^         |
    #  |        | n2      |
    #  --------------------

    @property
    def _n0(self) -> npt.NDArray[np.float64]:
        return -np.array([0, 1]).reshape((-1, 1))

    @property
    def _n1(self) -> npt.NDArray[np.float64]:
        return -np.array([1, 0]).reshape((-1, 1))

    @property
    def _n2(self) -> npt.NDArray[np.float64]:
        return -np.array([0, -1]).reshape((-1, 1))

    @property
    def _n3(self) -> npt.NDArray[np.float64]:
        return -np.array([-1, 0]).reshape((-1, 1))

    @property
    def normal_vecs(self) -> List[npt.NDArray[np.float64]]:
        return [self._n0, self._n1, self._n2, self._n3]

    # Right handed coordinate frame with z-axis out of plane and x-axis along normal
    #
    #           t0--->
    #       ---------
    #    ^  |       |
    # t3 |  |       | | t1
    #       |       | v
    #       ---------
    #       <--- t2

    @property
    def _t0(self) -> npt.NDArray[np.float64]:
        return self._n3

    @property
    def _t1(self) -> npt.NDArray[np.float64]:
        return self._n0

    @property
    def _t2(self) -> npt.NDArray[np.float64]:
        return self._n1

    @property
    def _t3(self) -> npt.NDArray[np.float64]:
        return self._n2

    @property
    def tangent_vecs(self) -> List[npt.NDArray[np.float64]]:
        return [self._t0, self._t1, self._t2, self._t3]

    # Corner normal vectors
    # nc0 -- nc1
    # |       |
    # nc3 -- nc2

    @property
    def _nc0(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self._n3 + self._n0)

    @property
    def _nc1(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self._n0 + self._n1)

    @property
    def _nc2(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self._n1 + self._n2)

    @property
    def _nc3(self) -> npt.NDArray[np.float64]:
        return normalize_vec(self._n2 + self._n3)

    @property
    def _tc0(self) -> npt.NDArray[np.float64]:
        return self._nc3

    @property
    def _tc1(self) -> npt.NDArray[np.float64]:
        return self._nc0

    @property
    def _tc2(self) -> npt.NDArray[np.float64]:
        return self._nc1

    @property
    def _tc3(self) -> npt.NDArray[np.float64]:
        return self._nc2

    def get_norm_and_tang_vecs_from_location(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if location.pos == ContactLocation.FACE:
            if location.idx == 0:
                return self._n0, self._t0
            elif location.idx == 1:
                return self._n1, self._t1
            elif location.idx == 2:
                return self._n2, self._t2
            elif location.idx == 3:
                return self._n3, self._t3
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        elif location.pos == ContactLocation.VERTEX:
            if location.idx == 0:
                return self._nc0, self._tc0
            elif location.idx == 1:
                return self._nc1, self._tc1
            elif location.idx == 2:
                return self._nc2, self._tc2
            elif location.idx == 3:
                return self._nc3, self._tc3
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
            if location.idx == 0:
                return self._v3, self._v1
            elif location.idx == 1:
                return self._v0, self._v2
            elif location.idx == 2:
                return self._v1, self._v3
            elif location.idx == 3:
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
            if location.idx == 0:
                return [self._v0, self._v1]
            elif location.idx == 1:
                return [self._v1, self._v2]
            elif location.idx == 2:
                return [self._v2, self._v3]
            elif location.idx == 3:
                return [self._v3, self._v0]
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        elif location.pos == ContactLocation.VERTEX:
            if location.idx == 0:
                return [self._v0]
            elif location.idx == 1:
                return [self._v1]
            elif location.idx == 2:
                return [self._v2]
            elif location.idx == 3:
                return [self._v3]
            else:
                raise NotImplementedError(
                    f"Location {location.pos}: {location.idx} not implemented"
                )
        else:
            breakpoint()
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        if location.pos == ContactLocation.VERTEX:
            raise NotImplementedError(f"Can't get hyperplane for vertex contact")
        elif location.pos == ContactLocation.FACE:
            if location.idx == 0:
                return self._face_0
            elif location.idx == 1:
                return self._face_1
            elif location.idx == 2:
                return self._face_2
            elif location.idx == 3:
                return self._face_3
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

        if location.idx == 0 or location.idx == 2:
            return self.width
        elif location.idx == 1 or location.idx == 3:
            return self.height
        else:
            raise ValueError(f"Can not get length for face {location.idx} for a box")

    def get_planes_for_collision_free_region(
        self, location: PolytopeContactLocation
    ) -> List[Hyperplane]:
        if not location.pos == ContactLocation.FACE:
            raise ValueError("Can only get collision free region for a face")

        planes = [
            self.faces[location.idx]
        ]  # we always want the hyperplane for the current face
        if location.idx == 0:
            planes.append(construct_2d_plane_from_points(self._v0, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v1))
        elif location.idx == 1:
            planes.append(construct_2d_plane_from_points(self._v1, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v2))
        elif location.idx == 2:
            planes.append(construct_2d_plane_from_points(self._v2, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v3))
        elif location.idx == 3:
            planes.append(construct_2d_plane_from_points(self._v3, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v0))
        else:
            raise ValueError(f"Can not get collision free region for {location}")

        return planes

    def get_p_c_B_from_lam(
        self, lam: float, loc: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        assert loc.pos == ContactLocation.FACE
        pv1, pv2 = self.get_proximate_vertices_from_location(loc)
        return lam * pv1 + (1 - lam) * pv2
