from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Box as DrakeBox
from pydrake.math import sqrt

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    construct_2d_plane_from_points,
)
from planning_through_contact.geometry.utilities import cross_2d, normalize_vec


@dataclass(frozen=True)
class Box2d(CollisionGeometry):
    """
    Implements a two-dimensional box collision geometry.
    """

    width: float
    height: float
    # v0 -- v1
    # |     |
    # v3 -- v2

    @property
    def collision_geometry_names(self) -> List[str]:
        return ["box::box_collision"]

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

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        return [self.faces[idx]]

    @property
    def num_collision_free_regions(self) -> int:
        return 4

    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
        return loc_idx  # for boxes, we have one collision-free region per face

    def get_planes_for_collision_free_region(self, idx: int) -> List[Hyperplane]:
        planes = []
        if idx == 0:
            planes.append(construct_2d_plane_from_points(self._v0, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v1))
        elif idx == 1:
            planes.append(construct_2d_plane_from_points(self._v1, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v2))
        elif idx == 2:
            planes.append(construct_2d_plane_from_points(self._v2, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v3))
        elif idx == 3:
            planes.append(construct_2d_plane_from_points(self._v3, self._com))
            planes.append(construct_2d_plane_from_points(self._com, self._v0))
        else:
            raise ValueError(f"Can not get collision free region for idx {idx}")

        return planes

    def get_signed_distance(self, pos: npt.NDArray) -> float:
        """
        Returns the signed distance from the pos to the closest point on the box.

        @param pos: position relative to the COM of the box.
        """

        if len(pos.shape) == 1:
            pos = pos.reshape((-1, 1))

        pos_x = pos[0, 0]
        pos_y = pos[1, 0]

        # Left
        if (
            pos_x <= -self.width / 2
            and pos_y >= -self.height / 2
            and pos_y <= self.height / 2
        ):
            return -pos_x - self.width / 2
        # Right
        elif (
            pos_x >= self.width / 2
            and pos_y >= -self.height / 2
            and pos_y <= self.height / 2
        ):
            return pos_x - self.width / 2
        # Top
        elif (
            pos_y >= self.height / 2
            and pos_x >= -self.width / 2
            and pos_x <= self.width / 2
        ):
            return pos_y - self.height / 2
        # Bottom
        elif (
            pos_y <= -self.height / 2
            and pos_x >= -self.width / 2
            and pos_x <= self.width / 2
        ):
            return -pos_y - self.height / 2

        # Bottom left corner
        elif pos_y <= -self.height / 2 and pos_x <= -self.width / 2:
            diff = pos - self.vertices[3]
            dist = sqrt(diff[0, 0] ** 2 + diff[1, 0] ** 2)
            return dist

        # Top left corner
        elif pos_y >= self.height / 2 and pos_x <= -self.width / 2:
            diff = pos - self.vertices[0]
            dist = sqrt(diff[0, 0] ** 2 + diff[1, 0] ** 2)
            return dist

        # Top right corner
        elif pos_y >= self.height / 2 and pos_x >= self.width / 2:
            diff = pos - self.vertices[1]
            dist = sqrt(diff[0, 0] ** 2 + diff[1, 0] ** 2)
            return dist

        # Bottom right corner
        elif pos_y <= -self.height / 2 and pos_x >= self.width / 2:
            diff = pos - self.vertices[2]
            dist = sqrt(diff[0, 0] ** 2 + diff[1, 0] ** 2)
            return dist
        else:  # inside box
            dist_to_left = self.width / 2 + pos_x
            dist_to_right = self.width / 2 - pos_x

            dist_to_top = self.height / 2 - pos_y
            dist_to_bottom = self.height / 2 + pos_y

            return -min((dist_to_top, dist_to_bottom, dist_to_left, dist_to_right))

    def get_contact_jacobian(self, pos: npt.NDArray) -> npt.NDArray[Any]:
        """
        Returns the contact jacobian for the point that is closest to the box.
        If the closest point is on a corner, one of the corner faces is picked
        arbitrarily (but deterministically).

        @param pos: position relative to the COM of the box.
        """

        def _make_jacobian(
            normal_vec: npt.NDArray[np.float64],
            tangent_vec: npt.NDArray[np.float64],
            pos: npt.NDArray[Any],
        ) -> npt.NDArray[Any]:
            col_1 = np.vstack((normal_vec, cross_2d(pos, normal_vec)))
            col_2 = np.vstack((tangent_vec, cross_2d(pos, tangent_vec)))
            J_T = np.hstack([col_1, col_2])
            return J_T.T

        if len(pos.shape) == 1:
            pos = pos.reshape((-1, 1))

        pos_x = pos[0, 0]
        pos_y = pos[1, 0]

        TOL = 1e-12

        planes_left = self.get_planes_for_collision_free_region(3)
        planes_right = self.get_planes_for_collision_free_region(1)
        planes_top = self.get_planes_for_collision_free_region(0)
        planes_bottom = self.get_planes_for_collision_free_region(2)
        # Left
        if planes_left[0].dist_to(pos) >= 0 and planes_left[1].dist_to(pos) >= 0:
            return _make_jacobian(self.normal_vecs[3], self.tangent_vecs[3], pos)
        # Right
        elif planes_right[0].dist_to(pos) >= 0 and planes_right[1].dist_to(pos) >= 0:
            return _make_jacobian(self.normal_vecs[1], self.tangent_vecs[1], pos)
        # Top
        elif planes_top[0].dist_to(pos) >= 0 and planes_top[1].dist_to(pos) >= 0:
            return _make_jacobian(self.normal_vecs[0], self.tangent_vecs[0], pos)
        # Bottom
        elif planes_bottom[0].dist_to(pos) >= 0 and planes_bottom[1].dist_to(pos) >= 0:
            return _make_jacobian(self.normal_vecs[2], self.tangent_vecs[2], pos)
        else:  # inside box we just return zero
            # TODO: Could this potentially confuse the solver?
            return np.zeros((2, 3))
