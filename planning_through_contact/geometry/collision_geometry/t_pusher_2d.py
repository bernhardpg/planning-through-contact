from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape
from pydrake.math import RigidTransform, RotationMatrix

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
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
class TPusher2d(CollisionGeometry):
    """
    Constructed such that box 1 stacks on top, and box 2 lies on the bottom:

     ____________
    |   box 1    |
    |____________|
        | b2 |
        |    |
        |    |
        |____|

    Origin is placed at the center of box 1.

    """

    box_1: Box2d = field(default_factory=lambda: Box2d(0.2, 0.05))
    box_2: Box2d = field(default_factory=lambda: Box2d(0.05, 0.15))

    @classmethod
    def from_drake(cls, drake_shape: DrakeShape):
        raise NotImplementedError()

    @property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        """
        v0___________v1
        |              |
        v7___v6___v3___v2
            |    |
            |    |
            |    |
            v5____v4

        """
        v0 = self.box_1.vertices[0]
        v1 = self.box_1.vertices[1]
        v2 = self.box_1.vertices[2]

        box_2_center = np.array(
            [0, -self.box_1.height / 2 - self.box_2.height / 2]
        ).reshape((-1, 1))
        v3 = box_2_center + self.box_2.vertices[1]
        v4 = box_2_center + self.box_2.vertices[2]
        v5 = box_2_center + self.box_2.vertices[3]
        v6 = box_2_center + self.box_2.vertices[0]

        v7 = self.box_1.vertices[3]
        return [v0, v1, v2, v3, v4, v5, v6, v7]

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def faces(self) -> List[Hyperplane]:
        """
        ______f0________
        f7              f1
        |_f6________f2__|
            |    |
            f5   f3
            |    |
            |_f4_|

        """
        wrap_around = lambda num: num % self.num_vertices
        pairwise_indices = [
            (idx, wrap_around(idx + 1)) for idx in range(self.num_vertices)
        ]
        hyperplane_points = [
            (self.vertices[i], self.vertices[j]) for i, j in pairwise_indices
        ]
        hyperplanes = [
            construct_2d_plane_from_points(p1, p2) for p1, p2 in hyperplane_points
        ]
        return hyperplanes

    def get_collision_free_region_for_loc_idx(
        self, loc_idx: PolytopeContactLocation
    ) -> int:
        if loc_idx == 0:
            return 0
        elif loc_idx == 1:
            return 1
        elif loc_idx == 2:
            return 2
        elif loc_idx == 3:
            return 2
        elif loc_idx == 4:
            return 3
        elif loc_idx == 5:
            return 4
        elif loc_idx == 6:
            return 4
        elif loc_idx == 7:
            return 5
        else:
            raise ValueError(f"No collision-free region for loc_idx {loc_idx}")

    @property
    def contact_locations(self) -> List[PolytopeContactLocation]:
        # TODO(bernhardpg): Only returns FACEs, should ideally return
        # both vertices and faces
        locs = [
            PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)
            for idx in range(len(self.faces))
        ]
        return locs

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        """
        Gets the contact faces for each collision-free set.
        This function is hand designed for the object geometry.
        """
        if idx == 0:
            return [self.faces[0]]
        elif idx == 1:
            return [self.faces[1]]
        elif idx == 2:
            return [self.faces[2], self.faces[3]]
        elif idx == 3:
            return [self.faces[4]]
        elif idx == 4:
            return [self.faces[5], self.faces[6]]
        elif idx == 5:
            return [self.faces[7]]
        else:
            raise ValueError(f"No collision-free region for idx {idx}")

    @property
    def num_collision_free_regions(self) -> int:
        return 6

    def get_planes_for_collision_free_region(self, idx: int) -> List[Hyperplane]:
        """
        Gets the faces that defines the collision free sets (except for the contact face)
        This function is hand designed for the object geometry.
       
           \       0      /
            \____________/
        5   |            | 1
            |____________|
           /    |    |    \
          /     |    | 2   \
           4    |    |
                |____|
               /      \
              /    3   \
            
        """
        UL = np.array([-1, 1]).reshape((-1, 1))
        UR = np.array([1, 1]).reshape((-1, 1))
        DR = np.array([1, -1]).reshape((-1, 1))
        DL = np.array([-1, -1]).reshape((-1, 1))

        planes = []
        if idx == 0:
            planes.append(
                construct_2d_plane_from_points(self.vertices[0] + UL, self.vertices[0])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[1], self.vertices[1] + UR)
            )
            return planes
        if idx == 1:
            planes.append(
                construct_2d_plane_from_points(self.vertices[1] + UR, self.vertices[1])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[2], self.vertices[2] + DR)
            )
            return planes
        if idx == 2:
            planes.append(
                construct_2d_plane_from_points(self.vertices[2] + DR, self.vertices[2])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[4], self.vertices[4] + DR)
            )
            return planes
        if idx == 3:
            planes.append(
                construct_2d_plane_from_points(self.vertices[4] + DR, self.vertices[4])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[5], self.vertices[5] + DL)
            )
            return planes
        if idx == 4:
            planes.append(
                construct_2d_plane_from_points(self.vertices[5] + DL, self.vertices[5])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[7], self.vertices[7] + DL)
            )
            return planes
        if idx == 5:
            planes.append(
                construct_2d_plane_from_points(self.vertices[7] + DL, self.vertices[7])
            )
            planes.append(
                construct_2d_plane_from_points(self.vertices[0], self.vertices[0] + UL)
            )
            return planes
        else:
            raise NotImplementedError("Currently only face 0 is supported")

    # TODO: All of the following code is copied straight from equilateralpolytope and should be unified!

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        vertices = np.hstack([self.vertices[idx] for idx in range(self.num_vertices)])
        return vertices

    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        if location.pos == ContactLocation.FACE:
            wrap_around = lambda num: num % self.num_vertices
            return [
                self.vertices[location.idx],
                self.vertices[wrap_around(location.idx + 1)],
            ]
        elif location.pos == ContactLocation.VERTEX:
            return [self.vertices[location.idx]]
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_neighbouring_vertices(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if location.pos == ContactLocation.VERTEX:
            wrap_around = lambda num: num % self.num_vertices
            idx_prev = wrap_around(location.idx - 1)
            idx_next = wrap_around(location.idx + 1)

            return self.vertices[idx_prev], self.vertices[idx_next]
        else:
            raise NotImplementedError(
                f"Location {location.pos}: {location.idx} not implemented"
            )

    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        if location.pos == ContactLocation.FACE:
            return self.faces[location.idx]
        else:
            raise NotImplementedError(
                f"Cannot get hyperplane from location {location.pos}: {location.idx}"
            )

    @property
    def normal_vecs(self) -> List[npt.NDArray[np.float64]]:
        normals = [
            -face.a for face in self.faces
        ]  # Normal vectors point into the object
        return normals

    @staticmethod
    def _get_tangent_vec(v: npt.NDArray[np.float64]):
        """
        Returns the 2d tangent vector calculated as the cross product with the z-axis pointing out of the plane.
        """
        return np.array([-v[1], v[0]]).reshape((-1, 1))

    @property
    def tangent_vecs(self) -> List[npt.NDArray[np.float64]]:
        tangents = [
            self._get_tangent_vec(self.normal_vecs[idx])
            for idx in range(self.num_vertices)
        ]
        return tangents

    @property
    def corner_normal_vecs(self) -> List[npt.NDArray[np.float64]]:
        wrap_around = lambda idx: idx % self.num_vertices
        indices = [(wrap_around(idx - 1), idx) for idx in range(self.num_vertices)]
        corner_normals = [
            normalize_vec(self.normal_vecs[i] + self.normal_vecs[j]) for i, j in indices
        ]
        return corner_normals

    @property
    def corner_tangent_vecs(self) -> List[npt.NDArray[np.float64]]:
        tangents = [
            self._get_tangent_vec(self.corner_normal_vecs[idx])
            for idx in range(self.num_vertices)
        ]
        return tangents

    def get_norm_and_tang_vecs_from_location(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if location.pos == ContactLocation.FACE:
            return self.normal_vecs[location.idx], self.tangent_vecs[location.idx]
        elif location.pos == ContactLocation.VERTEX:
            return (
                self.corner_normal_vecs[location.idx],
                self.corner_tangent_vecs[location.idx],
            )
        else:
            raise ValueError(
                f"Cannot get normal and tangent vecs from location {location.pos}"
            )

    def get_face_length(self, location: PolytopeContactLocation) -> float:
        raise NotImplementedError("Face length not yet implemented for the TPusher.")

    def get_p_B_c_from_lam(
        self, lam: float, loc: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        assert loc.pos == ContactLocation.FACE
        pv1, pv2 = self.get_proximate_vertices_from_location(loc)
        return lam * pv1 + (1 - lam) * pv2

    def get_as_boxes(
        self, z_value: float = 0.0
    ) -> Tuple[List[Box2d], List[RigidTransform]]:
        # NOTE(bernhardpg): Hardcoded for this specific geometry
        # BOX_1 is the vertical box
        BOX_1_WIDTH = 2
        BOX_1_HEIGHT = 4
        box_1 = Box2d(BOX_1_WIDTH * self.scale, BOX_1_HEIGHT * self.scale)
        transform_1 = RigidTransform(RotationMatrix.Identity(), np.array([0, -1, z_value / self.scale]) * self.scale)  # type: ignore

        # BOX_2 is the horisontal box
        BOX_2_WIDTH = 6
        BOX_2_HEIGHT = 2
        box_2 = Box2d(BOX_2_WIDTH * self.scale, BOX_2_HEIGHT * self.scale)
        transform_2 = RigidTransform(RotationMatrix.Identity(), np.array([0, 2, z_value / self.scale]) * self.scale)  # type: ignore

        return [box_1, box_2], [transform_1, transform_2]

    def get_lam_from_p_B_c(
        self, lam: float, loc: PolytopeContactLocation, project: bool = False
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    def get_force_comps_from_f_c_B(
        self, f_c_B, loc: PolytopeContactLocation
    ) -> Tuple[float, float]:
        raise NotImplementedError()
