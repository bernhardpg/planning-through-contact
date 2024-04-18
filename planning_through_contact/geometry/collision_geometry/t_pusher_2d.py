from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, List, Tuple

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
from planning_through_contact.geometry.utilities import cross_2d, normalize_vec


@dataclass(frozen=True)
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

    Origin is placed at com_offset from the origin of box_1.

    """

    box_1: Box2d = field(default_factory=lambda: Box2d(0.2, 0.05))
    box_2: Box2d = field(default_factory=lambda: Box2d(0.05, 0.15))

    @property
    def collision_geometry_names(self) -> List[str]:
        return [
            "t_pusher::t_pusher_bottom_collision",
            "t_pusher::t_pusher_top_collision",
        ]

    @classmethod
    def from_drake(cls, drake_shape: DrakeShape):
        raise NotImplementedError()

    @property
    def com_offset(self) -> npt.NDArray[np.float64]:
        y_offset = -0.04285714
        return np.array([0, y_offset]).reshape((-1, 1))

    @cached_property
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
        vs = [v0, v1, v2, v3, v4, v5, v6, v7]

        # Calculated COM for Tee
        vs_offset = [v - self.com_offset for v in vs]
        return vs_offset

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @cached_property
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

    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
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
    def width(self) -> float:
        return self.box_1.width

    @property
    def height(self) -> float:
        return self.box_1.height + self.box_2.height

    @property
    def contact_locations(self) -> List[PolytopeContactLocation]:
        # TODO(bernhardpg): Only returns FACEs, should ideally return
        # both vertices and faces
        locs = [
            PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)
            for idx in range(len(self.faces))
        ]
        return locs

    def get_contact_plane_idxs(self, idx: int) -> List[int]:
        """
        Gets the contact face idxs for each collision-free set.
        This function is hand designed for the object geometry.
        """
        if idx == 0:
            return [0]
        elif idx == 1:
            return [1]
        elif idx == 2:
            return [2, 3]
        elif idx == 3:
            return [4]
        elif idx == 4:
            return [5, 6]
        elif idx == 5:
            return [7]
        else:
            raise ValueError(f"No collision-free region for idx {idx}")

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        """
        Gets the contact faces for each collision-free set.
        This function is hand designed for the object geometry.
        """
        face_idxs = self.get_contact_plane_idxs(idx)
        return [self.faces[face_idx] for face_idx in face_idxs]

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
            raise NotImplementedError(f"Face {idx} not supported")

    # TODO: Much of the following code is copied straight from equilateralpolytope and should be unified!

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

    @cached_property
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

    @cached_property
    def tangent_vecs(self) -> List[npt.NDArray[np.float64]]:
        tangents = [
            self._get_tangent_vec(self.normal_vecs[idx])
            for idx in range(self.num_vertices)
        ]
        return tangents

    @cached_property
    def corner_normal_vecs(self) -> List[npt.NDArray[np.float64]]:
        wrap_around = lambda idx: idx % self.num_vertices
        indices = [(wrap_around(idx - 1), idx) for idx in range(self.num_vertices)]
        corner_normals = [
            normalize_vec(self.normal_vecs[i] + self.normal_vecs[j]) for i, j in indices
        ]
        return corner_normals

    @cached_property
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

    def get_as_boxes(
        self, z_value: float = 0.0
    ) -> Tuple[List[Box2d], List[RigidTransform]]:
        box_1 = self.box_1
        box_1_center = np.array([0, 0, z_value])
        box_1_center[:2] -= self.com_offset.flatten()
        transform_1 = RigidTransform(RotationMatrix.Identity(), box_1_center)  # type: ignore
        box_2 = self.box_2
        box_2_center = np.array([0, -self.box_1.height / 2 - self.box_2.height / 2, 0])
        box_2_center[:2] -= self.com_offset.flatten()
        transform_2 = RigidTransform(RotationMatrix.Identity(), box_2_center)

        return [box_1, box_2], [transform_1, transform_2]

    # TODO(bernhardpg): Remove, uses the old definitions for the collision free sets
    def get_faces_for_collision_free_set(
        self, location: PolytopeContactLocation
    ) -> List[Hyperplane]:
        """
        Gets the faces that defines the collision free sets outside of each face.
        This function is hand designed for the object geometry!
        """
        if not location.pos == ContactLocation.FACE:
            raise NotImplementedError(
                "Can only find faces for collisionfree regions for faces."
            )
        else:
            face_idx = location.idx
            if face_idx == 0:
                return [self.faces[0], self.faces[1]]  # normals pointing outwards
            elif face_idx == 1:
                return [self.faces[0], self.faces[1]]  # normals pointing outwards
            elif face_idx == 2:
                return [self.faces[2]]
            elif face_idx == 3:
                return [self.faces[3]]
            elif face_idx == 4:
                return [self.faces[4]]
            elif face_idx == 5:
                return [self.faces[5], self.faces[6]]
            elif face_idx == 6:
                return [self.faces[5], self.faces[6]]
            elif face_idx == 7:
                return [self.faces[7]]
            else:
                raise NotImplementedError("Currently only face 0 is supported")

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

        for face_idx in range(len(self.faces)):
            region_idx = self.get_collision_free_region_for_loc_idx(face_idx)
            planes = self.get_planes_for_collision_free_region(region_idx)
            planes.extend(self.get_contact_planes(region_idx))
            if np.all([plane.dist_to(pos) >= 0 for plane in planes]):
                # Concave corner requires more careful handling, as we don't know which face is
                # the closest
                if region_idx in [2, 4]:
                    face_idxs_for_region = self.get_contact_plane_idxs(region_idx)
                    faces_for_region = [self.faces[idx] for idx in face_idxs_for_region]
                    closest_face_idx = face_idxs_for_region[
                        np.argmin([f.dist_to(pos) for f in faces_for_region])
                    ]
                    return _make_jacobian(
                        self.normal_vecs[closest_face_idx],
                        self.tangent_vecs[closest_face_idx],
                        pos,
                    )
                else:
                    return _make_jacobian(
                        self.normal_vecs[face_idx], self.tangent_vecs[face_idx], pos
                    )

        # inside box we just return zero
        return np.zeros((2, 3))

        # raise RuntimeError(f"Position {pos} is not inside any region for the Tee.")

    def get_signed_distance(self, pos: npt.NDArray) -> float:
        """
        Returns the signed distance from the pos to the closest point on the box.

        @param pos: position relative to the COM of the box.
        """

        if len(pos.shape) == 1:
            pos = pos.reshape((-1, 1))

        for face_idx in range(len(self.faces)):
            region_idx = self.get_collision_free_region_for_loc_idx(face_idx)
            planes = self.get_planes_for_collision_free_region(region_idx)
            planes.extend(self.get_contact_planes(region_idx))
            if np.all([plane.dist_to(pos) >= 0 for plane in planes]):
                # Concave corner requires more careful handling, as we don't know which face is
                # the closest
                if region_idx in [2, 4]:
                    face_idxs_for_region = self.get_contact_plane_idxs(region_idx)
                    faces_for_region = [self.faces[idx] for idx in face_idxs_for_region]
                    return np.min([f.dist_to(pos) for f in faces_for_region])
                else:
                    return self.faces[face_idx].dist_to(pos)

        # we must be inside the box
        dists = [f.dist_to(pos) for f in self.faces]

        if dists[2] >= 0:  # we are inside box_2
            box_1_dists = [f.dist_to(pos) for f in self.box_1.faces]
            if not np.all(box_1_dists):
                raise RuntimeError(
                    "Finger is inside of box 1, but not all sdfs are negative. This must be a bug!"
                )
            # we return the least penetration
            return np.max(box_1_dists)
        else:  # we are inside box_1
            box_2_dists = [f.dist_to(pos) for f in self.box_2.faces]
            if not np.all(box_2_dists):
                raise RuntimeError(
                    "Finger is inside of box 2, but not all sdfs are negative. This must be a bug!"
                )
            # we return the least penetration
            return np.max(box_2_dists)
