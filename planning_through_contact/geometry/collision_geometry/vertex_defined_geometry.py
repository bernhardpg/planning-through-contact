from functools import cached_property
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
from planning_through_contact.geometry.utilities import cross_2d, normalize_vec


class VertexDefinedGeometry(CollisionGeometry):
    def __init__(self, vertices: List[npt.NDArray[np.float64]]) -> None:
        """
        Assumes the following vertex ordering
        v0 -- v1
        |     |
        v3 -- v2

        Faces:
        v0 - f0 - v1
        |          |
        f3         f1
        |          |
        v3 --f2--- v2

        Corner normal vectors:
        nc0 -- nc1
        |       |
        nc3 -- nc2
        """

        if vertices[0].shape == (2,):
            vertices = [v.reshape((2, 1)) for v in vertices]
        elif vertices[0].shape == (2, 1):
            pass
        else:
            raise RuntimeError(f"Vertex shape {vertices[0].shape} is wrong.")

        self._vertices = vertices

    @cached_property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        return self._vertices

    @property
    def com(self) -> npt.NDArray[np.float64]:
        """
        This can easily be extended to account for any Center of Mass (CoM).
        """
        return np.zeros((2, 1))

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    @cached_property
    def faces(self) -> List[Hyperplane]:
        wrap_around = lambda num: num % self.num_vertices
        pairwise_indices = [
            (idx, wrap_around(idx + 1)) for idx in range(self.num_vertices)
        ]
        hyperplane_points = [
            (self._vertices[i], self._vertices[j]) for i, j in pairwise_indices
        ]
        hyperplanes = [
            construct_2d_plane_from_points(p1, p2) for p1, p2 in hyperplane_points
        ]
        return hyperplanes

    @property
    def contact_locations(self) -> List[PolytopeContactLocation]:
        """
        All the faces where contact can be made.

        (NOTE: This does not return contact
        at vertices; this might be added in the future).
        """
        locs = [
            PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)
            for idx in range(len(self.faces))
        ]
        return locs

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        vertices = np.hstack([self._vertices[idx] for idx in range(self.num_vertices)])
        return vertices

    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        if location.pos == ContactLocation.FACE:
            wrap_around = lambda num: num % self.num_vertices
            return [
                self._vertices[location.idx],
                self._vertices[wrap_around(location.idx + 1)],
            ]
        elif location.pos == ContactLocation.VERTEX:
            return [self._vertices[location.idx]]
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

            return self._vertices[idx_prev], self._vertices[idx_next]
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
        raise NotImplementedError(
            f"Face length not yet implemented for {self.__name__}."
        )

    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
        return loc_idx  # for this class, we assume we have one collision-free region per face

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        # for this class, we assume we have one collision-free region per face
        return [self.faces[idx]]

    @property
    def num_collision_free_regions(self) -> int:
        # for this class, we assume we have one collision-free region per face
        return len(self.faces)

    def get_planes_for_collision_free_region(self, idx: int) -> List[Hyperplane]:
        """
        Get the two planes that defines the collision-free region (not including the
        contact plane on the slider).

        The region ordering is the same as the face ordering:
        v0 - f0 - v1
        |          |
        f3         f1
        |          |
        v3 --f2--- v2
        """
        planes = []
        wrap_around = lambda num: num % self.num_vertices
        CONST = 0.2
        planes.append(
            construct_2d_plane_from_points(
                self.vertices[idx],
                self.vertices[idx] + CONST * self.corner_normal_vecs[idx],
            )
        )
        planes.append(
            construct_2d_plane_from_points(
                self.vertices[wrap_around(idx + 1)]
                + CONST * self.corner_normal_vecs[wrap_around(idx + 1)],
                self.vertices[wrap_around(idx + 1)],
            )
        )

        return planes

    @property
    def max_contact_radius(self) -> float:
        test = float(np.max([np.linalg.norm(v - self.com) for v in self.vertices]))
        return test
