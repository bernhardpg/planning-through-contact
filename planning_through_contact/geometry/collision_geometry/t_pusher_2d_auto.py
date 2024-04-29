"""
Automatically generated version of the TPusher2d class.
"""

from dataclasses import dataclass, field
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
from planning_through_contact.geometry.utilities import normalize_vec

from .helpers import (
    compute_collision_free_regions,
    compute_outer_edges,
    compute_outer_vertices,
    compute_union_dimensions,
    extract_ordered_vertices,
    order_edges_by_connectivity,
)


@dataclass(frozen=True)
class TPusher2dAuto(CollisionGeometry):
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
    def primitive_boxes(self) -> dict:
        tee_boxes = [
            {
                "name": "box1",
                "size": [0.2, 0.05, 0.05],
                "transform": np.eye(4),
            },
            {
                "name": "box2",
                "size": [0.05, 0.15001, 0.05],  # Require a small overlap
                "transform": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -0.1],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
        ]
        return tee_boxes

    @cached_property
    def ordered_edges(
        self,
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        vertices = compute_outer_vertices(self.primitive_boxes)
        edges = compute_outer_edges(vertices, self.primitive_boxes)
        ordered_edges = order_edges_by_connectivity(edges, self.primitive_boxes)
        return ordered_edges

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
        ordered_vertices = extract_ordered_vertices(self.ordered_edges)
        vertices_np = [np.array(v).reshape((2,1)) for v in ordered_vertices]

        # Calculated COM for Tee
        vs_offset = [v - self.com_offset for v in vertices_np]
        return vs_offset

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @cached_property
    def faces(self) -> List[Hyperplane]:
        """Numbering might be different. The below is one example.
        ______f0________
        f7              f1
        |_f6________f2__|
            |    |
            f5   f3
            |    |
            |_f4_|

        """
        edges_np = [np.array(edge) for edge in self.ordered_edges]
        hyperplanes = [
            construct_2d_plane_from_points(*edge) for edge in edges_np
        ]
        return hyperplanes

    @property
    def width(self) -> float:
        width, _ = compute_union_dimensions(self.primitive_boxes)
        return width

    @property
    def height(self) -> float:
        _, height = compute_union_dimensions(self.primitive_boxes)
        return height

    @property
    def contact_locations(self) -> List[PolytopeContactLocation]:
        # TODO(bernhardpg): Only returns FACEs, should ideally return
        # both vertices and faces
        locs = [
            PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)
            for idx in range(len(self.faces))
        ]
        return locs

    @cached_property
    def num_collision_free_regions(self) -> int:
        planes_per_region, _ = compute_collision_free_regions(
            self.primitive_boxes, self.ordered_edges
        )
        return len(planes_per_region)

    @cached_property
    def planes_and_faces_per_region(
        self,
    ) -> Tuple[
        List[List[Tuple[np.ndarray, np.ndarray]]],
        List[List[Tuple[np.ndarray, np.ndarray]]],
    ]:
        planes_per_region, faces_per_region = compute_collision_free_regions(
            self.primitive_boxes, self.ordered_edges
        )
        planes_per_region_np = [
            [np.array(p) for p in planes] for planes in planes_per_region
        ]
        faces_per_region_np = [
            [np.array(f) for f in faces] for faces in faces_per_region
        ]
        return planes_per_region_np, faces_per_region_np

    @cached_property
    def face_to_region_mapping(self) -> List[int]:
        _, faces_per_region = self.planes_and_faces_per_region
        face_to_region_mapping = []
        for region_idx, faces in enumerate(faces_per_region):
            for _ in faces:
                face_to_region_mapping.append(region_idx)
        return face_to_region_mapping

    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
        face_to_region_mapping = self.face_to_region_mapping
        return face_to_region_mapping[loc_idx]

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        """
        Gets the contact faces for each collision-free set.
        This function is hand designed for the object geometry.
        """
        _, faces_per_region = self.planes_and_faces_per_region
        faces = faces_per_region[idx]
        return [construct_2d_plane_from_points(*face) for face in faces]

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
        planes_per_region, _ = self.planes_and_faces_per_region
        planes = planes_per_region[idx]
        return [construct_2d_plane_from_points(*plane) for plane in planes]

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
        # TODO
        box_1 = self.box_1
        box_1_center = np.array([0, 0, z_value])
        box_1_center[:2] -= self.com_offset.flatten()
        transform_1 = RigidTransform(RotationMatrix.Identity(), box_1_center)  # type: ignore
        box_2 = self.box_2
        box_2_center = np.array([0, -self.box_1.height / 2 - self.box_2.height / 2, 0])
        box_2_center[:2] -= self.com_offset.flatten()
        transform_2 = RigidTransform(RotationMatrix.Identity(), box_2_center)

        return [box_1, box_2], [transform_1, transform_2]
