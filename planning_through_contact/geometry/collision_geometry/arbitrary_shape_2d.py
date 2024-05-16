"""
Automatically generated version of the TPusher2d class.
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape
from pydrake.math import RigidTransform

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
from planning_through_contact.tools.utils import load_primitive_info

from .helpers import (
    compute_collision_free_regions,
    compute_com_from_uniform_density,
    compute_normalized_normal_vector_points_from_edges,
    compute_outer_edges,
    compute_outer_vertices,
    compute_union_dimensions,
    direct_edges_so_right_points_inside,
    extract_ordered_vertices,
    offset_boxes,
    order_edges_by_connectivity,
)


@dataclass
class ArbitraryShape2D(CollisionGeometry):
    def __init__(
        self, arbitrary_shape_pickle_path: str, com: Optional[np.ndarray] = None
    ):
        """NOTE: com computed using uniform density if None."""
        assert (
            arbitrary_shape_pickle_path is not None
            and arbitrary_shape_pickle_path != ""
        )
        self.arbitrary_shape_pickle_path = arbitrary_shape_pickle_path
        self.com = com

    # TODO: This needs to match the sdf file for simulation to work...
    @property
    def collision_geometry_names(self) -> List[str]:
        return [
            "arbitrary_shape::arbitrary_shape_bottom_collision",
            "arbitrary_shape::arbitrary_shape_top_collision",
        ]

    @classmethod
    def from_drake(cls, drake_shape: DrakeShape):
        raise NotImplementedError()

    @cached_property
    def com_offset(self) -> npt.NDArray[np.float64]:
        boxes = load_primitive_info(self.arbitrary_shape_pickle_path)
        primitive_types = [box["name"] for box in boxes]
        assert np.all(
            [t == "box" for t in primitive_types]
        ), f"Only boxes are supported. Got: {primitive_types}"
        if self.com is not None:
            return np.array([self.com[0], self.com[1]]).reshape((2, 1))
        # Compute the center of mass from uniform density
        logging.warning("COM not provided. Computing from uniform density.")
        x_com, y_com = compute_com_from_uniform_density(boxes)
        return np.array([x_com, y_com]).reshape((2, 1))

    @cached_property
    def primitive_boxes(self) -> dict:
        """
        The primitive boxes whose union represents the shape. The resulting shape is
        assumed to be planar on the xy-plane.
        NOTE: All boxes require a small overlap.
        """
        boxes = load_primitive_info(self.arbitrary_shape_pickle_path)
        primitive_types = [box["name"] for box in boxes]
        assert np.all(
            [t == "box" for t in primitive_types]
        ), f"Only boxes are supported. Got: {primitive_types}"

        # TODO: Take as input
        print(f"COM offset: {self.com_offset.flatten()}")
        x_com, y_com = self.com_offset.flatten()
        boxes = offset_boxes(boxes, [-x_com, -y_com])

        return boxes

    @cached_property
    def ordered_edges(
        self,
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        vertices = compute_outer_vertices(self.primitive_boxes)
        edges = compute_outer_edges(vertices, self.primitive_boxes)
        directed_edges = direct_edges_so_right_points_inside(
            edges, self.primitive_boxes
        )
        ordered_edges = order_edges_by_connectivity(
            directed_edges, self.primitive_boxes
        )

        return ordered_edges

    @cached_property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        ordered_vertices = extract_ordered_vertices(self.ordered_edges)
        vertices_np = [np.array(v).reshape((2, 1)) for v in ordered_vertices]
        return vertices_np

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @cached_property
    def faces(self) -> List[Hyperplane]:
        edges_np = [np.array(edge) for edge in self.ordered_edges]
        hyperplanes = [construct_2d_plane_from_points(*edge) for edge in edges_np]
        return hyperplanes

    @cached_property
    def width(self) -> float:
        width, _ = compute_union_dimensions(self.primitive_boxes)
        return width

    @cached_property
    def height(self) -> float:
        _, height = compute_union_dimensions(self.primitive_boxes)
        return height

    @cached_property
    def contact_locations(self) -> List[PolytopeContactLocation]:
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
        normals = compute_normalized_normal_vector_points_from_edges(
            self.ordered_edges, self.primitive_boxes
        )
        return normals

    @staticmethod
    def _get_tangent_vec(v: npt.NDArray[np.float64]):
        """
        Returns the 2d tangent vector calculated as the cross product with the z-axis
        pointing out of the plane.
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
        boxes_2d = []
        transforms = []
        for box in self.primitive_boxes:
            size = box["size"]
            transform = box["transform"]
            transform[:2, 3] -= self.com_offset.flatten()
            transform[2, 3] = z_value
            boxes_2d.append(Box2d(size[0], size[1]))
            transforms.append(RigidTransform(transform))
        return boxes_2d, transforms
