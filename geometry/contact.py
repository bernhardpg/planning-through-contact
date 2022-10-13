import itertools
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet
from pydrake.math import eq, ge, le

from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator


@dataclass
class RigidBody:
    name: str
    dim: int
    geometry: Literal["point", "box", "y_plane"]
    width: float = 0  # TODO generalize
    height: float = 0
    position_curve_order: int = 2

    def __post_init__(self) -> None:
        self.pos = BezierVariable(
            self.dim, self.position_curve_order, name=f"{self.name}_pos"
        )

    @property
    def vel(self) -> BezierVariable:
        return self.pos.get_derivative()

    @property
    def pos_x(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[0, :]

    @property
    def pos_y(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[1, :]


@dataclass
class ContactMode:
    name: str
    constraints: List[npt.NDArray[sym.Formula]]
    all_vars: npt.NDArray[sym.Variable]

    def __post_init__(self):
        self.polyhedron = PolyhedronFormulator(self.constraints).formulate_polyhedron(
            variables=self.all_vars, make_bounded=True
        )


class PositionMode(Enum):
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8


@dataclass
class CollisionPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    position_mode: PositionMode  # TODO move somewhere else to generalize?
    order: int = 2  # TODO remove?

    @staticmethod
    def _create_signed_distance_func(
        body_a, body_b, position_mode: PositionMode
    ) -> sym.Expression:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")
        elif body_a.geometry == "box" and body_b.geometry == "box":
            raise NotImplementedError("Box and box contact not implemented")

        x_offset = body_a.width if body_a.geometry == "box" else body_b.width
        y_offset = body_a.height if body_a.geometry == "box" else body_b.height

        if position_mode == PositionMode.LEFT:  # body_a is on left side of body_b
            dx = body_b.pos_x - body_a.pos_x - x_offset
            dy = 0
        elif position_mode == PositionMode.RIGHT:
            dx = body_a.pos_x - body_b.pos_x - x_offset
            dy = 0
        elif position_mode == PositionMode.TOP:  # body_a on top of body_b
            dx = 0
            dy = body_a.pos_y - body_b.pos_y - y_offset
        elif position_mode == PositionMode.BOTTOM:
            dx = 0
            dy = body_b.pos_y - body_a.pos_y - y_offset
        else:
            raise NotImplementedError(f"Position mode not implemented: {position_mode}")

        return dx + dy  # NOTE convex relaxation

    @staticmethod
    def _create_normal_vec(
        body_a, body_b, position_mode: PositionMode
    ) -> npt.NDArray[np.float64]:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")
        elif body_a.geometry == "box" and body_b.geometry == "box":
            raise NotImplementedError("Box and box contact not implemented")

        # Normal vector: from body_a to body_b
        if position_mode == PositionMode.LEFT:  # body_a left side of body_b
            n_hat = np.array([[1, 0]]).T
        elif position_mode == PositionMode.RIGHT:
            n_hat = np.array([[-1, 0]]).T
        elif position_mode == PositionMode.TOP:
            n_hat = np.array([[0, -1]]).T
        elif position_mode == PositionMode.BOTTOM:
            n_hat = np.array([[0, 1]]).T
        else:
            raise NotImplementedError(f"Position mode not implemented: {position_mode}")

        return n_hat

    def __post_init__(self):
        self.sdf = self._create_signed_distance_func(
            self.body_a, self.body_b, self.position_mode
        )
        self.n_hat = self._create_normal_vec(
            self.body_a, self.body_b, self.position_mode
        )
        self.lam_n = BezierVariable(
            dim=1, order=self.order, name=f"{self.name}_lam_n"
        ).x
        self.lam_f = BezierVariable(
            dim=1, order=self.order, name=f"{self.name}_lam_f"
        ).x
        assert self.dim == 2  # TODO for now only works for 2D
        self.d_hat = np.array([[-self.n_hat[1, 0]], [self.n_hat[0, 0]]])
        self.additional_constraints = []

    @property
    def name(self) -> str:
        return f"{self.body_a.name}_{self.body_b.name}"

    @property
    def dim(self) -> int:
        return self.body_a.dim

    @property
    def contact_jacobian(self) -> npt.NDArray[np.float64]:
        # v_rel = v_body_b - v_body_a = J (v_body_a, v_body_b)^T
        return np.hstack((-np.eye(self.dim), np.eye(self.dim)))

    @property
    def normal_jacobian(self) -> npt.NDArray[np.float64]:
        return self.n_hat.T.dot(self.contact_jacobian)

    @property
    def tangential_jacobian(self) -> npt.NDArray[np.float64]:
        return self.d_hat.T.dot(self.contact_jacobian)

    @property
    def rel_tangential_sliding_vel(self) -> npt.NDArray[sym.Expression]:
        return self.tangential_jacobian.dot(
            np.vstack((self.body_a.vel.x, self.body_b.vel.x))
        )

    def get_tangential_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.tangential_jacobian)

    def get_normal_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.normal_jacobian)

    def _get_jacobian_for_bodies(
        self, bodies: List[RigidBody], jacobian: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # (1, num_bodies * num_dims)
        jacobian_for_all_bodies = np.zeros((1, len(bodies) * self.dim))

        body_a_idx_in_J = bodies.index(self.body_a.name) * self.dim
        body_b_idx_in_J = bodies.index(self.body_b.name) * self.dim
        body_a_cols_in_J = np.arange(body_a_idx_in_J, body_a_idx_in_J + self.dim)
        body_b_cols_in_J = np.arange(body_b_idx_in_J, body_b_idx_in_J + self.dim)
        body_a_cols_in_local_J = np.arange(0, self.dim)
        body_b_cols_in_local_J = np.arange(self.dim, 2 * self.dim)

        jacobian_for_all_bodies[:, body_a_cols_in_J] = jacobian[
            :, body_a_cols_in_local_J
        ]
        jacobian_for_all_bodies[:, body_b_cols_in_J] = jacobian[
            :, body_b_cols_in_local_J
        ]
        return jacobian_for_all_bodies

    def add_constraint_to_all_modes(self, constraints) -> None:
        self.additional_constraints = sum(
            [self.additional_constraints, constraints], []
        )

    def add_force_balance(self, force_balance):
        self.force_balance = force_balance

    def formulate_contact_modes(self, all_variables, allow_sliding: bool = False):
        assert self.force_balance is not None

        no_contact_constraints = [
            ge(self.sdf, 0),
            eq(self.lam_n, 0),
            le(self.lam_f, self.friction_coeff * self.lam_n),
            ge(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]

        rolling_constraints = [
            eq(self.sdf, 0),
            ge(self.lam_n, 0),
            eq(self.rel_tangential_sliding_vel, 0),
            le(self.lam_f, self.friction_coeff * self.lam_n),
            ge(self.lam_f, -self.friction_coeff * self.lam_n),
            *self.force_balance,
            *self.additional_constraints,
        ]
        modes_constraints = [
            ("no_contact", no_contact_constraints),
            ("rolling", rolling_constraints),
        ]

        if allow_sliding:
            sliding_positive_constraints = [
                eq(self.sdf, 0),
                ge(self.lam_n, 0),
                ge(self.rel_tangential_sliding_vel, 0),
                eq(self.lam_f, -self.friction_coeff * self.lam_n),
                *self.force_balance,
                *self.additional_constraints,
            ]

            sliding_negative_constraints = [
                eq(self.sdf, 0),
                ge(self.lam_n, 0),
                le(self.rel_tangential_sliding_vel, 0),
                eq(self.lam_f, self.friction_coeff * self.lam_n),
                *self.force_balance,
                *self.additional_constraints,
            ]
            modes_constraints.append(("sliding_positive", sliding_positive_constraints))
            modes_constraints.append(("sliding_negative", sliding_negative_constraints))

        self.contact_modes = [
            ContactMode(f"{self.name}_{name}", constraints, all_variables)
            for name, constraints in modes_constraints
        ]


def calc_intersection_of_contact_modes(
    modes: List[ContactMode],
) -> Tuple[bool, Optional[ConvexSet]]:
    pairwise_combinations = itertools.combinations(modes, 2)
    all_modes_intersect = all(
        map(
            lambda pair: pair[0].polyhedron.IntersectsWith(pair[1].polyhedron),
            pairwise_combinations,
        )
    )
    if all_modes_intersect:
        polys = [m.polyhedron for m in modes]
        intersection = reduce(lambda p1, p2: p1.Intersection(p2), polys)
        names = [m.name for m in modes]
        name = "_W_".join(names)
        return (True, (name, intersection))
    else:
        return (False, (None, None))
