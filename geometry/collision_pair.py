import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet
from pydrake.math import eq, ge, le

from geometry.bezier import BezierVariable
from geometry.contact_mode import (
    ContactMode,
    ContactModeConfig,
    ContactModeType,
    PositionModeType,
    calc_intersection_of_contact_modes,
)
from geometry.polyhedron import PolyhedronFormulator
from geometry.rigid_body import RigidBody


@dataclass
class CollisionPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    position_mode: PositionModeType
    force_curve_order: int = 2

    def __post_init__(self):
        self.sdf = self._create_signed_distance_func(
            self.body_a, self.body_b, self.position_mode
        )
        self.n_hat = self._create_normal_vec(
            self.body_a, self.body_b, self.position_mode
        )
        self.d_hat = self._create_tangential_vec(self.n_hat)

        self.lam_n = BezierVariable(
            dim=1, order=self.force_curve_order, name=f"{self.name}_lam_n"
        ).x
        self.lam_f = BezierVariable(
            dim=1, order=self.force_curve_order, name=f"{self.name}_lam_f"
        ).x
        self.additional_constraints = []
        self.contact_modes_formulated = False

    @staticmethod
    def _create_position_mode_constraints(
        body_a, body_b, position_mode: PositionModeType
    ) -> npt.NDArray[sym.Formula]:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")
        elif body_a.geometry == "box" and body_b.geometry == "box":
            if (
                position_mode == PositionModeType.LEFT
                or position_mode == PositionModeType.RIGHT
            ):
                raise NotImplementedError
            elif (
                position_mode == PositionModeType.TOP
                or position_mode == PositionModeType.BOTTOM
            ):
                x_constraint_left = ge(
                    body_a.pos_x + body_a.width, body_b.pos_x - body_b.width
                )
                x_constraint_right = le(
                    body_a.pos_x - body_a.width, body_b.pos_x + body_b.width
                )
                return np.array([x_constraint_left, x_constraint_right])
        else:
            box = body_a if body_a.geometry == "box" else body_b
            point = body_a if body_a.geometry == "point" else body_b

            if (
                position_mode == PositionModeType.LEFT
                or position_mode == PositionModeType.RIGHT
            ):
                y_constraint_top = le(point.pos_y, box.pos_y + box.height)
                y_constraint_bottom = ge(point.pos_y, box.pos_y - box.height)
                return np.array([y_constraint_top, y_constraint_bottom])
            elif (
                position_mode == PositionModeType.TOP
                or position_mode == PositionModeType.BOTTOM
            ):
                x_constraint_left = ge(point.pos_x, box.pos_x - box.width)
                x_constraint_right = le(point.pos_x, box.pos_x + box.width)
                return np.array([x_constraint_left, x_constraint_right])
            else:
                raise NotImplementedError(
                    f"Position mode not implemented: {position_mode}"
                )

    @staticmethod
    def _create_signed_distance_func(
        body_a, body_b, position_mode: PositionModeType
    ) -> sym.Expression:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")
        elif body_a.geometry == "box" and body_b.geometry == "box":
            x_offset = body_a.width + body_b.width
            y_offset = body_a.height + body_b.height
        else:
            box = body_a if body_a.geometry == "box" else body_b
            x_offset = box.width
            y_offset = box.height

        if position_mode == PositionModeType.LEFT:  # body_a is on left side of body_b
            dx = body_b.pos_x - body_a.pos_x - x_offset
            dy = 0
        elif position_mode == PositionModeType.RIGHT:
            dx = body_a.pos_x - body_b.pos_x - x_offset
            dy = 0
        elif position_mode == PositionModeType.TOP:  # body_a on top of body_b
            dx = 0
            dy = body_a.pos_y - body_b.pos_y - y_offset
        elif position_mode == PositionModeType.BOTTOM:
            dx = 0
            dy = body_b.pos_y - body_a.pos_y - y_offset
        else:
            raise NotImplementedError(f"Position mode not implemented: {position_mode}")

        return dx + dy  # NOTE convex relaxation

    @staticmethod
    def _create_normal_vec(
        body_a, body_b, position_mode: PositionModeType
    ) -> npt.NDArray[np.float64]:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")

        # Normal vector: from body_a to body_b
        if position_mode == PositionModeType.LEFT:  # body_a left side of body_b
            n_hat = np.array([[1, 0]]).T
        elif position_mode == PositionModeType.RIGHT:
            n_hat = np.array([[-1, 0]]).T
        elif position_mode == PositionModeType.TOP:
            n_hat = np.array([[0, -1]]).T
        elif position_mode == PositionModeType.BOTTOM:
            n_hat = np.array([[0, 1]]).T
        else:
            raise NotImplementedError(f"Position mode not implemented: {position_mode}")

        return n_hat

    def _create_tangential_vec(self, n_hat: npt.NDArray[np.float64]):
        assert self.dim == 2  # TODO for now only works for 2D
        d_hat = np.array([[-n_hat[1, 0]], [n_hat[0, 0]]])
        return d_hat

    @property
    def name(self) -> str:
        return f"({self.body_a.name},{self.body_b.name})"

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

    @property
    def allowed_contact_modes(self) -> List[ContactModeType]:
        assert self.contact_modes is not None
        return list(self.contact_modes.keys())

    def _get_jacobian_for_bodies(
        self, bodies: List[RigidBody], local_jacobian: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # (1, num_bodies * num_dims)
        jacobian_for_all_bodies = np.zeros((1, len(bodies) * self.dim))

        body_a_idx_in_J = bodies.index(self.body_a) * self.dim
        body_b_idx_in_J = bodies.index(self.body_b) * self.dim
        body_a_cols_in_J = np.arange(body_a_idx_in_J, body_a_idx_in_J + self.dim)
        body_b_cols_in_J = np.arange(body_b_idx_in_J, body_b_idx_in_J + self.dim)
        body_a_cols_in_local_J = np.arange(0, self.dim)
        body_b_cols_in_local_J = np.arange(self.dim, 2 * self.dim)

        jacobian_for_all_bodies[:, body_a_cols_in_J] = local_jacobian[
            :, body_a_cols_in_local_J
        ]
        jacobian_for_all_bodies[:, body_b_cols_in_J] = local_jacobian[
            :, body_b_cols_in_local_J
        ]
        return jacobian_for_all_bodies

    def get_tangential_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.tangential_jacobian)

    def get_normal_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.normal_jacobian)

    def add_constraint_to_all_modes(self, constraints) -> None:
        self.additional_constraints = sum(
            [self.additional_constraints, constraints], []
        )

    def add_force_balance(self, force_balance):
        self.force_balance = force_balance

    def get_contact_mode(self, contact_mode: ContactModeType) -> ContactMode:
        if not self.contact_modes_formulated:
            raise RuntimeError("Contact modes not formulated for {self.name}")
        return self.contact_modes[contact_mode]

    def formulate_contact_modes(
        self,
        all_variables: npt.NDArray[sym.Variable],
        allow_sliding: bool = False,
    ):
        if self.contact_modes_formulated:
            raise ValueError(f"Contact modes already formulated for {self.name}")

        if self.force_balance is None:
            raise ValueError(
                "Force balance must be set before formulating contact modes"
            )

        position_mode_constraints = self._create_position_mode_constraints(
            self.body_a, self.body_b, self.position_mode
        )

        modes_constraints = {
            ContactModeType.NO_CONTACT: [
                ge(self.sdf, 0),
                eq(self.lam_n, 0),
                le(self.lam_f, self.friction_coeff * self.lam_n),
                ge(self.lam_f, -self.friction_coeff * self.lam_n),
                *position_mode_constraints,
                *self.force_balance,
                *self.additional_constraints,
            ],
            ContactModeType.ROLLING: [
                eq(self.sdf, 0),
                ge(self.lam_n, 0),
                eq(self.rel_tangential_sliding_vel, 0),
                le(self.lam_f, self.friction_coeff * self.lam_n),
                ge(self.lam_f, -self.friction_coeff * self.lam_n),
                *position_mode_constraints,
                *self.force_balance,
                *self.additional_constraints,
            ],
        }

        if allow_sliding:
            modes_constraints[ContactModeType.SLIDING_POSITIVE] = [
                eq(self.sdf, 0),
                ge(self.lam_n, 0),
                ge(self.rel_tangential_sliding_vel, 0),
                eq(self.lam_f, -self.friction_coeff * self.lam_n),
                *position_mode_constraints,
                *self.force_balance,
                *self.additional_constraints,
            ]
            modes_constraints[ContactModeType.SLIDING_NEGATIVE] = [
                eq(self.sdf, 0),
                ge(self.lam_n, 0),
                le(self.rel_tangential_sliding_vel, 0),
                eq(self.lam_f, self.friction_coeff * self.lam_n),
                *position_mode_constraints,
                *self.force_balance,
                *self.additional_constraints,
            ]

        self.contact_modes = {
            mode_type: ContactMode(
                f"{self.name}",
                contact_constraints,
                all_variables,
                mode_type,
            )
            for mode_type, contact_constraints in modes_constraints.items()
        }
        # TODO I HATE this, very against functional programming principles. Find an alternative?
        self.contact_modes_formulated = True


class CollisionPairHandler:
    def __init__(
        self,
        rigid_bodies: List[RigidBody],
        collision_pairs: List[CollisionPair],  # TODO Will be removed
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ) -> None:
        self.rigid_bodies = rigid_bodies
        self.collision_pairs = collision_pairs
        self.all_decision_vars = CollisionPairHandler.collect_all_decision_vars(
            self.rigid_bodies, self.collision_pairs
        )
        unactuated_dofs = self._get_unactuated_dofs(
            self.rigid_bodies, self.position_dim
        )
        force_balance_constraints = self.construct_force_balance(
            collision_pairs,
            self.rigid_bodies,
            external_forces,
            unactuated_dofs,
        )
        for p in self.collision_pairs:
            p.add_force_balance(force_balance_constraints)
        for p in self.collision_pairs:
            p.add_constraint_to_all_modes(additional_constraints)

        for p in self.collision_pairs:
            p.formulate_contact_modes(self.all_decision_vars, allow_sliding)

    @staticmethod
    def collect_all_decision_vars(
        bodies: List[RigidBody],
        collision_pairs: List[CollisionPair],
    ) -> npt.NDArray[sym.Variable]:
        all_pos_vars = np.concatenate([b.pos.x.flatten() for b in bodies])
        all_normal_force_vars = np.concatenate(
            [p.lam_n.flatten() for p in collision_pairs]
        )
        all_friction_force_vars = np.concatenate(
            [p.lam_f.flatten() for p in collision_pairs]
        )
        all_vars = np.concatenate(
            [all_pos_vars, all_normal_force_vars, all_friction_force_vars]
        )
        return all_vars

    @property
    def num_bodies(self) -> int:
        return len(self.rigid_bodies)

    @property
    def position_curve_order(self) -> int:
        return self.collision_pairs[0].body_a.position_curve_order

    @property
    def all_position_vars(self) -> npt.NDArray[sym.Variable]:
        return self.all_decision_vars[
            : self.num_bodies * (self.position_curve_order + 1) * self.position_dim
        ]

    @property
    def all_force_vars(self) -> npt.NDArray[sym.Variable]:
        return self.all_decision_vars[len(self.all_position_vars) :]

    @property
    def position_dim(self) -> int:
        return self.rigid_bodies[0].dim

    @property
    def collision_pairs_by_name(self) -> Dict[str, CollisionPair]:
        return {p.name: p for p in self.collision_pairs}

    def _get_unactuated_dofs(
        self, rigid_bodies: List[RigidBody], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [i for i, b in enumerate(rigid_bodies) if not b.actuated]
        unactuated_dofs = np.concatenate(
            [np.arange(idx * dim, (idx + 1) * dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    @staticmethod
    def construct_force_balance(
        collision_pairs: List[CollisionPair],
        bodies: List[RigidBody],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int64],
    ) -> List[sym.Formula]:
        normal_jacobians = np.vstack(
            [p.get_normal_jacobian_for_bodies(bodies) for p in collision_pairs]
        )
        tangential_jacobians = np.vstack(
            [p.get_tangential_jacobian_for_bodies(bodies) for p in collision_pairs]
        )

        normal_forces = np.concatenate([p.lam_n for p in collision_pairs])
        friction_forces = np.concatenate([p.lam_f for p in collision_pairs])

        all_force_balances = eq(
            normal_jacobians.T.dot(normal_forces)
            + tangential_jacobians.T.dot(friction_forces)
            + external_forces,
            0,
        )
        force_balance = all_force_balances[unactuated_dofs, :]
        return force_balance

    def all_possible_contact_cfg_perms(self) -> List[ContactModeConfig]:
        # [(n_m), (n_m), ... (n_m)] n_p times --> n_m * n_p
        all_allowed_contact_modes = [
            [(pair.name, mode) for mode in pair.allowed_contact_modes]
            for pair in self.collision_pairs
        ]
        # Cartesian product:
        # S = P_1 X P_2 X ... X P_n_p
        # |S| = |P_1| * |P_2| * ... * |P_n_p|
        #     = n_m * n_m * ... * n_m
        #     = n_m^n_p
        all_possible_permutations = [
            ContactModeConfig({name: mode for name, mode in perm})
            for perm in itertools.product(*all_allowed_contact_modes)
        ]
        return all_possible_permutations

    def create_convex_set_from_mode_config(
        self,
        config: "ContactModeConfig",
        name: Optional[str] = None,
    ) -> Optional[Tuple[ConvexSet, str]]:
        contact_modes = [
            self.collision_pairs_by_name[pair].get_contact_mode(mode)
            for pair, mode in config.modes.items()
        ]
        intersects, (
            calculated_name,
            intersection,
        ) = calc_intersection_of_contact_modes(contact_modes)
        if not intersects:
            return None

        if config.additional_constraints is not None:
            additional_set = PolyhedronFormulator(
                config.additional_constraints
            ).formulate_polyhedron(self.all_decision_vars)
            intersects = intersects and intersection.IntersectsWith(additional_set)
            if not intersects:
                return None

            intersection = intersection.Intersection(additional_set)

        name = f"{name}: {calculated_name}" if name is not None else calculated_name
        return intersection, name
