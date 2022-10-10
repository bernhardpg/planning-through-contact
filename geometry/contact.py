from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq, ge, le

from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator


def create_possible_mode_combinations(
    collision_pairs: List["CollisionPair"],
) -> List[Tuple["ContactMode", "ContactMode"]]:
    all_combinations = sum(
        [
            pair_i.create_permutations(pair_j)
            for pair_i, pair_j in combinations(collision_pairs, 2)
        ],
        [],
    )
    return all_combinations


def create_force_balance_constraint(
    collision_pairs: List["CollisionPair"], gravitational_force: npt.NDArray[np.float64]
):
    individual_forces = [
        pair.normal_jacobian.T.dot(pair.normal_force.x)
        + pair.tangential_jacobian.T.dot(pair.friction_forces.x)
        for pair in collision_pairs
    ]
    force_balance_constraint = eq(sum(individual_forces) + gravitational_force, 0)
    return force_balance_constraint


@dataclass
class CollisionGeometry:
    name: str
    dim: int
    order: int = 2

    def __post_init__(self) -> None:
        self.pos = BezierVariable(self.dim, self.order, name=f"{self.name}_pos")

    @property
    def vel(self) -> BezierVariable:
        return self.pos.get_derivative()


@dataclass
class CollisionPair:
    body_a: CollisionGeometry
    body_b: CollisionGeometry
    friction_coeff: float
    signed_distance_func: BezierVariable
    normal_vector: npt.NDArray[np.float64]
    friction_cone_rays: npt.NDArray[
        np.float64
    ]  # (d1, d2, ..., dn), size: (dim, n_rays)
    contact_jacobian: npt.NDArray[np.float64]
    order: int = 2

    @property
    def name(self) -> str:
        return f"{self.body_a.name}_w_{self.body_b.name}"

    # TODO remove
    @property
    def rel_contact_vel(self) -> BezierVariable:
        return self.body_b.vel - self.body_a.vel

    @property
    def pos(self) -> npt.NDArray[sym.Variable]:
        return np.vstack((self.body_a.pos.x, self.body_b.pos.x))

    @property
    def vel(self) -> npt.NDArray[sym.Expression]:
        return np.vstack((self.body_a.vel.x, self.body_b.vel.x))

    @property
    def tangential_vel(self) -> BezierVariable:
        return self.tangential_jacobian.dot(self.vel)

    @property
    def tangential_jacobian(self):
        return self.friction_cone_rays.T.dot(self.contact_jacobian)

    @property
    def normal_jacobian(self):
        return self.normal_vector.T.dot(self.contact_jacobian)

    @property
    def n_friction_cone_rays(self) -> None:
        return self.friction_cone_rays.shape[1]

    def __post_init__(self):
        assert self.body_a.dim == self.body_b.dim
        self.dim = self.body_a.dim

        # Only force strength, not vector
        self.normal_force = BezierVariable(
            dim=1, order=self.order, name=f"{self.name}_normal_force"
        )
        # Only force strengths, not vector
        self.friction_forces = BezierVariable(
            dim=self.n_friction_cone_rays,
            order=self.order,
            name=f"{self.name}_friction_force",
        )

        self.contact_modes = []
        no_contact = ContactMode(
            self.name,
            "no_contact",
            self.pos,
            self.signed_distance_func,
            self.tangential_vel,
            self.normal_force,
            self.friction_forces,
            self.friction_coeff,
        )
        self.contact_modes.append(no_contact)
        rolling = ContactMode(
            self.name,
            "rolling",
            self.pos,
            self.signed_distance_func,
            self.tangential_vel,
            self.normal_force,
            self.friction_forces,
            self.friction_coeff,
        )
        self.contact_modes.append(rolling)
        for idx in range(self.n_friction_cone_rays):
            sliding_along_i = ContactMode(
                self.name,
                "sliding",
                self.pos,
                self.signed_distance_func,
                self.tangential_vel,
                self.normal_force,
                self.friction_forces,
                self.friction_coeff,
                fc_direction_idx=idx,
            )
            self.contact_modes.append(sliding_along_i)

        # TODO Add extra constraints to constrain table position

        # TODO one big force balance constraint must be added for all unactuated forces in the entire problem
        # constraints.append(self.create_force_balance_constraints())

    def add_constraint_to_modes(self, c: npt.NDArray[sym.Formula]) -> None:
        for mode in self.contact_modes:
            mode.constraints.append(c)

    def create_mode_polyhedrons(self, vars: npt.NDArray[sym.Formula]) -> None:
        for mode in self.contact_modes:
            mode.create_polyhedron(vars)

    # TODO this can be cleaned up
    def create_mode_pos_polyhedrons(self, pos_vars: npt.NDArray[sym.Formula]) -> None:
        for mode in self.contact_modes:
            mode.create_pos_polyhedron(pos_vars)

    @property
    def slack_vars(self) -> npt.NDArray[sym.Variable]:
        slack_vars = np.vstack(
            [
                mode.slack_var.x
                for mode in self.contact_modes
                if mode.slack_var is not None
            ]
        )
        return slack_vars

    @property
    def force_vars(self) -> npt.NDArray[sym.Variable]:
        return np.vstack((self.normal_force.x, self.friction_forces.x))

    def create_permutations(
        self, other: "CollisionPair"
    ) -> List[Tuple["ContactMode", "ContactMode"]]:
        def position_constraints_overlap(perm: Tuple["ContactMode", "ContactMode"]):
            m1, m2 = perm
            overlapping = m1.pos_polyhedron.IntersectsWith(m2.pos_polyhedron)
            if not overlapping:
                print(f"No overlap between {m1.name} and {m2.name}")
            return overlapping

        all_overlapping_modes = list(
            filter(
                position_constraints_overlap,
                product(self.contact_modes, other.contact_modes),
            )
        )
        return all_overlapping_modes


@dataclass
class ContactMode:
    pair_name: str
    mode: Literal["no_contact", "rolling", "sliding"]
    pos: npt.NDArray[sym.Variable]
    signed_distance_func: npt.NDArray[npt.NDArray[sym.Formula]]
    tangential_vel: BezierVariable
    normal_force: BezierVariable
    friction_forces: BezierVariable
    friction_coeff: float
    fc_direction_idx: Optional[int] = None
    EPS: float = 0

    def __post_init__(self):
        self.name = (
            f"{self.pair_name}_{self.mode}"
            if self.fc_direction_idx is None
            else f"{self.pair_name}_{self.mode}_in_dir_{self.fc_direction_idx}"
        )
        self.num_directions = self.tangential_vel.shape[0]  # TODO unclean?

        self.constraints = []
        if self.mode == "no_contact":
            self.slack_var = None
            self.constraints.append(self.create_sdf_constraint("no_contact"))
            self.constraints.append(self.create_normal_force_constraints("zero_force"))
        elif self.mode == "rolling" or self.mode == "sliding":
            self.slack_var = BezierVariable(
                dim=1,
                order=self.tangential_vel.shape[1] - 1,  # TODO hardcoded
                name=f"{self.name}_gamma",
            )

            self.constraints.append(self.create_sdf_constraint("in_contact"))
            self.constraints.append(
                self.create_normal_force_constraints("nonzero_force")
            )
            if self.mode == "rolling":
                self.constraints.append(self.create_fc_constraint("inside"))
                self.constraints.append(self.create_sliding_vel_constraint("zero"))
                self.constraints.append(self.create_tang_vel_constraint("active"))
                self.constraints.append(
                    self.create_friction_force_constraint("positive")
                )

            elif self.mode == "sliding":
                assert self.fc_direction_idx is not None
                self.constraints.append(self.create_fc_constraint("outside"))
                self.constraints.append(self.create_sliding_vel_constraint("positive"))

                # TODO do I need to do this for every permutation of direction?
                for idx in range(self.num_directions):
                    if idx == self.fc_direction_idx:
                        self.constraints.append(
                            self.create_tang_vel_constraint("active", idx=idx)
                        )
                        self.constraints.append(
                            self.create_friction_force_constraint("positive", idx=idx)
                        )
                    else:
                        self.constraints.append(
                            self.create_tang_vel_constraint("inactive", idx=idx)
                        )
                        self.constraints.append(
                            self.create_friction_force_constraint("zero", idx=idx)
                        )

        else:
            raise NotImplementedError

    #
    #        # TODO remove this?
    #        relevant_pos_variables = self.all_vars_flattened[
    #            [0, 1, 2, 3, 4, 5, -2, -1]
    #        ]  # TODO hardcoded
    #        self.convex_set_position = formulator.formulate_polyhedron(
    #            self.all_vars_flattened[:6], remove_constraints_not_in_vars=True
    #        )

    def create_normal_force_constraints(
        self, type: Literal["zero_force", "nonzero_force"]
    ) -> npt.NDArray[sym.Formula]:
        if type == "zero_force":
            return self.normal_force == 0
        elif type == "nonzero_force":
            return self.normal_force >= self.EPS
        else:
            raise ValueError

    def create_sdf_constraint(
        self, type: Literal["no_contact", "in_contact"]
    ) -> npt.NDArray[sym.Formula]:
        if type == "no_contact":
            return ge(self.signed_distance_func, 0)  # NOTE need continuity in position!
        elif type == "in_contact":
            return eq(self.signed_distance_func, 0)
        else:
            raise ValueError

    def create_fc_constraint(
        self, type: Literal["inside", "outside"]
    ) -> npt.NDArray[sym.Formula]:
        if type == "inside":
            return le(
                np.ones((self.friction_forces.dim, 1)).T.dot(self.friction_forces.x)
                - self.friction_coeff * self.normal_force.x,
                self.EPS,
            )
        elif type == "outside":
            return eq(
                np.ones((self.friction_forces.dim, 1)).T.dot(self.friction_forces.x)
                - self.friction_coeff * self.normal_force.x,
                0,
            )
        else:
            raise ValueError

    # TODO refactor these into one function?
    def create_sliding_vel_constraint(
        self, type: Literal["zero", "positive"]
    ) -> npt.NDArray[sym.Formula]:
        if type == "zero":
            return eq(self.slack_var.x, 0)
        elif type == "positive":
            return ge(self.slack_var.x, self.EPS)
        else:
            raise ValueError

    def create_tang_vel_constraint(
        self, type: Literal["active", "inactive"], idx: Optional[int] = None
    ) -> npt.NDArray[sym.Formula]:
        if type == "active":
            if idx is not None:
                return eq(self.tangential_vel[idx, :] + self.slack_var.x, 0)
            else:
                return eq(self.tangential_vel + self.slack_var.x, 0)
        elif type == "inactive":
            if idx is not None:
                return ge(self.tangential_vel[idx, :] + self.slack_var.x, self.EPS)
            else:
                return ge(self.tangential_vel + self.slack_var.x, self.EPS)
        else:
            raise ValueError

    def create_friction_force_constraint(
        self, type: Literal["zero", "positive"], idx: Optional[int] = None
    ) -> npt.NDArray[sym.Formula]:
        if type == "zero":
            if idx is not None:
                return eq(self.friction_forces.x[idx, :], 0)
            else:
                return eq(self.friction_forces.x, 0)
        elif type == "positive":
            if idx is not None:
                return ge(self.friction_forces.x[idx, :], self.EPS)
            else:
                return ge(self.friction_forces.x, self.EPS)
        else:
            raise ValueError

    def add_constraint(self, c: sym.Expression) -> None:
        self.constraints.append(c)

    def create_polyhedron(self, vars: npt.NDArray[sym.Variable]) -> None:
        formulator = PolyhedronFormulator(self.constraints)
        self.polyhedron = formulator.formulate_polyhedron(vars, make_bounded=True)

    # TODO: This can be cleaned up
    def create_pos_polyhedron(self, pos_vars: npt.NDArray[sym.Variable]) -> None:
        formulator = PolyhedronFormulator(self.constraints)
        self.pos_polyhedron = formulator.formulate_polyhedron(
            pos_vars, make_bounded=True, remove_constraints_not_in_vars=True
        )
        self.pos_constraints = formulator.get_constraints_in_vars(pos_vars)

    # TODO is this unused?
    def get_transition(
        self, other: "ContactMode"
    ) -> Literal[
        "breaking_contact", "making_contact", "none"
    ]:  # TODO replace these with enums?
        if self.mode == "no_contact" and (
            other.mode == "rolling" or other.mode == "sliding"
        ):
            return "making_contact"
        elif (
            self.mode == "rolling" or self.mode == "sliding"
        ) and other.mode == "no_contact":
            return "breaking_contact"
        else:
            return "none"
