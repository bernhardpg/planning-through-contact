import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from typing import List, Literal, Union, Optional

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import (
    MathematicalProgram,
    Solve,
    MathematicalProgramResult,
    L1NormCost,
    Cost,
    Binding,
    Constraint,
)

from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator


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
        return f"({self.body_a.name}, {self.body_b.name})"

    # TODO remove
    @property
    def rel_contact_vel(self) -> BezierVariable:
        return self.body_b.vel - self.body_a.vel

    @property
    def vel(self) -> BezierVariable:
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
        self.normal_force = BezierVariable(dim=1, order=self.order, name="normal_force")
        # Only force strengths, not vector
        self.friction_forces = BezierVariable(
            dim=self.n_friction_cone_rays, order=self.order, name="friction_force"
        )

        self.contact_modes = []
        no_contact = ContactMode(
            "no_contact",
            self.signed_distance_func,
            self.tangential_vel,
            self.normal_force,
            self.friction_forces,
            self.friction_coeff,
        )
        self.contact_modes.append(no_contact)
        rolling = ContactMode(
            "rolling",
            self.signed_distance_func,
            self.tangential_vel,
            self.normal_force,
            self.friction_forces,
            self.friction_coeff,
        )
        self.contact_modes.append(rolling)
        for idx in range(self.n_friction_cone_rays):
            sliding_along_i = ContactMode(
                "sliding",
                self.signed_distance_func,
                self.tangential_vel,
                self.normal_force,
                self.friction_forces,
                self.friction_coeff,
                fc_direction_idx=idx,
            )
            self.contact_modes.append(sliding_along_i)

        breakpoint()
        # TODO Add extra constraints to constrain table position

        # TODO one big force balance constraint must be added for all unactuated forces in the entire problem
        # constraints.append(self.create_force_balance_constraints())
        # TODO sliding
        # self.sliding = ...


@dataclass
class ContactMode:
    mode: Literal["no_contact", "rolling", "sliding"]
    signed_distance_func: npt.NDArray[npt.NDArray[sym.Formula]]
    tangential_vel: BezierVariable
    normal_force: BezierVariable
    friction_forces: BezierVariable
    friction_coeff: float
    fc_direction_idx: Optional[int] = None
    EPS: float = 1e-5

    def __post_init__(self):
        self.name = (
            f"{self.mode}"
            if self.fc_direction_idx is None
            else f"{self.mode}_in_dir_{self.fc_direction_idx}"
        )
        self.num_directions = self.tangential_vel.shape[0]  # TODO unclean?

        self.constraints = []
        if self.mode == "no_contact":
            self.constraints.append(self.create_sdf_constraint("no_contact"))
            self.constraints.append(self.create_normal_force_constraints("zero_force"))
        elif self.mode == "rolling" or self.mode == "sliding":
            self.slack_var = BezierVariable(
                dim=1,
                order=self.tangential_vel.shape[1] - 1,  # TODO hardcoded
                name="gamma",
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
                assert not self.fc_direction_idx == None
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

    #        formulator = PolyhedronFormulator(constraints)
    #        self.convex_set = formulator.formulate_polyhedron(self.all_vars_flattened)
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
            return ge(self.signed_distance_func, self.EPS)
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

    # TODO clean up
    @property
    def x(self) -> npt.NDArray[sym.Variable]:
        return self.all_vars_flattened

    # TODO clean up
    @property
    def A(self) -> npt.NDArray[np.float64]:
        return self.convex_set.A()

    # TODO clean up
    @property
    def b(self) -> npt.NDArray[np.float64]:
        return self.convex_set.b()

    # TODO clean up
    @property
    def all_vars_flattened(self) -> npt.NDArray[sym.Variable]:
        all_vars = np.concatenate(
            (
                self.position_vars,
                self.normal_force_vars,
                self.friction_force_vars,
                [self.slack_var],  # TODO bad code
            )
        )
        return np.concatenate([var.x.flatten() for var in all_vars])

    # TODO clean up
    @property
    def velocity_vars(self) -> npt.NDArray[sym.Variable]:
        return np.array([pos.get_derivative() for pos in self.position_vars])

    # TODO clean up
    @property
    def rel_sliding_vel(self) -> npt.NDArray[sym.Expression]:
        # Must get the zero-th element, as the result will be contained in a numpy array, and we just want the BezierVariable object
        return self.tangential_jacobian.dot(self.velocity_vars)[0]

    # TODO clean up
    @property
    def normal_vel(self) -> npt.NDArray[sym.Expression]:
        # Must get the zero-th element, as the result will be contained in a numpy array, and we just want the BezierVariable object
        return self.normal_jacobian.dot(self.velocity_vars)[0]

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
