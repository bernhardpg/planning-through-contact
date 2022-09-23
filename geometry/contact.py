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
class ContactMode:
    # TODO: also needs to take in contact pairs!
    position_vars: npt.NDArray[BezierVariable]
    position_constraints: npt.NDArray[npt.NDArray[sym.Formula]]
    normal_force_vars: npt.NDArray[BezierVariable]
    friction_force_vars: npt.NDArray[BezierVariable]
    mode: Literal["no_contact", "rolling_contact", "sliding_contact"]
    friction_coeff: float
    normal_jacobian: npt.NDArray[np.float64]
    tangential_jacobian: npt.NDArray[np.float64]
    name: Optional[str] = None
    EPS: float = 1e-5

    def __post_init__(self):
        # TODO Will need one slack variable per contact point
        self.slack_vars = BezierVariable(
            dim=self.position_vars[0].dim,
            order=self.velocity_vars[0].order,
            name="gamma",
        )

        constraints = [self.position_constraints]
        if self.mode == "no_contact":
            constraints.append(self.create_contact_force_constraints("zero"))
            constraints.append(self.create_force_balance_constraints())
        elif self.mode == "rolling_contact":
            constraints.append(self.create_contact_force_constraints("nonzero"))
            # There are multiple friction cone constraint in a list, so here we merge the lists
            constraints = sum(
                (constraints, self.create_friction_cone_constraints("inside")), []
            )
            constraints.append(self.create_force_balance_constraints())
        elif self.mode == "sliding_contact":
            constraints.append(self.create_contact_force_constraints("nonzero"))
            constraints = sum(
                (constraints, self.create_friction_cone_constraints("outside")), []
            )
            constraints.append(self.create_force_balance_constraints())
        else:
            raise NotImplementedError

        self.convex_set = PolyhedronFormulator(constraints).formulate_polyhedron(
            self.all_vars_flattened
        )

        self.convex_set_position = PolyhedronFormulator(
            [self.position_constraints]
        ).formulate_polyhedron(self.all_vars_flattened)

    @property
    def x(self) -> npt.NDArray[sym.Variable]:
        return self.all_vars_flattened

    @property
    def A(self) -> npt.NDArray[np.float64]:
        return self.convex_set.A()

    @property
    def b(self) -> npt.NDArray[np.float64]:
        return self.convex_set.b()

    @property
    def all_vars_flattened(self) -> npt.NDArray[sym.Variable]:
        all_vars = np.concatenate(
            (
                self.position_vars,
                self.normal_force_vars,
                self.friction_force_vars,
                [self.slack_vars],  # TODO bad code
            )
        )
        return np.concatenate([var.x.flatten() for var in all_vars])

    @property
    def velocity_vars(self) -> npt.NDArray[sym.Variable]:
        return np.array([pos.get_derivative() for pos in self.position_vars])

    @property
    def rel_sliding_vel(self) -> npt.NDArray[sym.Expression]:
        # Must get the zero-th element, as the result will be contained in a numpy array, and we just want the BezierVariable object
        return self.tangential_jacobian.dot(self.velocity_vars)[0]

    @property
    def normal_vel(self) -> npt.NDArray[sym.Expression]:
        # Must get the zero-th element, as the result will be contained in a numpy array, and we just want the BezierVariable object
        return self.normal_jacobian.dot(self.velocity_vars)[0]

    def create_contact_force_constraints(
        self, type: Literal["zero", "nonzero"]
    ) -> npt.NDArray[sym.Formula]:
        # TODO: Must also deal with contact pairs here when we have more than one!
        lam_n = self.normal_force_vars[0]
        if type == "zero":
            return lam_n == 0
        elif type == "nonzero":
            return lam_n >= self.EPS
        else:
            raise ValueError

    def create_force_balance_constraints(self) -> npt.NDArray[sym.Formula]:
        # TODO must generalize to jacobian
        lam_f = self.friction_force_vars[0]
        lam_n = self.normal_force_vars[0]
        constraint = lam_f == lam_n
        return constraint

    def create_friction_cone_constraints(
        self, type: Literal["inside", "outside"]
    ) -> List[npt.NDArray[sym.Formula]]:
        # TODO: Need to figure out how to handle contact pairs!
        LAM_G = 9.81  # TODO: should not be hardcoded
        lam_f = self.friction_force_vars[0]

        nonzero_friction_force_constraint = lam_f >= self.EPS
        rel_vel_constraint = self.slack_vars + self.rel_sliding_vel == 0

        if type == "inside":
            fc_constraint = lam_f <= self.friction_coeff * LAM_G
            slack_constraint = self.slack_vars == 0
        elif type == "outside":
            fc_constraint = lam_f == self.friction_coeff * LAM_G
            slack_constraint = self.slack_vars >= self.EPS

        return [
            fc_constraint,
            slack_constraint,
            nonzero_friction_force_constraint,
            rel_vel_constraint,
        ]

    def get_transition(
        self, other: "ContactMode"
    ) -> Literal[
        "breaking_contact", "making_contact", "none"
    ]:  # TODO replace these with enums?
        if self.mode == "no_contact" and (
            other.mode == "rolling_contact" or other.mode == "sliding_contact"
        ):
            return "making_contact"
        elif (
            self.mode == "rolling_contact" or self.mode == "sliding_contact"
        ) and other.mode == "no_contact":
            return "breaking_contact"
        else:
            return "none"
