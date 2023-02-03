from dataclasses import dataclass
from typing import NamedTuple, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgramResult

from geometry.two_d.contact.contact_point_2d import ContactPoint2d
from geometry.two_d.contact.types import ContactLocation, ContactMode
from geometry.two_d.rigid_body_2d import Point2d, RigidBody2d
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray
from tools.utils import evaluate_np_formulas_array


class EvaluatedContactFrameConstraints(NamedTuple):
    in_frame_A: npt.NDArray[np.float64]
    in_frame_B: npt.NDArray[np.float64]


class ContactFrameConstraints(NamedTuple):
    in_frame_A: Union[NpFormulaArray, NpExpressionArray]
    in_frame_B: Union[NpFormulaArray, NpExpressionArray]

    def evaluate(
        self, result: MathematicalProgramResult
    ) -> EvaluatedContactFrameConstraints:
        evaluated_in_frame_A = evaluate_np_formulas_array(self.in_frame_A, result)
        evaluated_in_frame_B = evaluate_np_formulas_array(self.in_frame_B, result)
        return EvaluatedContactFrameConstraints(
            evaluated_in_frame_A, evaluated_in_frame_B
        )


class ContactPairConstraints(NamedTuple):
    friction_cone: NpFormulaArray
    relaxed_so_2: sym.Formula
    # so_2_cut: sym.Formula # FIX:
    equal_contact_points: ContactFrameConstraints
    equal_and_opposite_forces: ContactFrameConstraints
    equal_relative_positions: ContactFrameConstraints


@dataclass
class ContactPair2d:
    def __init__(
        self,
        pair_name: str,
        body_A: RigidBody2d,
        contact_location_A: ContactLocation,
        body_B: RigidBody2d,
        contact_location_B: ContactLocation,
        friction_coeff: float,
    ) -> None:
        self.body_A = body_A
        self.contact_point_A = ContactPoint2d(
            body_A,
            body_B,
            contact_location_A,
            contact_location_B,
            friction_coeff,
            name=f"{pair_name}_{body_A.name}",
        )
        self.body_B = body_B
        self.contact_point_B = ContactPoint2d(
            body_B,
            body_A,
            contact_location_B,
            contact_location_A,
            friction_coeff,
            name=f"{pair_name}_{body_B.name}",
        )

        # There is no relative rotation to a point contact
        if not (isinstance(body_A, Point2d) or isinstance(body_B, Point2d)):
            self.cos_th = sym.Variable(f"{pair_name}_cos_th")
            self.sin_th = sym.Variable(f"{pair_name}_sin_th")
            self.R_AB = np.array(
                [[self.cos_th, -self.sin_th], [self.sin_th, self.cos_th]]
            )

        p_AB_A_x = sym.Variable(f"{pair_name}_p_AB_A_x")
        p_AB_A_y = sym.Variable(f"{pair_name}_p_AB_A_y")
        self.p_AB_A = np.array([p_AB_A_x, p_AB_A_y]).reshape((-1, 1))

        p_BA_B_x = sym.Variable(f"{pair_name}_p_BA_B_x")
        p_BA_B_y = sym.Variable(f"{pair_name}_p_BA_B_y")
        self.p_BA_B = np.array([p_BA_B_x, p_BA_B_y]).reshape((-1, 1))

    @property
    def contact_points(self) -> Tuple[ContactPoint2d, ContactPoint2d]:
        return (self.contact_point_A, self.contact_point_B)

    @property
    def orientation_variables(self) -> NpVariableArray:
        return np.array([self.cos_th, self.sin_th])

    @property
    def variables(self) -> NpVariableArray:
        return np.concatenate(
            [
                self.orientation_variables,
                self.contact_point_A.variables,
                self.contact_point_B.variables,
                self.p_AB_A.flatten(),
                self.p_BA_B.flatten(),
            ]
        )

    def create_equal_contact_point_constraints(self) -> ContactFrameConstraints:
        p_Ac_A = self.contact_point_A.contact_position
        p_Bc_B = self.contact_point_B.contact_position

        p_Bc_A = self.R_AB.dot(p_Bc_B)
        eq_contact_point_in_A = eq(p_Ac_A, self.p_AB_A + p_Bc_A)

        p_Ac_B = self.R_AB.T.dot(p_Ac_A)
        eq_contact_point_in_B = eq(p_Bc_B, self.p_BA_B + p_Ac_B)

        return ContactFrameConstraints(eq_contact_point_in_A, eq_contact_point_in_B)

    def create_equal_rel_position_constraints(self) -> ContactFrameConstraints:
        rel_pos_equal_in_A = eq(self.p_AB_A, -self.R_AB.dot(self.p_BA_B))
        rel_pos_equal_in_B = eq(self.p_BA_B, -self.R_AB.T.dot(self.p_AB_A))

        return ContactFrameConstraints(rel_pos_equal_in_A, rel_pos_equal_in_B)

    def create_equal_and_opposite_forces_constraint(self) -> ContactFrameConstraints:
        f_c_A = self.contact_point_A.contact_force
        f_c_B = self.contact_point_B.contact_force

        equal_and_opposite_in_A = eq(f_c_A, -self.R_AB.dot(f_c_B))
        equal_and_opposite_in_B = eq(f_c_B, -self.R_AB.T.dot(f_c_A))

        return ContactFrameConstraints(equal_and_opposite_in_A, equal_and_opposite_in_B)

    def create_relaxed_so2_constraint(self) -> sym.Formula:
        relaxed_so_2_constraint = (self.R_AB.T.dot(self.R_AB))[
            0, 0
        ] <= 1  # cos_th^2 + sin_th^2 <= 1
        return relaxed_so_2_constraint

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        return np.concatenate(
            [
                self.contact_point_A.create_friction_cone_constraints(),
                self.contact_point_B.create_friction_cone_constraints(),
            ]
        )

    def create_constraints(self, contact_mode: ContactMode) -> ContactPairConstraints:
        if contact_mode == ContactMode.ROLLING:
            return ContactPairConstraints(
                self.create_friction_cone_constraints(),
                self.create_relaxed_so2_constraint(),
                self.create_equal_contact_point_constraints(),
                self.create_equal_and_opposite_forces_constraint(),
                self.create_equal_rel_position_constraints(),
            )
        else:
            raise NotImplementedError("Contact mode {contact_mode} not implemented.")
