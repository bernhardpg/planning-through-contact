from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq
from pydrake.solvers import MathematicalProgramResult

from geometry.hyperplane import Hyperplane, calculate_convex_hull_cut_for_so_2
from geometry.two_d.box_2d import RigidBody2d
from geometry.two_d.contact.contact_point_2d import ContactPoint2d
from geometry.two_d.contact.types import ContactMode, ContactPosition, ContactType
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
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
    non_penetration_cut: sym.Formula
    equal_contact_points: ContactFrameConstraints
    equal_and_opposite_forces: ContactFrameConstraints
    equal_relative_positions: ContactFrameConstraints


@dataclass
class ContactPair2d:
    name: str
    body_A: RigidBody2d
    body_A_contact_location: PolytopeContactLocation
    body_B: RigidBody2d
    body_B_contact_location: PolytopeContactLocation
    contact_type: ContactType
    friction_coeff: float = 0.7

    def create_instance(self, contact_mode: ContactMode) -> "ContactPair2dInstance":
        """
        Creates an instance of the contact pair with the specified contact mode.
        The instance will create a new set of variables and constraints, intended to be used as
        a ctrl point in an optimization program.
        """
        return ContactPair2dInstance(
            self.name,
            self.body_A,
            self.body_A_contact_location,
            self.body_B,
            self.body_B_contact_location,
            self.contact_type,
            contact_mode,
            self.friction_coeff,
        )


class ContactPair2dInstance:
    def __init__(
        self,
        name: str,
        body_A: RigidBody2d,
        body_A_contact_location: PolytopeContactLocation,
        body_B: RigidBody2d,
        body_B_contact_location: PolytopeContactLocation,
        contact_type: ContactType,
        contact_mode: ContactMode,
        friction_coeff: float,
    ) -> None:
        self.name = name
        self.contact_mode = contact_mode
        self.contact_type = contact_type
        self.body_A = body_A
        if contact_mode == ContactMode.ROLLING:
            fix_friction_cone_A = None
            fix_friction_cone_B = None
        else: # Sliding
            if contact_mode == ContactMode.SLIDING_LEFT: # B is sliding left on A
                fix_friction_cone_A = "LEFT"
                fix_friction_cone_B = "LEFT"
            if contact_mode == ContactMode.SLIDING_RIGHT: # B is sliding right on A
                fix_friction_cone_A = "RIGHT"
                fix_friction_cone_B = "RIGHT"
            
        self.contact_point_A = ContactPoint2d(
            body_A,
            body_A_contact_location,
            fix_friction_cone=fix_friction_cone_A,
            friction_coeff=friction_coeff,
            name=f"{name}_{body_A.name}",
        )
        self.body_B = body_B
        self.contact_point_B = ContactPoint2d(
            body_B,
            body_B_contact_location,
            fix_friction_cone=fix_friction_cone_B,
            friction_coeff=friction_coeff,
            name=f"{name}_{body_B.name}",
        )

        # Local position from A to B in A frame
        p_AB_A_x = sym.Variable(f"{name}_p_AB_A_x")
        p_AB_A_y = sym.Variable(f"{name}_p_AB_A_y")
        self.p_AB_A = np.array([p_AB_A_x, p_AB_A_y]).reshape((-1, 1))

        # Rotation from A to B
        self.cos_th = sym.Variable(f"{name}_cos_th")
        self.sin_th = sym.Variable(f"{name}_sin_th")
        self.R_AB = np.array([[self.cos_th, -self.sin_th], [self.sin_th, self.cos_th]])

        # Local position from B to A in B frame
        p_BA_B_x = sym.Variable(f"{name}_p_BA_B_x")
        p_BA_B_y = sym.Variable(f"{name}_p_BA_B_y")
        self.p_BA_B = np.array([p_BA_B_x, p_BA_B_y]).reshape((-1, 1))

    @property
    def bodies(self) -> Tuple[RigidBody2d, RigidBody2d]:
        return self.body_A, self.body_B

    @property
    def contact_points(self) -> Tuple[ContactPoint2d, ContactPoint2d]:
        return (self.contact_point_A, self.contact_point_B)

    @property
    def orientation_variables(self) -> NpVariableArray:
        return np.array([self.cos_th, self.sin_th])

    def _get_nonfixed_contact_point(self) -> ContactPoint2d:
        point = next(
            point
            for point in self.contact_points
            if point.contact_location.pos == ContactPosition.FACE
        )
        return point

    def get_nonfixed_contact_point_variable(self) -> sym.Variable:
        point = self._get_nonfixed_contact_point()
        return point.lam

    def get_nonfixed_contact_position(self) -> NpExpressionArray:
        point = self._get_nonfixed_contact_point()
        # Ignore type for now
        return point.contact_position  # type: ignore

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
        # cos_th^2 + sin_th^2 <= 1
        relaxed_so_2_constraint = (self.R_AB.T.dot(self.R_AB))[0, 0] <= 1
        return relaxed_so_2_constraint

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        # FIX: Remember to remove one of these for point contacts!
        return np.concatenate(
            [
                self.contact_point_A.create_friction_cone_constraints(),
                self.contact_point_B.create_friction_cone_constraints(),
            ]
        )

    def _get_contact_point_of_type(self, type: ContactPosition) -> ContactPoint2d:
        contact_point = next(
            (
                contact_point
                for contact_point in self.contact_points
                if contact_point.contact_location.pos == type
            )
        )  # There will only be one match, hence we use 'next'
        return contact_point

    @staticmethod
    def _create_nonpenetration_hyperplane(
        contact_point: npt.NDArray[np.float64],
        vertex: npt.NDArray[np.float64],
        face_hyperplane: Hyperplane,
        relative_to: Literal["A", "B"],
    ) -> Hyperplane:
        """
        Helper function for analytically creating the hyperplane corresponding to a non-penetration constraint between a vertex and a face.
        """
        p = vertex - contact_point

        if relative_to == "A":
            a_x = face_hyperplane.a[0] * p[0] + face_hyperplane.a[1] * p[1]
            a_y = face_hyperplane.a[1] * p[0] - face_hyperplane.a[0] * p[1]
        else:
            a_x = face_hyperplane.a[0] * p[0] - face_hyperplane.a[1] * p[1]
            a_y = face_hyperplane.a[1] * p[0] + face_hyperplane.a[0] * p[1]

        a = np.array([a_x, a_y]).reshape((-1, 1))
        b = np.zeros(
            a.shape
        )  # A nonpenetration hyperplane always passes through the origin
        return Hyperplane(a, b)

    def create_non_penetration_cut(self) -> sym.Formula:
        vertex_contact = self._get_contact_point_of_type(ContactPosition.VERTEX)
        contact_point: npt.NDArray[np.float64] = vertex_contact.contact_position  # type: ignore
        if not contact_point.dtype == np.float64:
            raise ValueError("dtype of contact point must be np.float64")

        face_contact = self._get_contact_point_of_type(ContactPosition.FACE)
        contact_hyperplane = face_contact.get_contact_hyperplane()

        # Points we are enforcing nonpenetration for
        v1, v2 = vertex_contact.get_neighbouring_vertices()

        # Choose the orientation depending on which contact is vertex and face contact
        relative_to = "A" if vertex_contact == self.contact_point_B else "B"

        # Hyperplanes obtained by enforcing that vertex vi is nonpenetrating
        plane_1 = self._create_nonpenetration_hyperplane(
            contact_point, v1, contact_hyperplane, relative_to
        )
        plane_2 = self._create_nonpenetration_hyperplane(
            contact_point, v2, contact_hyperplane, relative_to
        )

        so_2_cut: Hyperplane = calculate_convex_hull_cut_for_so_2(plane_1, plane_2)
        x = np.array([self.cos_th, self.sin_th]).reshape((-1, 1))
        nonpenetration_cut = (so_2_cut.a.T.dot(x) - so_2_cut.b)[
            0, 0
        ] >= 0  # Use [0,0] to extract scalar value

        return nonpenetration_cut

    def create_constraints(self) -> ContactPairConstraints:
        return ContactPairConstraints(
            self.create_friction_cone_constraints(),
            self.create_relaxed_so2_constraint(),
            self.create_non_penetration_cut(),
            self.create_equal_contact_point_constraints(),
            self.create_equal_and_opposite_forces_constraint(),
            self.create_equal_rel_position_constraints(),
        )

    @property
    def contact_forces(self) -> List[NpExpressionArray]:
        return [self.contact_point_A.contact_force, self.contact_point_B.contact_force]

    def create_squared_contact_forces(self) -> sym.Expression:
        squared_forces = [f.T.dot(f) for f in self.contact_forces]
        sum_of_squared_forces = np.sum(squared_forces, axis=0)[
            0, 0
        ]  # Extract scalar value with [0,0]
        return sum_of_squared_forces

    def get_squared_contact_forces_for_body(self, body: RigidBody2d) -> sym.Expression:
        if body == self.body_A:
            f = self.contact_point_A.contact_force
        elif body == self.body_B:
            f = self.contact_point_B.contact_force
        else:
            raise ValueError("Body not a part of contact pair")

        squared_forces = f.T.dot(f)
        return squared_forces
