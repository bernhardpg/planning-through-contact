from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgramResult

from geometry.hyperplane import (
    Hyperplane,
    calculate_convex_hull_cut_for_so_2,
    get_angle_between_planes,
)
from geometry.two_d.box_2d import RigidBody2d
from geometry.two_d.contact.contact_point_2d import (
    ContactForceDefinition,
    ContactPoint2d,
)
from geometry.two_d.contact.types import ContactLocation, ContactMode
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import two_d_rotation_matrix_from_angle
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


class PairContactConstraints(NamedTuple):
    friction_cone: NpFormulaArray
    so_2: sym.Formula
    relaxed_so_2: sym.Formula
    non_penetration_cut: sym.Formula
    equal_contact_points: ContactFrameConstraints
    equal_and_opposite_forces: ContactFrameConstraints
    equal_relative_positions: ContactFrameConstraints
    rotation_bounds: NpFormulaArray
    lam_bounds: NpFormulaArray


class LineContactConstraints(NamedTuple):
    friction_cone: NpFormulaArray


@dataclass
class ContactPairDefinition:
    name: str
    body_A: RigidBody2d
    body_A_contact_location: PolytopeContactLocation
    body_B: RigidBody2d
    body_B_contact_location: PolytopeContactLocation
    friction_coeff: float = 0.7

    def create_instance(
        self, contact_mode: ContactMode, instance_postfix: Optional[str] = None
    ) -> "AbstractContactPair":
        """
        Creates an instance of the contact pair with the specified contact mode.
        The instance will create a new set of variables and constraints, intended to be used as
        a ctrl point in an optimization program.
        """

        full_instance_name = (
            self.name if instance_postfix is None else f"{self.name}_{instance_postfix}"
        )
        if self.body_A_contact_location.pos == self.body_B_contact_location.pos:
            return FaceOnFaceContact(
                full_instance_name,
                self.body_A,
                self.body_A_contact_location,
                self.body_B,
                self.body_B_contact_location,
                contact_mode,
                self.friction_coeff,
            )

        else:
            return PointOnFaceContact(
                full_instance_name,
                self.body_A,
                self.body_A_contact_location,
                self.body_B,
                self.body_B_contact_location,
                contact_mode,
                self.friction_coeff,
            )


@dataclass
class AbstractContactPair(ABC):
    name: str
    body_A: RigidBody2d
    body_A_contact_location: PolytopeContactLocation
    body_B: RigidBody2d
    body_B_contact_location: PolytopeContactLocation
    contact_mode: ContactMode
    friction_coeff: float

    @property
    @abstractmethod
    def R_AB(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        pass

    @property
    @abstractmethod
    def p_AB_A(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        pass

    @property
    @abstractmethod
    def p_BA_B(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        pass

    @property
    @abstractmethod
    def contact_points(self) -> List[ContactPoint2d]:
        pass

    @abstractmethod
    def get_nonfixed_contact_point(self) -> ContactPoint2d:
        pass

    def get_nonfixed_contact_point_variable(self) -> sym.Variable:
        point = self.get_nonfixed_contact_point()
        return point.lam

    def get_nonfixed_contact_position(
        self,
    ) -> List[Union[NpExpressionArray, npt.NDArray[np.float64]]]:
        point = self.get_nonfixed_contact_point()
        return point.get_contact_positions()

    @property
    @abstractmethod
    def variables(self) -> NpVariableArray:
        pass

    @abstractmethod
    def create_constraints(
        self,
    ) -> Union[PairContactConstraints, LineContactConstraints]:
        pass

    @property
    @abstractmethod
    def contact_forces(self) -> List[NpExpressionArray]:
        pass

    @abstractmethod
    def create_squared_contact_forces(self) -> sym.Expression:
        pass

    @abstractmethod
    def get_squared_contact_forces_for_body(self, body: RigidBody2d) -> sym.Expression:
        pass

    @property
    def bodies(self) -> Tuple[RigidBody2d, RigidBody2d]:
        return self.body_A, self.body_B

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        face_contact = self.get_nonfixed_contact_point()
        return face_contact.create_friction_cone_constraints()

    # TODO: this function could use some cleanup
    def _calculate_friction_cone_states(
        self,
        contact_mode: ContactMode,
        body_A_contact_location: PolytopeContactLocation,
    ) -> Tuple[Literal["LEFT", "RIGHT"], Literal["LEFT", "RIGHT"]]:
        moving_contact_point: Literal["A", "B"] = (
            "A" if body_A_contact_location.pos == ContactLocation.FACE else "B"
        )
        if contact_mode == ContactMode.ROLLING:
            fix_friction_cone = None
        else:  # Sliding
            if contact_mode == ContactMode.SLIDING_LEFT:  # B is sliding left on A
                fix_friction_cone = "LEFT"
            elif contact_mode == ContactMode.SLIDING_RIGHT:  # B is sliding left on A
                fix_friction_cone = "RIGHT"

        fix_friction_cone_A = None
        fix_friction_cone_B = None
        if fix_friction_cone is not None:
            fix_friction_cone_A = (
                fix_friction_cone if moving_contact_point == "A" else None
            )
            fix_friction_cone_B = (
                fix_friction_cone if moving_contact_point == "B" else None
            )

        return fix_friction_cone_A, fix_friction_cone_B  # type: ignore


@dataclass
class FaceOnFaceContact(AbstractContactPair):
    def __post_init__(
        self,
    ) -> None:
        fix_friction_cone_A, _ = self._calculate_friction_cone_states(
            self.contact_mode, self.body_A_contact_location
        )
        left_force = ContactForceDefinition(
            f"{self.name}_{self.body_A.name}",
            self.friction_coeff,
            self.body_A_contact_location,
            self.body_A,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
            displacement=-0.1,  # FIX: Should not be hardcoded
        )

        right_force = ContactForceDefinition(
            f"{self.name}_{self.body_A.name}",
            self.friction_coeff,
            self.body_A_contact_location,
            self.body_A,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
            displacement=0.1,  # FIX: Should not be hardcoded
        )

        self.contact_point_A = ContactPoint2d(
            self.body_A,
            self.body_A_contact_location,
            [left_force, right_force],
            self.friction_coeff,
            name=f"{self.name}_{self.body_A.name}",
        )

    @property
    def R_AB(self) -> npt.NDArray[np.float64]:
        plane_A = self.body_A.get_hyperplane_from_location(self.body_A_contact_location)
        plane_B = self.body_B.get_hyperplane_from_location(self.body_B_contact_location)
        theta = get_angle_between_planes(plane_A, plane_B)
        return two_d_rotation_matrix_from_angle(theta)

    @property
    def p_BA_B(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        return -self.R_AB.dot(self.p_AB_A)

    @property
    def p_AB_A(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        floating_pos = self.contact_point_A._contact_position
        pos_to_com_B = self.body_B.get_shortest_vec_from_com_to_face(
            self.body_B_contact_location
        )
        relative_position = floating_pos + self.R_AB.dot(pos_to_com_B)
        return relative_position

    @property
    def contact_points(self) -> List[ContactPoint2d]:
        return [self.contact_point_A]

    def get_nonfixed_contact_point(self) -> ContactPoint2d:
        return self.contact_point_A

    @property
    def variables(self) -> NpVariableArray:
        return self.contact_point_A.variables

    def create_constraints(self) -> LineContactConstraints:
        return LineContactConstraints(self.create_friction_cone_constraints())

    @property
    def contact_forces(self) -> List[NpExpressionArray]:
        return [self.contact_point_A.contact_force]

    def create_squared_contact_forces(self) -> sym.Expression:
        squared_forces = [f.T.dot(f) for f in self.contact_forces]
        sum_of_squared_forces = np.sum(squared_forces, axis=0)[
            0, 0
        ]  # Extract scalar value with [0,0]
        return sum_of_squared_forces

    def get_squared_contact_forces_for_body(self, body: RigidBody2d) -> sym.Expression:
        if body == self.body_A:
            forces = self.contact_point_A.get_contact_forces()
        elif body == self.body_B:
            raise ValueError("Cannot get contact force for body B in a line contact")
            # f = -self.contact_point_A.contact_force
        else:
            raise ValueError("Body not a part of contact pair")

        squared_forces = np.sum([f.T.dot(f) for f in forces], axis=0)

        return squared_forces


@dataclass
class PointOnFaceContact(AbstractContactPair):
    def __post_init__(self) -> None:
        (
            fix_friction_cone_A,
            fix_friction_cone_B,
        ) = self._calculate_friction_cone_states(
            self.contact_mode, self.body_A_contact_location
        )

        force_def_A = ContactForceDefinition(
            f"{self.name}_{self.body_A.name}",
            self.friction_coeff,
            self.body_A_contact_location,
            self.body_A,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
        )

        force_def_B = ContactForceDefinition(
            f"{self.name}_{self.body_B.name}",
            self.friction_coeff,
            self.body_B_contact_location,
            self.body_B,
            fixed_to_friction_cone_boundary=fix_friction_cone_B,
        )

        self.contact_point_A = ContactPoint2d(
            self.body_A,
            self.body_A_contact_location,
            [force_def_A],
            self.friction_coeff,
            name=f"{self.name}_{self.body_A.name}",
        )
        self.contact_point_B = ContactPoint2d(
            self.body_B,
            self.body_B_contact_location,
            [force_def_B],
            self.friction_coeff,
            name=f"{self.name}_{self.body_B.name}",
        )

        # NOTE: These variables cannot be defined as part of the property, because then they will be instantiated multiple times.

        # Rotation from A to B
        self._cos_th = sym.Variable(f"{self.name}_cos_th")
        self._sin_th = sym.Variable(f"{self.name}_sin_th")

        # Local position from A to B in A frame
        self._p_AB_A_x = sym.Variable(f"{self.name}_p_AB_A_x")
        self._p_AB_A_y = sym.Variable(f"{self.name}_p_AB_A_y")

        # Local position from B to A in B frame
        self._p_BA_B_x = sym.Variable(f"{self.name}_p_BA_B_x")
        self._p_BA_B_y = sym.Variable(f"{self.name}_p_BA_B_y")

    @property
    def R_AB(self) -> NpExpressionArray:
        return np.array([[self._cos_th, -self._sin_th], [self._sin_th, self._cos_th]])

    @property
    def p_AB_A(self) -> NpExpressionArray:
        return np.array([self._p_AB_A_x, self._p_AB_A_y]).reshape((-1, 1))

    @property
    def p_BA_B(self) -> NpExpressionArray:
        return np.array([self._p_BA_B_x, self._p_BA_B_y]).reshape((-1, 1))

    @property
    def contact_points(self) -> List[ContactPoint2d]:
        return [self.contact_point_A, self.contact_point_B]

    @property
    def orientation_variables(self) -> NpVariableArray:
        return np.array([self._cos_th, self._sin_th])

    def get_nonfixed_contact_point(self) -> ContactPoint2d:
        point = next(
            point
            for point in self.contact_points
            if point.contact_location.pos == ContactLocation.FACE
        )
        return point

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
        p_Ac_A = self.contact_point_A._contact_position
        p_Bc_B = self.contact_point_B._contact_position

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

    def create_so2_constraint(self) -> sym.Formula:
        # cos_th^2 + sin_th^2 == 1
        so_2_constraint = (self.R_AB.T.dot(self.R_AB))[0, 0] == 1
        return so_2_constraint

    def create_rotation_bounds(self) -> NpFormulaArray:
        ones = np.ones(self.orientation_variables.size)
        rotation_bounds = np.concatenate(
            [
                le(self.orientation_variables, ones),
                le(-ones, self.orientation_variables),
            ]
        )
        return rotation_bounds

    def create_lam_bounds(self) -> NpFormulaArray:
        lam = self.get_nonfixed_contact_point_variable()
        # TODO remmeber to do this for line contact too!
        bounds = np.array([lam <= 1, 0 <= lam])
        return bounds

    def create_relaxed_so2_constraint(self) -> sym.Formula:
        # cos_th^2 + sin_th^2 <= 1
        relaxed_so_2_constraint = (self.R_AB.T.dot(self.R_AB))[0, 0] <= 1
        return relaxed_so_2_constraint

    def _get_contact_point_of_type(self, type: ContactLocation) -> ContactPoint2d:
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
        vertex_contact = self._get_contact_point_of_type(ContactLocation.VERTEX)
        contact_point: npt.NDArray[np.float64] = vertex_contact._contact_position  # type: ignore
        if not contact_point.dtype == np.float64:
            raise ValueError("dtype of contact point must be np.float64")

        face_contact = self._get_contact_point_of_type(ContactLocation.FACE)
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
        x = np.array([self._cos_th, self._sin_th]).reshape((-1, 1))
        nonpenetration_cut = (so_2_cut.a.T.dot(x) - so_2_cut.b)[
            0, 0
        ] >= 0  # Use [0,0] to extract scalar value

        return nonpenetration_cut

    def create_constraints(self) -> PairContactConstraints:
        return PairContactConstraints(
            self.create_friction_cone_constraints(),
            self.create_so2_constraint(),
            self.create_relaxed_so2_constraint(),
            self.create_non_penetration_cut(),
            self.create_equal_contact_point_constraints(),
            self.create_equal_and_opposite_forces_constraint(),
            self.create_equal_rel_position_constraints(),
            self.create_rotation_bounds(),
            self.create_lam_bounds(),
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
