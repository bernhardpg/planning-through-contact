from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq, le
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    ContactMode,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    calculate_convex_hull_cut_for_so_2,
    get_angle_between_planes,
)
from planning_through_contact.geometry.in_plane.contact_force import (
    ContactForceDefinition,
)
from planning_through_contact.geometry.in_plane.contact_point import ContactPoint
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)
from planning_through_contact.tools.utils import evaluate_np_formulas_array


class EvaluatedContactFrameConstraints(NamedTuple):
    in_frame_A: npt.NDArray[np.float64]
    in_frame_B: npt.NDArray[np.float64]


class ContactFrameConstraints(NamedTuple):
    in_frame_A: Union[NpFormulaArray, NpExpressionArray]
    in_frame_B: Union[NpFormulaArray, NpExpressionArray]
    type_A: Literal["linear", "quadratic"]
    type_B: Literal["linear", "quadratic"]

    def evaluate(
        self, result: MathematicalProgramResult
    ) -> EvaluatedContactFrameConstraints:
        evaluated_in_frame_A = evaluate_np_formulas_array(self.in_frame_A, result)
        evaluated_in_frame_B = evaluate_np_formulas_array(self.in_frame_B, result)
        return EvaluatedContactFrameConstraints(
            evaluated_in_frame_A, evaluated_in_frame_B
        )


class PointContactConstraints(NamedTuple):
    friction_cone: NpFormulaArray
    so_2: sym.Formula
    relaxed_so_2: sym.Formula
    equal_contact_points: ContactFrameConstraints
    equal_and_opposite_forces: ContactFrameConstraints
    equal_relative_positions: ContactFrameConstraints
    rotation_bounds: NpFormulaArray
    convex_hull_bounds: NpFormulaArray
    non_penetration_cut: Optional[sym.Formula]


class LineContactConstraints(NamedTuple):
    friction_cone: NpFormulaArray
    convex_hull_bounds: NpFormulaArray


@dataclass
class ContactPairDefinition:
    name: str
    body_A: RigidBody
    body_A_contact_location: PolytopeContactLocation
    body_B: RigidBody
    body_B_contact_location: PolytopeContactLocation
    friction_coeff: float = 0.5

    def create_pair(
        self,
        contact_mode: ContactMode,
        contact_pos_var: Optional[sym.Variable] = None,
        instance_postfix: Optional[str] = None,
    ) -> "AbstractContactPair":
        """
        Creates the contact pair with the specified contact mode, from the contact pair definition.
        The pair will contain a new set of variables and constraints, intended to be used as
        a ctrl point in an optimization program.
        """

        name = (
            self.name if instance_postfix is None else f"{self.name}_{instance_postfix}"
        )
        if self.body_A_contact_location.pos == self.body_B_contact_location.pos:
            return FaceOnFaceContact(
                name,
                self.body_A,
                self.body_A_contact_location,
                self.body_B,
                self.body_B_contact_location,
                contact_mode,
                self.friction_coeff,
                contact_pos_var,
            )

        else:
            return PointOnFaceContact(
                name,
                self.body_A,
                self.body_A_contact_location,
                self.body_B,
                self.body_B_contact_location,
                contact_mode,
                self.friction_coeff,
                contact_pos_var,
            )


@dataclass
class AbstractContactPair(ABC):
    name: str
    body_A: RigidBody
    body_A_contact_location: PolytopeContactLocation
    body_B: RigidBody
    body_B_contact_location: PolytopeContactLocation
    contact_mode: ContactMode
    friction_coeff: float
    contact_pos_var: Optional[sym.Variable] = None

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
    def contact_points(self) -> List[ContactPoint]:
        pass

    @abstractmethod
    def get_nonfixed_contact_point(self) -> ContactPoint:
        pass

    def get_nonfixed_contact_point_variable(self) -> sym.Variable:
        point = self.get_nonfixed_contact_point()
        return point.lam

    def get_nonfixed_contact_position(
        self,
    ) -> Union[NpExpressionArray, npt.NDArray[np.float64]]:
        point = self.get_nonfixed_contact_point()
        return point.contact_position

    @property
    @abstractmethod
    def variables(self) -> NpVariableArray:
        pass

    @abstractmethod
    def create_constraints(
        self,
    ) -> Union[PointContactConstraints, LineContactConstraints]:
        pass

    @property
    @abstractmethod
    def contact_forces(self) -> List[NpExpressionArray]:
        pass

    @abstractmethod
    def create_squared_contact_forces(self) -> sym.Expression:
        pass

    @abstractmethod
    def get_squared_contact_forces_for_body(self, body: RigidBody) -> sym.Expression:
        pass

    @property
    def bodies(self) -> Tuple[RigidBody, RigidBody]:
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

        # TODO(bernhardpg): Use length of face, not a hardcoded value
        # Plan:
        # Find the length of both faces
        # Pick the smallest face, and constrain it to lie within the largest face

        body_A_face_length = self.body_A.geometry.get_face_length(
            self.body_A_contact_location
        )
        body_B_face_length = self.body_B.geometry.get_face_length(
            self.body_B_contact_location
        )

        self.shortest_face_length = min(body_A_face_length, body_B_face_length)
        self.longest_face_length = max(body_A_face_length, body_B_face_length)

        self.BOX_WIDTH = 0.2  # FIX: This should be fixed!

        left_force = ContactForceDefinition(
            f"{self.name}_{self.body_A.name}_left",
            self.friction_coeff,
            self.body_A_contact_location,
            self.body_A.geometry,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
            displacement=-self.shortest_face_length / 2,
        )

        right_force = ContactForceDefinition(
            f"{self.name}_{self.body_A.name}_right",
            self.friction_coeff,
            self.body_A_contact_location,
            self.body_A.geometry,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
            displacement=self.shortest_face_length / 2,
        )

        # We only need to carry around one contact point for face-on-face
        # contact (the forces as always exactly equal and opposite)
        self.contact_point_A = ContactPoint(
            self.body_A,
            self.body_A_contact_location,
            [left_force, right_force],
            self.friction_coeff,
            name=f"{self.name}_{self.body_A.name}",
            contact_position_var=self.contact_pos_var,
        )

    @property
    def R_AB(self) -> npt.NDArray[np.float64]:
        plane_A = self.body_A.geometry.get_hyperplane_from_location(
            self.body_A_contact_location
        )
        plane_B = self.body_B.geometry.get_hyperplane_from_location(
            self.body_B_contact_location
        )
        theta = get_angle_between_planes(plane_A, plane_B)
        if not np.isclose(theta % np.pi, 0):
            raise NotImplementedError(
                "Caution: this functionality has not been properly tested and may be incorrect"
            )

        return two_d_rotation_matrix_from_angle(theta)

    @property
    def p_AB_A(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        p_Ac = self.contact_point_A.contact_position
        p_Bc = self.body_B.geometry.get_shortest_vec_from_com_to_loc(
            self.body_B_contact_location
        )
        p_cB = -p_Bc
        p_AB = p_Ac + self.R_AB.dot(p_cB)

        raise NotImplementedError(
            "Caution: this functionality has not been properly tested and may be incorrect"
        )
        return p_AB

    @property
    def p_BA_B(self) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        return -self.R_AB.dot(self.p_AB_A)

    @property
    def contact_points(self) -> List[ContactPoint]:
        return [self.contact_point_A]

    def get_nonfixed_contact_point(self) -> ContactPoint:
        return self.contact_point_A

    @property
    def variables(self) -> NpVariableArray:
        return self.contact_point_A.variables

    # TODO(bernhardpg): Rename
    def create_convex_hull_bounds(self) -> NpFormulaArray:
        lam = self.get_nonfixed_contact_point_variable()
        # Make sure we stay inside the largest face
        normalized_distance_from_point_to_edge = (
            self.shortest_face_length / 2
        ) / self.longest_face_length
        bounds = np.array(
            [
                lam + normalized_distance_from_point_to_edge <= 1,
                0 <= lam - normalized_distance_from_point_to_edge,
            ]
        )
        return bounds

    def create_constraints(self) -> LineContactConstraints:
        return LineContactConstraints(
            self.create_friction_cone_constraints(), self.create_convex_hull_bounds()
        )

    @property
    def contact_forces(self) -> List[NpExpressionArray]:
        return self.contact_point_A.get_contact_forces()

    def create_squared_contact_forces(self) -> sym.Expression:
        squared_forces = [f.T.dot(f) for f in self.contact_forces]
        sum_of_squared_forces = np.sum(squared_forces, axis=0)[
            0, 0
        ]  # Extract scalar value with [0,0]
        return sum_of_squared_forces

    def get_squared_contact_forces_for_body(self, body: RigidBody) -> sym.Expression:
        if body == self.body_A:
            forces = self.contact_point_A.get_contact_forces()
        elif body == self.body_B:
            # NOTE: In a line contact, we only define one-sided forces. Hence we return the same forces here
            forces = self.contact_point_A.get_contact_forces()
        else:
            raise ValueError("Body not a part of contact pair")

        squared_forces = np.sum([f.T.dot(f) for f in forces], axis=0).item()

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
            self.body_A.geometry,
            fixed_to_friction_cone_boundary=fix_friction_cone_A,
        )

        force_def_B = ContactForceDefinition(
            f"{self.name}_{self.body_B.name}",
            self.friction_coeff,
            self.body_B_contact_location,
            self.body_B.geometry,
            fixed_to_friction_cone_boundary=fix_friction_cone_B,
        )

        self.contact_point_A = ContactPoint(
            self.body_A,
            self.body_A_contact_location,
            [force_def_A],
            self.friction_coeff,
            name=f"{self.name}_{self.body_A.name}",
        )
        self.contact_point_B = ContactPoint(
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
    def p_AB_A(self) -> NpVariableArray:
        return np.array([self._p_AB_A_x, self._p_AB_A_y]).reshape((-1, 1))

    @property
    def p_BA_B(self) -> NpVariableArray:
        return np.array([self._p_BA_B_x, self._p_BA_B_y]).reshape((-1, 1))

    @property
    def contact_points(self) -> List[ContactPoint]:
        return [self.contact_point_A, self.contact_point_B]

    @property
    def orientation_variables(self) -> NpVariableArray:
        return np.array([self._cos_th, self._sin_th])

    def get_nonfixed_contact_point(self) -> ContactPoint:
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

    def _determine_constraint_type(
        self, contact_point: ContactPoint
    ) -> Literal["linear", "quadratic"]:
        return (
            "linear"
            if contact_point.contact_location.pos == ContactLocation.VERTEX
            else "quadratic"
        )

    def create_equal_contact_point_constraints(self) -> ContactFrameConstraints:
        # One of these will be constant (one must lie on a corner)
        p_Ac_A = self.contact_point_A.contact_position
        p_Bc_B = self.contact_point_B.contact_position

        p_Bc_A = self.R_AB.dot(p_Bc_B)
        eq_contact_point_in_A = eq(p_Ac_A, self.p_AB_A + p_Bc_A).flatten()
        constraint_type_A = self._determine_constraint_type(self.contact_point_B)

        p_Ac_B = self.R_AB.T.dot(p_Ac_A)
        eq_contact_point_in_B = eq(p_Bc_B, self.p_BA_B + p_Ac_B).flatten()
        constraint_type_B = self._determine_constraint_type(self.contact_point_A)

        return ContactFrameConstraints(
            eq_contact_point_in_A,
            eq_contact_point_in_B,
            constraint_type_A,
            constraint_type_B,
        )

    def create_equal_rel_position_constraints(self) -> ContactFrameConstraints:
        rel_pos_equal_in_A = eq(self.p_AB_A, -self.R_AB.dot(self.p_BA_B)).flatten()
        rel_pos_equal_in_B = eq(self.p_BA_B, -self.R_AB.T.dot(self.p_AB_A)).flatten()

        return ContactFrameConstraints(
            rel_pos_equal_in_A, rel_pos_equal_in_B, "quadratic", "quadratic"
        )

    def create_equal_and_opposite_forces_constraint(self) -> ContactFrameConstraints:
        f_c_A = self.contact_point_A.contact_force
        f_c_B = self.contact_point_B.contact_force

        equal_and_opposite_in_A = eq(f_c_A, -self.R_AB.dot(f_c_B)).flatten()
        equal_and_opposite_in_B = eq(f_c_B, -self.R_AB.T.dot(f_c_A)).flatten()

        return ContactFrameConstraints(
            equal_and_opposite_in_A, equal_and_opposite_in_B, "quadratic", "quadratic"
        )

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

    def create_convex_hull_bounds(self) -> NpFormulaArray:
        lam = self.get_nonfixed_contact_point_variable()
        bounds = np.array([lam <= 1, 0 <= lam])
        return bounds

    def create_relaxed_so2_constraint(self) -> sym.Formula:
        # cos_th^2 + sin_th^2 <= 1
        relaxed_so_2_constraint = (self.R_AB.T.dot(self.R_AB))[0, 0] <= 1
        return relaxed_so_2_constraint

    def _get_contact_point_of_type(self, type: ContactLocation) -> ContactPoint:
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
        # NOTE: The code is not unit tested, as it is not currently used. Kept around in case it will become useful.
        raise NotImplementedError("Note: This is not yet tested!")

        vertex_contact = self._get_contact_point_of_type(ContactLocation.VERTEX)
        contact_point: npt.NDArray[np.float64] = vertex_contact.contact_position  # type: ignore
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

    def create_constraints(self) -> PointContactConstraints:
        return PointContactConstraints(
            self.create_friction_cone_constraints(),
            self.create_so2_constraint(),
            self.create_relaxed_so2_constraint(),
            self.create_equal_contact_point_constraints(),
            self.create_equal_and_opposite_forces_constraint(),
            self.create_equal_rel_position_constraints(),
            self.create_rotation_bounds(),
            self.create_convex_hull_bounds(),
            None,  # We are so far not using non-penetration cuts
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

    def get_squared_contact_forces_for_body(self, body: RigidBody) -> sym.Expression:
        if body == self.body_A:
            f = self.contact_point_A.contact_force
        elif body == self.body_B:
            f = self.contact_point_B.contact_force
        else:
            raise ValueError("Body not a part of contact pair")

        squared_forces = f.T.dot(f)
        return squared_forces
