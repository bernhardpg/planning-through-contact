from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


def add_continuity_constraints_btwn_modes(
    outgoing_mode: "AbstractContactMode",
    incoming_mode: "AbstractContactMode",
    edge: GcsEdge,
):
    incoming_vars_first = incoming_mode.get_continuity_vars("first")
    rhs = incoming_vars_first.create_expressions_with_vertex_variables(
        edge.xv(), incoming_mode
    )

    outgoing_vars_last = outgoing_mode.get_continuity_vars("last")
    lhs = outgoing_vars_last.create_expressions_with_vertex_variables(
        edge.xu(), outgoing_mode
    )

    constraint = eq(lhs, rhs)
    for c in constraint:
        edge.AddConstraint(c)


@dataclass
class ContinuityVariables:
    """
    A collection of the variables that continuity is enforced over
    """

    p_BF: NpVariableArray | NpExpressionArray
    p_WB: NpVariableArray
    cos_th: sym.Variable
    sin_th: sym.Variable

    def vector(self) -> NpVariableArray | NpExpressionArray:
        return np.concatenate(
            (self.p_BF.flatten(), self.p_WB.flatten(), (self.cos_th, self.sin_th))  # type: ignore
        )

    def get_pure_variables(self) -> NpVariableArray:
        """
        Function that returns a vector with only the symbolic variables (as opposed to having some be symbolic Expressions)
        """
        # FaceContactMode: some variables are sym.Expression
        if isinstance(self.p_BF[0, 0], sym.Expression):
            lam = list(self.p_BF[0, 0].GetVariables())[0]
            vars = np.concatenate(([lam], self.vector()[2:]))
            return vars
        else:  # NonCollisionMode: All variables are just sym.Variable
            return self.vector()

    def create_expressions_with_vertex_variables(
        self, vertex_vars: NpVariableArray, mode: "AbstractContactMode"
    ) -> NpExpressionArray:
        A, b = sym.DecomposeAffineExpressions(self.vector(), self.get_pure_variables())
        var_idxs = mode.get_variable_indices_in_gcs_vertex(self.get_pure_variables())
        rhs = A.dot(vertex_vars[var_idxs]) + b
        return rhs


@dataclass
class AbstractModeVariables(ABC):
    num_knot_points: int
    time_in_mode: float
    dt: float

    @abstractmethod
    def eval_result(self, result: MathematicalProgramResult) -> "AbstractModeVariables":
        pass

    @property
    @abstractmethod
    def R_WBs(self):
        pass

    @property
    @abstractmethod
    def p_WBs(self):
        pass

    @property
    @abstractmethod
    def v_WBs(self):
        pass

    @property
    @abstractmethod
    def omega_WBs(self):
        pass

    @property
    @abstractmethod
    def p_c_Ws(self):
        pass

    @property
    @abstractmethod
    def f_c_Ws(self):
        pass


@dataclass
class AbstractContactMode(ABC):
    """
    Abstract base class for planar pushing contact modes.

    Each contact mode will create a mathematicalprogram to handle variables and constraints.
    """

    name: str
    num_knot_points: int
    time_in_mode: float
    contact_location: PolytopeContactLocation
    object: RigidBody

    @abstractmethod
    def get_convex_set(self) -> opt.ConvexSet:
        pass

    @abstractmethod
    def get_continuity_vars(
        self, first_or_last: Literal["first", "last"]
    ) -> ContinuityVariables:
        pass

    @abstractmethod
    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> AbstractModeVariables:
        pass

    @abstractmethod
    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
    ) -> "AbstractContactMode":
        pass

    def _get_vars_solution_for_vertex_vars(
        self,
        vertex_vars: NpVariableArray,
        vars: NpVariableArray,
        result: MathematicalProgramResult,
    ) -> npt.NDArray[np.float64]:
        return result.GetSolution(
            vertex_vars[self.get_variable_indices_in_gcs_vertex(vars)]
        )

    def _get_var_solution_for_vertex_vars(
        self,
        vertex_vars: NpVariableArray,
        var: sym.Variable,
        result: MathematicalProgramResult,
    ) -> float:
        return result.GetSolution(
            vertex_vars[self.get_variable_indices_in_gcs_vertex(np.array([var]))]
        ).item()
