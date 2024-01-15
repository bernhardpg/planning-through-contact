from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


def add_continuity_constraints_btwn_modes(
    outgoing_mode: "AbstractContactMode",
    incoming_mode: "AbstractContactMode",
    edge: GcsEdge,
    only_continuity_on_slider: bool = False,
    continuity_on_pusher_velocities: bool = False,
):
    lhs = outgoing_mode.get_continuity_terms(
        edge, "last", only_continuity_on_slider, continuity_on_pusher_velocities
    )
    rhs = incoming_mode.get_continuity_terms(
        edge, "first", only_continuity_on_slider, continuity_on_pusher_velocities
    )

    constraint = eq(lhs, rhs)
    for c in constraint:
        edge.AddConstraint(c)


@dataclass
class ContinuityVariables:
    """
    A collection of the variables that continuity is enforced over
    """

    p_BP: NpVariableArray | NpExpressionArray
    p_WB: NpVariableArray
    cos_th: sym.Variable
    sin_th: sym.Variable
    # For non-collision modes we also enforce continuity over pusher velocities
    # to ensure smooth transitions between Bezier curves
    v_BP: Optional[NpVariableArray] = None

    def vector(
        self, include_pusher_velocities: bool = False
    ) -> NpVariableArray | NpExpressionArray:
        if include_pusher_velocities:
            assert self.v_BP is not None
            return np.concatenate(
                (self.p_BP.flatten(), self.p_WB.flatten(), (self.cos_th, self.sin_th), self.v_BP.flatten())  # type: ignore
            )
        else:
            return np.concatenate(
                (self.p_BP.flatten(), self.p_WB.flatten(), (self.cos_th, self.sin_th))  # type: ignore
            )

    def get_pure_variables(
        self, include_pusher_velocities: bool = False
    ) -> NpVariableArray:
        """
        Function that returns a vector with only the symbolic variables (as opposed to having some be symbolic Expressions)
        """
        vars = sym.Variables()
        for expr_or_var in self.vector(include_pusher_velocities):
            if isinstance(expr_or_var, sym.Expression):
                vars.insert(expr_or_var.GetVariables())
            elif isinstance(expr_or_var, sym.Variable):
                vars.insert(expr_or_var)
            else:
                raise RuntimeError("Must be a variable or expression!")

        return np.array(list(vars))

    def slider_vector(self) -> NpVariableArray | NpExpressionArray:
        return np.concatenate(
            (self.p_WB.flatten(), (self.cos_th, self.sin_th))  # type: ignore
        )

    def create_expressions_with_vertex_variables(
        self,
        vertex_vars: NpVariableArray,
        mode: "AbstractContactMode",
        only_continuity_on_slider: bool = True,
        continuity_on_pusher_velocities: bool = False,
    ) -> NpExpressionArray:
        if only_continuity_on_slider:
            # TODO(bernhardpg): This needs to be updated if I use equality elimination on NonCollisionModes too
            vars = self.slider_vector()
            exprs = vars  # for just the slider, all entries are variables
        else:  # contuity on both objects
            exprs = self.vector(continuity_on_pusher_velocities)
            vars = self.get_pure_variables(continuity_on_pusher_velocities)

        A, b = sym.DecomposeAffineExpressions(exprs, vars)
        var_idxs = mode.get_variable_indices_in_gcs_vertex(vars)
        expr = A.dot(vertex_vars[var_idxs]) + b
        return expr


@dataclass
class AbstractModeVariables(ABC):
    contact_location: PolytopeContactLocation
    num_knot_points: int
    time_in_mode: float  # TODO: remove, there is no time!
    dt: float  # TODO: Remove, there is no time!
    pusher_radius: float

    @abstractmethod
    def eval_result(self, result: MathematicalProgramResult) -> "AbstractModeVariables":
        pass

    def __len__(self) -> int:
        return self.num_knot_points


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
    config: PlanarPlanConfig

    @abstractmethod
    def get_convex_set(self, make_bounded: bool) -> opt.ConvexSet:
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
    def get_variable_solutions(
        self, result: MathematicalProgramResult
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
        config: PlanarPlanConfig,
    ) -> "AbstractContactMode":
        pass

    def _get_vars_solution_for_vertex_vars(
        self,
        vertex_vars: NpVariableArray,
        vars: NpVariableArray | NpExpressionArray,
        result: MathematicalProgramResult,
    ) -> npt.NDArray[np.float64]:
        # vars can be an expression when we use equality elimination
        if isinstance(vars[0], sym.Expression):
            pure_vars = sym.Variables()
            for expr in vars:
                pure_vars.insert(expr.GetVariables())
            pure_vars = np.array(list(pure_vars))
            relevant_vertex_vars = vertex_vars[
                self.get_variable_indices_in_gcs_vertex(pure_vars)
            ]
            A, b = sym.DecomposeAffineExpressions(vars, pure_vars)
            vertex_exprs = A.dot(relevant_vertex_vars) + b
            return sym.Evaluate(result.GetSolution(vertex_exprs)).flatten()
        else:
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

    def get_continuity_terms(
        self,
        edge: GcsEdge,
        first_or_last: Literal["first", "last"],
        only_continuity_on_slider: bool = False,
        continuity_on_pusher_velocities: bool = False,
    ) -> NpExpressionArray | NpVariableArray:
        vars = self.get_continuity_vars(first_or_last)
        edge_vars = edge.xu() if first_or_last == "last" else edge.xv()
        terms = vars.create_expressions_with_vertex_variables(
            edge_vars, self, only_continuity_on_slider, continuity_on_pusher_velocities
        )
        return terms
