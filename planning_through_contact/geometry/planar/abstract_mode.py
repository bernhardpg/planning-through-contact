from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.continuity_variables import (
    ContinuityVariables,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.tools.types import NpVariableArray

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class AbstractModeVariables(ABC):
    num_knot_points: int
    time_in_mode: float
    dt: float

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
