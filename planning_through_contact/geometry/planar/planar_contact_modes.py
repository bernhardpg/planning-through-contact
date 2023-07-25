from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    LinearConstraint,
    LinearCost,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    QuadraticCost,
)

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.polyhedron import PolyhedronFormulator
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.tools.utils import forward_differences

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


# TODO: perhaps this class can be unified with the other classes?
@dataclass
class ContinuityVariables:
    """
    A collection of the variables that continuity is enforced over
    """

    p_BF: NpVariableArray | NpExpressionArray
    p_WB: NpVariableArray
    cos_th: sym.Variable
    sin_th: sym.Variable

    @property
    def vector(self) -> NpVariableArray | NpExpressionArray:
        return np.concatenate(
            (self.p_BF.flatten(), self.p_WB.flatten(), (self.cos_th, self.sin_th))  # type: ignore
        )

    def get_pure_variables(self) -> NpVariableArray:
        """
        Function that returns a vector with only the symbolic variables (as opposed to having some be symbolic Expressions)
        """
        if not isinstance(self.p_BF[0, 0], sym.Expression):
            raise RuntimeError(
                "This function should only be called on instances that come from FaceContactMode"
            )

        # NOTE: Very specific way of picking out the variables
        # TODO: clean up this
        lam = list(self.p_BF[0, 0].GetVariables())[0]
        vars = np.concatenate(([lam], self.vector[2:]))
        return vars


@dataclass
class PlanarPlanSpecs:
    num_knot_points_contact: int = 4
    num_knot_points_non_collision: int = 2
    time_in_contact: float = 2
    time_non_collision: float = 0.5


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


@dataclass
class FaceContactVariables(AbstractModeVariables):
    lams: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    normal_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    friction_forces: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    cos_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    sin_ths: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    p_WB_xs: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )
    p_WB_ys: NpVariableArray | npt.NDArray[np.float64]  # (num_knot_points, )

    pv1: npt.NDArray[np.float64]
    pv2: npt.NDArray[np.float64]
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]

    @classmethod
    def from_prog(
        cls,
        prog: MathematicalProgram,
        object_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        num_knot_points: int,
        time_in_mode: float,
    ) -> "FaceContactVariables":
        # Contact positions
        lams = prog.NewContinuousVariables(num_knot_points, "lam")
        pv1, pv2 = object_geometry.get_proximate_vertices_from_location(
            contact_location
        )

        # Contact forces
        normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            normal_vec,
            tangent_vec,
        ) = object_geometry.get_norm_and_tang_vecs_from_location(contact_location)

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")

        dt = time_in_mode / num_knot_points

        return FaceContactVariables(
            num_knot_points,
            time_in_mode,
            dt,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            pv1,
            pv2,
            normal_vec,
            tangent_vec,
        )

    def eval_result(self, result: MathematicalProgramResult) -> "FaceContactVariables":
        return FaceContactVariables(
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            result.GetSolution(self.lams),
            result.GetSolution(self.normal_forces),
            result.GetSolution(self.friction_forces),
            result.GetSolution(self.cos_ths),
            result.GetSolution(self.sin_ths),
            result.GetSolution(self.p_WB_xs),
            result.GetSolution(self.p_WB_ys),
            self.pv1,
            self.pv2,
            self.normal_vec,
            self.tangent_vec,
        )

    @property
    def R_WBs(self):
        Rs = [
            np.array([[cos, -sin], [sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]
        return Rs

    @property
    def p_WBs(self):
        return [
            np.array([x, y]).reshape((2, 1)) for x, y in zip(self.p_WB_xs, self.p_WB_ys)
        ]

    @property
    def f_c_Bs(self):
        return [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(self.normal_forces, self.friction_forces)
        ]

    @property
    def p_c_Bs(self):
        return [lam * self.pv1 + (1 - lam) * self.pv2 for lam in self.lams]

    @property
    def v_WBs(self):
        return forward_differences(self.p_WBs, self.dt)

    @property
    def cos_th_dots(self):
        return forward_differences(self.cos_ths, self.dt)

    @property
    def sin_th_dots(self):
        return forward_differences(self.sin_ths, self.dt)

    @property
    def v_c_Bs(self):
        return forward_differences(
            self.p_c_Bs, self.dt
        )  # NOTE: Not real velocity, only time differentiation of coordinates (not equal as B is not an inertial frame)!

    @property
    def omega_WBs(self):
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
        ]
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, R_WB_dots)]
        return oms

    @property
    def p_c_Ws(self):
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_c_Bs)
        ]

    @property
    def f_c_Ws(self):
        return [R_WB.dot(f_c_B) for f_c_B, R_WB in zip(self.f_c_Bs, self.R_WBs)]


@dataclass
class FaceContactMode(AbstractContactMode):
    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
    ) -> "FaceContactMode":
        name = str(contact_location)
        return cls(
            name,
            specs.num_knot_points_contact,
            specs.time_in_contact,
            contact_location,
            object,
        )

    def __post_init__(self) -> None:
        self.prog = MathematicalProgram()
        self.variables = FaceContactVariables.from_prog(
            self.prog,
            self.object.geometry,
            self.contact_location,
            self.num_knot_points,
            self.time_in_mode,
        )
        self._define_constraints()
        self._define_costs()

    def _define_constraints(self) -> None:
        # TODO: take this from drake simulation
        FRICTION_COEFF = 0.5

        for lam in self.variables.lams:
            self.prog.AddLinearConstraint(lam >= 0)
            self.prog.AddLinearConstraint(lam <= 1)

        # SO(2) constraints
        for c, s in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddConstraint(c**2 + s**2 == 1)

        # Friction cone constraints
        for c_n in self.variables.normal_forces:
            self.prog.AddLinearConstraint(c_n >= 0)
        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            self.prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Quasi-static dynamics
        use_midpoint = True
        for k in range(self.num_knot_points - 1):
            v_WB = self.variables.v_WBs[k]
            omega_WB = self.variables.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            if use_midpoint:
                f_c_B = self._get_midpoint(self.variables.f_c_Bs, k)
                p_c_B = self._get_midpoint(self.variables.p_c_Bs, k)
                R_WB = self._get_midpoint(self.variables.R_WBs, k)
            else:
                f_c_B = self.variables.f_c_Bs[k]
                p_c_B = self.variables.p_c_Bs[k]
                R_WB = self.variables.R_WBs[k]

            x_dot, dyn = self.quasi_static_dynamics(
                v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, self.object.mass
            )
            quasi_static_dynamic_constraint = eq(x_dot - dyn, 0)
            for row in quasi_static_dynamic_constraint:
                self.prog.AddConstraint(row)

        # Ensure sticking on the contact point
        for v_c_B in self.variables.v_c_Bs:
            # NOTE: This is not constraining the real velocity, but it does ensure sticking
            self.prog.AddLinearConstraint(eq(v_c_B, 0))

    def _define_costs(self) -> None:
        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in self.variables.v_WBs]).item()  # type: ignore
        self.prog.AddQuadraticCost(sq_linear_vels)

        sq_angular_vels = np.sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(
                    self.variables.cos_th_dots, self.variables.sin_th_dots
                )
            ]
        )
        self.prog.AddQuadraticCost(sq_angular_vels)  # type: ignore

    def set_slider_initial_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[0] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[0] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[0], pose.pos()))

    def set_slider_final_pose(self, pose: PlanarPose) -> None:
        self.prog.AddLinearConstraint(self.variables.cos_ths[-1] == np.cos(pose.theta))
        self.prog.AddLinearConstraint(self.variables.sin_ths[-1] == np.sin(pose.theta))
        self.prog.AddLinearConstraint(eq(self.variables.p_WBs[-1], pose.pos()))

    def formulate_convex_relaxation(self) -> None:
        self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)

    def get_convex_set(self) -> opt.Spectrahedron:
        if self.relaxed_prog is None:
            raise RuntimeError(
                "Relaxed program must be constructed before convex set can be retrieved"
            )
        return opt.Spectrahedron(self.relaxed_prog)

    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        # TODO(bernhardpg): Make this work with Drake SDP relaxation

        # NOTE: This function relies on the fact that the sdp relaxation
        # returns an ordering of variables [1, x1, x2, ...],
        # where [x1, x2, ...] is the original ordering in self.prog
        idxs = self.prog.FindDecisionVariableIndices(vars)
        idxs_shifted = [
            i + 1 for i in idxs
        ]  # We must shift the indices by one because of the SDP relaxation which adds a 1 as the first variable
        return idxs_shifted

    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> FaceContactVariables:
        # TODO: This can probably be cleaned up somehow
        lams = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.lams, result)  # type: ignore
        normal_forces = self._get_vars_solution_for_vertex_vars(
            vertex.x(), self.variables.normal_forces, result  # type: ignore
        )
        friction_forces = self._get_vars_solution_for_vertex_vars(
            vertex.x(), self.variables.friction_forces, result  # type: ignore
        )
        cos_ths = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.cos_ths, result)  # type: ignore
        sin_ths = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.sin_ths, result)  # type: ignore
        p_WB_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_xs, result)  # type: ignore
        p_WB_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_ys, result)  # type: ignore

        return FaceContactVariables(
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            self.variables.pv1,
            self.variables.pv2,
            self.variables.normal_vec,
            self.variables.tangent_vec,
        )

    def get_continuity_vars(
        self, first_or_last: Literal["first", "last"]
    ) -> ContinuityVariables:
        if first_or_last == "first":
            return ContinuityVariables(
                self.variables.p_c_Bs[0],
                self.variables.p_WBs[0],
                self.variables.cos_ths[0],
                self.variables.sin_ths[0],
            )
        else:
            return ContinuityVariables(
                self.variables.p_c_Bs[-1],
                self.variables.p_WBs[-1],
                self.variables.cos_ths[-1],
                self.variables.sin_ths[-1],
            )

    def get_cost_terms(self) -> Tuple[List[List[int]], List[LinearCost]]:
        if self.relaxed_prog is None:
            raise RuntimeError(
                "Relaxed program must be constructed before cost can be formulated for vertex."
            )

        costs = self.relaxed_prog.linear_costs()
        evaluators = [cost.evaluator() for cost in costs]
        # NOTE: here we must get the indices from the relaxed program!
        var_idxs = [
            self.relaxed_prog.FindDecisionVariableIndices(cost.variables())
            for cost in costs
        ]
        return var_idxs, evaluators

    @staticmethod
    def _get_midpoint(vals, k: int):
        return (vals[k] + vals[k + 1]) / 2

    @staticmethod
    def quasi_static_dynamics(
        v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, OBJECT_MASS
    ):
        G = 9.81
        # TODO(bernhardpg): Compute f_max and tau_max correctly
        f_max = FRICTION_COEFF * G * OBJECT_MASS
        tau_max = f_max * 0.2

        A = np.diag(
            # [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
            [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
        )  # Ellipsoidal Limit surface approximation

        # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"
        # R = np.zeros((3, 3), dtype="O")
        # R[2, 2] = 1
        # R[0:2, 0:2] = R_WB

        # Contact torques
        tau_c_B = cross_2d(p_c_B, f_c_B)

        x_dot = np.concatenate((v_WB, [[omega_WB]]))
        wrench_B = np.concatenate((f_c_B, [[tau_c_B]]))
        dynamics = A.dot(
            wrench_B
        )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference

        # TODO(bernhardpg): Use wrench in world frame after writing unit tests
        # wrench_W = R.dot(wrench_B)
        # dynamics = A.dot(
        #     wrench_@
        # )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference

        # x_dot, f(x,u)
        return x_dot, dynamics  # (3,1), (3,1)


@dataclass
class NonCollisionVariables(AbstractModeVariables):
    p_BF_xs: NpVariableArray | npt.NDArray[np.float64]
    p_BF_ys: NpVariableArray | npt.NDArray[np.float64]
    p_WB_x: sym.Variable | float
    p_WB_y: sym.Variable | float
    cos_th: sym.Variable | float
    sin_th: sym.Variable | float

    @property
    def p_BFs(self):
        return [
            np.expand_dims(np.array([x, y]), 1)
            for x, y in zip(self.p_BF_xs, self.p_BF_ys)
        ]  # (2, 1)

    @property
    def p_WB(self):
        return np.expand_dims(np.array([self.p_WB_x, self.p_WB_y]), 1)  # (2, 1)

    @property
    def R_WBs(self):
        Rs = [
            np.array([[self.cos_th, -self.sin_th], [self.sin_th, self.cos_th]])
        ] * self.num_knot_points
        return Rs

    @property
    def p_WBs(self):
        return [self.p_WB] * self.num_knot_points

    @property
    def v_WBs(self):
        NUM_DIMS = 2
        return [np.zeros((NUM_DIMS, 1))] * self.num_knot_points

    @property
    def omega_WBs(self):
        return [0] * self.num_knot_points

    @property
    def p_c_Ws(self):
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_BFs)
        ]

    @property
    def f_c_Ws(self):
        NUM_DIMS = 2
        return [np.zeros((NUM_DIMS, 1))] * self.num_knot_points


@dataclass
class NonCollisionMode(AbstractContactMode):
    @classmethod
    def create_from_plan_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
    ) -> "NonCollisionMode":
        name = f"NON_COLL_{contact_location.idx}"
        return cls(
            name,
            specs.num_knot_points_non_collision,
            specs.time_non_collision,
            contact_location,
            object,
        )

    def __post_init__(self) -> None:
        self.dt = self.time_in_mode / self.num_knot_points

        self.planes = self.object.geometry.get_planes_for_collision_free_region(
            self.contact_location
        )
        self.prog = MathematicalProgram()
        self.variables = self._define_variables()
        self._define_constraints()
        self._define_cost()

    def _define_variables(self) -> NonCollisionVariables:
        # Finger location
        p_BF_xs = self.prog.NewContinuousVariables(self.num_knot_points, "p_BF_x")
        p_BF_ys = self.prog.NewContinuousVariables(self.num_knot_points, "p_BF_y")

        # We only need one variable for the pose of the object
        p_WB_x = self.prog.NewContinuousVariables(1, "p_WB_x").item()
        p_WB_y = self.prog.NewContinuousVariables(1, "p_WB_y").item()
        cos_th = self.prog.NewContinuousVariables(1, "cos_th").item()
        sin_th = self.prog.NewContinuousVariables(1, "sin_th").item()

        return NonCollisionVariables(
            self.num_knot_points,
            self.time_in_mode,
            self.dt,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def _define_constraints(self) -> None:
        for k in range(self.num_knot_points):
            p_BF = self.variables.p_BFs[k]

            for plane in self.planes:
                dist_to_face = (plane.a.T.dot(p_BF) - plane.b).item()  # a'x >= b
                self.prog.AddLinearConstraint(dist_to_face >= 0)

    def _define_cost(self) -> None:
        position_diffs = np.array(
            [
                p_next - p_curr
                for p_next, p_curr in zip(
                    self.variables.p_BFs[1:], self.variables.p_BFs[:-1]
                )
            ]
        )
        squared_eucl_dist = np.sum([d.T.dot(d) for d in position_diffs.T])
        self.prog.AddCost(squared_eucl_dist)

    def get_variable_indices_in_gcs_vertex(self, vars: NpVariableArray) -> List[int]:
        return self.prog.FindDecisionVariableIndices(vars)

    def get_variable_solutions_for_vertex(
        self, vertex: GcsVertex, result: MathematicalProgramResult
    ) -> NonCollisionVariables:
        # TODO: This can probably be cleaned up somehow
        p_BF_xs = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BF_xs, result)  # type: ignore
        p_BF_ys = self._get_vars_solution_for_vertex_vars(vertex.x(), self.variables.p_BF_ys, result)  # type: ignore
        p_WB_x = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_x, result)  # type: ignore
        p_WB_y = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.p_WB_y, result)  # type: ignore
        cos_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.cos_th, result)  # type: ignore
        sin_th = self._get_var_solution_for_vertex_vars(vertex.x(), self.variables.sin_th, result)  # type: ignore
        return NonCollisionVariables(
            self.variables.num_knot_points,
            self.variables.time_in_mode,
            self.variables.dt,
            p_BF_xs,
            p_BF_ys,
            p_WB_x,
            p_WB_y,
            cos_th,
            sin_th,
        )

    def get_convex_set(self) -> opt.Spectrahedron:
        # Create a temp program without a quadratic cost that we can use to create a polyhedron
        temp_prog = MathematicalProgram()
        x = temp_prog.NewContinuousVariables(self.prog.num_vars(), "x")
        # Some linear constraints will be added as bounding box constraints
        for c in self.prog.GetAllConstraints():
            if not (
                isinstance(c.evaluator(), LinearConstraint)
                or isinstance(c.evaluator(), BoundingBoxConstraint)
            ):
                raise ValueError("Constraints must be linear!")

            idxs = self.get_variable_indices_in_gcs_vertex(c.variables())
            vars = x[idxs]
            temp_prog.AddConstraint(c.evaluator(), vars)

        # NOTE: Here, we are using the Spectrahedron constructor, which is really creating a polyhedron,
        # because there is no PSD constraint. In the future, it is cleaner to use an interface for the HPolyhedron class.
        poly = opt.Spectrahedron(temp_prog)

        # NOTE: They sets will likely be unbounded

        return poly

    def get_convex_set_in_positions(self) -> opt.Spectrahedron:
        # Construct a small temporary program in R^2 that will allow us to check
        # for positional intersections between regions
        NUM_DIMS = 2
        temp_prog = MathematicalProgram()
        x = temp_prog.NewContinuousVariables(NUM_DIMS, "x")

        for plane in self.planes:
            dist_to_face = plane.a.T.dot(x) - plane.b  # a'x >= b
            temp_prog.AddLinearConstraint(ge(dist_to_face, 0))

        # NOTE: Here, we are using the Spectrahedron constructor, which is really creating a polyhedron,
        # because there is no PSD constraint. In the future, it is cleaner to use an interface for the HPolyhedron class.
        # TODO: Replace this with an interface to the HPolyhedron class, once this is implemented in Drake.
        poly = opt.Spectrahedron(temp_prog)

        # NOTE: They sets will likely be unbounded
        return poly

    def get_continuity_vars(
        self, first_or_last: Literal["first", "last"]
    ) -> ContinuityVariables:
        if first_or_last == "first":
            return ContinuityVariables(
                self.variables.p_BFs[0],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )
        else:
            return ContinuityVariables(
                self.variables.p_BFs[-1],
                self.variables.p_WB,
                self.variables.cos_th,  # type: ignore
                self.variables.sin_th,  # type: ignore
            )

    def get_cost_term(self) -> Tuple[List[int], QuadraticCost]:
        assert len(self.prog.quadratic_costs()) == 1

        cost = self.prog.quadratic_costs()[0]  # only one cost term for these modes

        var_idxs = self.get_variable_indices_in_gcs_vertex(cost.variables())
        return var_idxs, cost.evaluator()
