from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    Spectrahedron,
)
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    CommonSolverOption,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
)
from pydrake.symbolic import DecomposeAffineExpressions, Variable
from underactuated.exercises.humanoids.footstep_planning_gcs_utils import plot_rectangle

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    LinTrajSegment,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.tools.utils import evaluate_np_expressions_array

GcsVertex = GraphOfConvexSets.Vertex
GcsEdge = GraphOfConvexSets.Edge


@dataclass
class InPlaneSteppingStone:
    x_pos: float
    z_pos: float
    width: float
    name: Optional[str] = None

    @property
    def height(self) -> float:
        return self.z_pos

    @property
    def center(self) -> npt.NDArray[np.float64]:
        """
        Returns the surface center of the stone.
        """
        return np.array([self.x_pos, self.z_pos])

    @property
    def x_min(self) -> float:
        return self.x_pos - self.width / 2

    @property
    def x_max(self) -> float:
        return self.x_pos + self.width / 2

    def plot(self, **kwargs):
        center = np.array([self.x_pos, self.z_pos / 2])

        return plot_rectangle(center, self.width, self.height, **kwargs)


class InPlaneTerrain:
    def __init__(self) -> None:
        self.stepping_stones = []

    def add_stone(
        self,
        x_pos: float,
        z_pos: float,
        width: float,
        name: Optional[str] = None,
    ) -> InPlaneSteppingStone:
        stone = InPlaneSteppingStone(x_pos, z_pos, width, name)
        self.stepping_stones.append(stone)
        return stone

    def get_stone_by_name(self, name):
        for stone in self.stepping_stones:
            if stone.name == name:
                return stone

        raise ValueError(f"No stone in the terrain has name {name}.")

    @property
    def max_height(self) -> float:
        return max([s.height for s in self.stepping_stones])

    def plot(
        self, title: Optional[str] = None, max_height: Optional[float] = None, **kwargs
    ):
        # make light green the default facecolor
        if not "facecolor" in kwargs:
            kwargs["facecolor"] = [0, 1, 0, 0.1]

        # plot stepping stones disposition
        labels = ["Stepping stone", None]
        for i, stone in enumerate(self.stepping_stones):
            stone.plot(label=labels[min(i, 1)], **kwargs)

        if title is not None:
            plt.title(title)
            # get current plot axis if one is not given
        ax = plt.gca()

        if max_height:
            ax.set_ylim((0.0, max_height))


@dataclass
class PotatoRobot:
    # TODO(bernhardpg): set reasonable inertia properties
    # This is from a sphere with mass = 50 kg and all axis = 0.5 meter
    mass: float = 50.0  # kg
    # TODO(bernhardpg): compute inertia from dimensions
    inertia: float = 5.0  # kg m**2
    foot_length: float = 0.3  # m
    foot_height: float = 0.15  # m
    step_span: float = 0.8  # m
    desired_com_height: float = 1.5  # m
    size: Tuple[float, float, float] = (0.5, 0.5, 1.0)

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def depth(self) -> float:
        return self.size[1]

    @property
    def height(self) -> float:
        return self.size[2]


@dataclass
class FootstepPlanningConfig:
    robot: PotatoRobot = field(default_factory=lambda: PotatoRobot())
    period: float = 1.0
    period_steps: int = 6

    @property
    def dt(self) -> float:
        return self.period / self.period_steps


@dataclass
class KnotPointValue:
    p_WB: npt.NDArray[np.float64]
    theta_WB: float
    # TODO(bernhardpg): Expand to Left and Right foot
    p_BF_W: npt.NDArray[np.float64]
    f_F_1W: npt.NDArray[np.float64]
    f_F_2W: npt.NDArray[np.float64]


@dataclass
class FootstepPlanKnotPoints:
    p_WB: npt.NDArray[np.float64]
    theta_WB: npt.NDArray[np.float64]
    # TODO(bernhardpg): Expand to Left and Right foot
    p_BF_W: npt.NDArray[np.float64]
    f_F_1W: npt.NDArray[np.float64]
    f_F_2W: npt.NDArray[np.float64]

    @property
    def p_WF(self) -> npt.NDArray[np.float64]:  # (num_steps, 2)
        return self.p_WB + self.p_BF_W


class FootstepTrajectory:
    """
    state = [p_WB; theta_WB]
    input = [p_BF_W; f_F_1W; f_F_2W]

    Assuming linear state movement between knot points, and constant inputs.

    Note: If provided, the last knot point of the inputs will not be used.
    """

    def __init__(self, knot_points: FootstepPlanKnotPoints, dt: float) -> None:
        self.knot_points = knot_points
        self.dt = dt

    @property
    def end_time(self) -> float:
        return self.num_steps * self.dt

    @property
    def num_steps(self) -> int:
        return self.knot_points.p_WB.shape[0]

    @classmethod
    def from_knot_points(
        cls, knot_points: List[KnotPointValue], dt: float
    ) -> "FootstepTrajectory":
        p_WBs = np.array([k.p_WB for k in knot_points])
        theta_WBs = np.array([k.theta_WB for k in knot_points])
        p_BF_Ws = np.array([k.p_BF_W for k in knot_points])
        f_F_1Ws = np.array([k.f_F_1W for k in knot_points])
        f_F_2Ws = np.array([k.f_F_2W for k in knot_points])

        merged_knot_points = FootstepPlanKnotPoints(
            p_WBs, theta_WBs, p_BF_Ws, f_F_1Ws, f_F_2Ws
        )
        return cls(merged_knot_points, dt)

    @classmethod
    def from_segments(
        cls, segments: List[FootstepPlanKnotPoints], dt: float
    ) -> "FootstepTrajectory":
        p_WBs = np.vstack([k.p_WB for k in segments])
        theta_WBs = np.array([k.theta_WB for k in segments]).reshape((-1, 1))
        p_BF_Ws = np.vstack([k.p_BF_W for k in segments])
        f_F_1Ws = np.vstack([k.f_F_1W for k in segments])
        f_F_2Ws = np.vstack([k.f_F_2W for k in segments])

        merged_knot_points = FootstepPlanKnotPoints(
            p_WBs, theta_WBs, p_BF_Ws, f_F_1Ws, f_F_2Ws
        )

        return cls(merged_knot_points, dt)


@dataclass
class KnotPoint:
    def __init__(
        self,
        stone: InPlaneSteppingStone,
        robot: PotatoRobot,
        name: Optional[str] = None,
    ) -> None:
        # Assume we always only have one foot in contact
        if name is not None:
            self.name = f"{stone.name}_{name}"

        self.prog = MathematicalProgram()

        # declare states
        self.p_WB = self.prog.NewContinuousVariables(2, "p_WB")
        self.v_WB = self.prog.NewContinuousVariables(2, "v_WB")
        self.theta_WB = self.prog.NewContinuousVariables(1, "theta_WB")[0]
        self.omega_WB = self.prog.NewContinuousVariables(1, "omega_WB")[0]

        # declare inputs
        self.p_BF_W = self.prog.NewContinuousVariables(2, "p_BF_W")
        self.f_F_1W = self.prog.NewContinuousVariables(2, "f_F_1W")
        self.f_F_2W = self.prog.NewContinuousVariables(2, "f_F_2W")

        # compute the foot position
        self.p_WF = self.p_WB + self.p_BF_W

        # auxilliary vars
        # TODO(bernhardpg): we might be able to get around this once we
        # have SDP constraints over the edges
        self.tau_F_1 = self.prog.NewContinuousVariables(1, "tau_F_1")[0]
        self.tau_F_2 = self.prog.NewContinuousVariables(1, "tau_F_2")[0]

        # linear acceleration
        self.a_WB = (1 / robot.mass) * (self.f_F_1W + self.f_F_2W)

        # angular acceleration
        self.theta_ddot = (1 / robot.inertia) * (self.tau_F_1 + self.tau_F_2)

        # torque = arm x force
        self.p_BF_1W = self.p_BF_W + np.array([robot.foot_length / 2, 0])
        self.p_BF_2W = self.p_BF_W - np.array([robot.foot_length / 2, 0])

        self.prog.AddConstraint(self.tau_F_1 == cross_2d(self.p_BF_1W, self.f_F_1W))
        self.prog.AddConstraint(self.tau_F_2 == cross_2d(self.p_BF_2W, self.f_F_2W))

        # Stay on the stepping stone
        self.prog.AddConstraint(stone.x_min <= self.p_WF[0])
        self.prog.AddConstraint(self.p_WF[0] <= stone.x_max)
        self.prog.AddConstraint(self.p_WF[1] == stone.z_pos)

        # Don't move the feet too far from the robot
        MAX_DIST = 0.4
        self.prog.AddConstraint(self.p_WB[0] - self.p_WF[0] <= MAX_DIST)
        self.prog.AddConstraint(self.p_WB[0] - self.p_WF[0] <= MAX_DIST)

        # TODO(bernhardpg): Friction cone must be formulated differently
        # when we have tilted ground
        mu = 0.5
        for f in (self.f_F_1W, self.f_F_2W):
            self.prog.AddLinearConstraint(f[1] >= 0)
            self.prog.AddLinearConstraint(f[0] <= mu * f[1])
            self.prog.AddLinearConstraint(f[0] >= -mu * f[1])

        # TODO(bernhardpg): Step span limit

        # TODO(bernhardpg): Costs

    def get_state(self) -> npt.NDArray:
        return np.concatenate([self.p_WB, [self.theta_WB], self.v_WB, [self.omega_WB]])

    def get_input(self) -> npt.NDArray:
        return np.concatenate([self.f_F_1W, self.f_F_2W, self.p_BF_W])

    def get_robot_pose(self) -> npt.NDArray:
        return np.concatenate([self.p_WB, [self.theta_WB]])

    def get_vars(self) -> npt.NDArray:
        return np.concatenate(
            (self.get_state(), self.get_input(), (self.tau_F_1, self.tau_F_2))
        )

    def get_var_in_vertex(
        self,
        var: Variable,
        vertex_vars: npt.NDArray,
    ) -> float | Variable:
        idx = self.prog.FindDecisionVariableIndex(var)
        return vertex_vars[idx]  # type: ignore

    def get_vars_in_vertex(
        self, vars: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        idxs = self.prog.FindDecisionVariableIndices(vars)
        return vertex_vars[idxs]

    def get_dynamics(self) -> npt.NDArray:
        return np.concatenate(
            [self.v_WB, [self.omega_WB], self.a_WB, [self.theta_ddot]]
        )

    def get_lhs_in_vertex_vars(self, vertex_vars: npt.NDArray) -> NpVariableArray:
        """
        Gets the left-hand-side of the forward euler integration in the variables of the provided vertex
        s_next = s_curr + dt * f(s_curr, u_curr)
        """
        s_next = self.get_state()
        idxs = self.prog.FindDecisionVariableIndices(s_next)
        return vertex_vars[idxs]

    def get_rhs_in_vertex_vars(
        self, vertex_vars: npt.NDArray, dt: float
    ) -> NpExpressionArray:
        """
        Gets the right-hand-side of the forward euler integration in the variables of the provided vertex
        s_next = s_curr + dt * f(s_curr, u_curr)
        """
        s_curr = self.get_state()
        dynamics = (
            self.get_dynamics()
        )  # will be a np.array of variables and expressions
        s_next = s_curr + dt * dynamics
        vars = self.get_vars()

        # note: the dynamics are always linear (we introduced some aux vars to achieve this)
        A, b = DecomposeAffineExpressions(s_next, vars)
        idxs = self.prog.FindDecisionVariableIndices(vars)
        x = vertex_vars[idxs]

        rhs = A @ x + b
        return rhs

    def get_convex_set(self) -> Spectrahedron:
        relaxed_prog = MakeSemidefiniteRelaxation(self.prog)
        spectrahedron = Spectrahedron(relaxed_prog)
        return spectrahedron

    def get_knot_point_val_from_vertex(
        self, vertex_vars_vals: npt.NDArray[np.float64]
    ) -> KnotPointValue:
        p_WB = self.get_vars_in_vertex(self.p_WB, vertex_vars_vals)
        theta_WB = self.get_var_in_vertex(self.theta_WB, vertex_vars_vals)
        p_BF_W = self.get_vars_in_vertex(self.p_BF_W, vertex_vars_vals)
        f_F_1W = self.get_vars_in_vertex(self.f_F_1W, vertex_vars_vals)
        f_F_2W = self.get_vars_in_vertex(self.f_F_2W, vertex_vars_vals)

        return KnotPointValue(p_WB, theta_WB, p_BF_W, f_F_1W, f_F_2W)


@dataclass
class FootstepPlanSegment:
    def __init__(
        self,
        stone: InPlaneSteppingStone,
        one_or_two_feet: Literal["one_foot", "two_feet"],
        robot: PotatoRobot,
        config: FootstepPlanningConfig,
        name: Optional[str] = None,
    ) -> None:
        # Assume we always only have one foot in contact
        if name is not None:
            self.name = f"{stone.name}_{name}"

        self.config = config
        num_steps = self.config.period_steps

        self.prog = MathematicalProgram()

        # declare states
        self.p_WB = self.prog.NewContinuousVariables(num_steps, 2, "p_WB")
        self.v_WB = self.prog.NewContinuousVariables(num_steps, 2, "v_WB")
        self.theta_WB = self.prog.NewContinuousVariables(num_steps, "theta_WB")
        self.omega_WB = self.prog.NewContinuousVariables(num_steps, "omega_WB")

        # declare inputs
        self.p_BF_W = self.prog.NewContinuousVariables(num_steps, 2, "p_BF_W")
        self.f_F_1W = self.prog.NewContinuousVariables(num_steps, 2, "f_F_1W")
        self.f_F_2W = self.prog.NewContinuousVariables(num_steps, 2, "f_F_2W")

        # compute the foot position
        self.p_WF = self.p_WB + self.p_BF_W

        # auxilliary vars
        # TODO(bernhardpg): we might be able to get around this once we
        # have SDP constraints over the edges
        self.tau_F_1 = self.prog.NewContinuousVariables(num_steps, "tau_F_1")
        self.tau_F_2 = self.prog.NewContinuousVariables(num_steps, "tau_F_2")

        # linear acceleration
        g = np.array([0, -9.81])
        self.a_WB = (1 / robot.mass) * (self.f_F_1W + self.f_F_2W) + g

        # TODO: remove
        # enforce no z-acceleration at first and last step
        self.prog.AddLinearEqualityConstraint(self.a_WB[0][1] == 0)
        self.prog.AddLinearEqualityConstraint(self.a_WB[num_steps - 1][1] == 0)

        # angular acceleration
        self.theta_ddot = (1 / robot.inertia) * (self.tau_F_1 + self.tau_F_2)

        # contact points positions
        self.p_BF_1W = self.p_BF_W + np.array([robot.foot_length / 2, 0])
        self.p_BF_2W = self.p_BF_W - np.array([robot.foot_length / 2, 0])

        for k in range(num_steps):
            # torque = arm x force
            self.prog.AddQuadraticConstraint(
                self.tau_F_1[k] - cross_2d(self.p_BF_1W[k], self.f_F_1W[k]), 0, 0
            )
            self.prog.AddQuadraticConstraint(
                self.tau_F_2[k] - cross_2d(self.p_BF_2W[k], self.f_F_2W[k]), 0, 0
            )

            # Stay on the stepping stone
            self.prog.AddLinearConstraint(stone.x_min <= self.p_WF[k][0])
            self.prog.AddLinearConstraint(self.p_WF[k][0] <= stone.x_max)
            self.prog.AddLinearEqualityConstraint(self.p_WF[k][1] == stone.z_pos)

            # Don't move the feet too far from the robot
            MAX_DIST = 0.4
            self.prog.AddLinearConstraint(self.p_WB[k][0] - self.p_WF[k][0] <= MAX_DIST)
            self.prog.AddLinearConstraint(
                self.p_WB[k][0] - self.p_WF[k][0] >= -MAX_DIST
            )

            # TODO(bernhardpg): Friction cone must be formulated differently
            # when we have tilted ground
            mu = 0.5  # TODO: move friction coeff
            for f in (self.f_F_1W, self.f_F_2W):
                self.prog.AddLinearConstraint(f[k][1] >= 0)
                self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])

        # dynamics
        dt = self.config.dt
        for k in range(num_steps - 1):
            s_next = self.get_state(k + 1)
            s_curr = self.get_state(k)
            f = self.get_dynamics(k)
            # forward euler
            self.prog.AddLinearConstraint(eq(s_next, s_curr + dt * f))

            # foot can't move during segment
            const = eq(self.p_WF[k], self.p_WF[k + 1])
            for c in const:
                self.prog.AddLinearEqualityConstraint(c)

        # TODO(bernhardpg): Step span limit

        self.costs = {"sq_forces": [], "sq_lin_vel": [], "sq_rot_vel": []}

        cost_force = 1e-5
        cost_lin_vel = 10
        cost_ang_vel = 0.1

        # squared forces
        for k in range(num_steps):
            f1 = self.f_F_1W[k]
            f2 = self.f_F_2W[k]
            sq_forces = f1.T @ f1 + f2.T @ f2
            c = self.prog.AddQuadraticCost(cost_force * sq_forces)
            self.costs["sq_forces"].append(c)

        # squared robot velocity
        for k in range(num_steps):
            v = self.v_WB[k]
            sq_lin_vel = v.T @ v
            c = self.prog.AddQuadraticCost(cost_lin_vel * sq_lin_vel)
            self.costs["sq_lin_vel"].append(c)

            sq_rot_vel = self.omega_WB[k] ** 2
            c = self.prog.AddQuadraticCost(cost_ang_vel * sq_rot_vel)
            self.costs["sq_rot_vel"].append(c)

    def get_state(self, k: int) -> npt.NDArray:
        return np.concatenate(
            [self.p_WB[k], [self.theta_WB[k]], self.v_WB[k], [self.omega_WB[k]]]
        )

    def get_input(self, k: int) -> npt.NDArray:
        return np.concatenate([self.f_F_1W[k], self.f_F_2W[k], self.p_BF_W[k]])

    def get_robot_pose(self, k: int) -> npt.NDArray:
        return np.concatenate([self.p_WB[k], [self.theta_WB[k]]])

    def get_vars(self, k: int) -> npt.NDArray:
        return np.concatenate(
            (self.get_state(k), self.get_input(k), (self.tau_F_1[k], self.tau_F_2[k]))
        )

    def get_dynamics(self, k: int) -> npt.NDArray:
        return np.concatenate(
            [self.v_WB[k], [self.omega_WB[k]], self.a_WB[k], [self.theta_ddot[k]]]
        )

    def add_pose_constraint(
        self, k: int, p_WB: npt.NDArray[np.float64], theta_WB: float
    ) -> None:
        self.prog.AddLinearConstraint(eq(self.p_WB[k], p_WB))
        self.prog.AddLinearConstraint(self.theta_WB[k] == theta_WB)

    def add_spatial_vel_constraint(
        self, k: int, v_WB: npt.NDArray[np.float64], omega_WB: float
    ) -> None:
        self.prog.AddLinearConstraint(eq(self.v_WB[k], v_WB))
        self.prog.AddLinearConstraint(self.omega_WB[k] == omega_WB)

    def get_var_in_vertex(
        self,
        var: Variable,
        vertex_vars: npt.NDArray,
    ) -> float | Variable:
        idx = self.prog.FindDecisionVariableIndex(var)
        return vertex_vars[idx]  # type: ignore

    def get_vars_in_vertex(
        self, vars: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        idxs = self.prog.FindDecisionVariableIndices(vars)
        return vertex_vars[idxs]

    def get_convex_set(self) -> Spectrahedron:
        relaxed_prog = MakeSemidefiniteRelaxation(self.prog)
        spectrahedron = Spectrahedron(relaxed_prog)
        return spectrahedron

    def evaluate_with_result(
        self, result: MathematicalProgramResult
    ) -> FootstepPlanKnotPoints:
        p_WB = result.GetSolution(self.p_WB)
        theta_WB = result.GetSolution(self.theta_WB)
        p_BF_W = result.GetSolution(self.p_BF_W)
        f_F_1W = result.GetSolution(self.f_F_1W)
        f_F_2W = result.GetSolution(self.f_F_2W)

        return FootstepPlanKnotPoints(p_WB, theta_WB, p_BF_W, f_F_1W, f_F_2W)

    def evaluate_costs_with_result(
        self, result: MathematicalProgramResult
    ) -> Dict[str, List[float]]:
        cost_vals = {}
        for key, val in self.costs.items():
            cost_vals[key] = []

            for binding in val:
                vars = result.GetSolution(binding.variables())
                cost_vals[key].append(binding.evaluator().Eval(vars))

        for key in cost_vals:
            cost_vals[key] = np.array(cost_vals[key])

        return cost_vals


class VertexPointPair(NamedTuple):
    v: GcsVertex
    p: KnotPoint

    def get_vars_in_vertex(self, vars: npt.NDArray) -> npt.NDArray:
        return self.p.get_vars_in_vertex(vars, self.v.x())

    def get_knot_point_val(self, result: MathematicalProgramResult) -> KnotPointValue:
        return self.p.get_knot_point_val_from_vertex(result.GetSolution(self.v.x()))


class FootstepPlanner:
    def __init__(
        self,
        config: FootstepPlanningConfig,
        terrain: InPlaneTerrain,
        initial_position: npt.NDArray[np.float64],
        target_position: npt.NDArray[np.float64],
    ) -> None:
        self.config = config

        initial_stone = terrain.stepping_stones[0]
        target_stone = terrain.stepping_stones[1]

        robot = config.robot
        dt = config.dt

        point_1 = KnotPoint(initial_stone, robot, name="1")
        point_2 = KnotPoint(initial_stone, robot, name="2")
        point_3 = KnotPoint(target_stone, robot, name="3")
        point_4 = KnotPoint(target_stone, robot, name="4")

        points = [point_1, point_2, point_3, point_4]

        self.gcs = GraphOfConvexSets()

        # Add initial and target vertices
        self.source = self.gcs.AddVertex(Point(initial_position), name="source")
        self.target = self.gcs.AddVertex(Point(target_position), name="target")

        # Add all knot points as vertices
        pairs = self._add_points_as_vertices(self.gcs, points)

        edges_to_add = [(0, 1), (1, 2), (1, 2), (2, 3)]

        self._add_edges_with_dynamics_constraints(self.gcs, edges_to_add, pairs, dt)

        # TODO: Continuity constraints on subsequent contacts within the same region

        self._add_edge_to_source_or_target(pairs[0], "source")

        for pair in pairs:  # connect all the vertices to the target
            self._add_edge_to_source_or_target(pair, "target")

        self.vertex_name_to_pairs = {pair.v.name(): pair for pair in pairs}

    def _add_edge_to_source_or_target(
        self,
        pair: VertexPointPair,
        source_or_target: Literal["source", "target"] = "source",
    ) -> None:
        if source_or_target == "source":
            s = self.source
            # source -> v
            e = self.gcs.AddEdge(s, pair.v)
        else:  # target
            s = self.target
            # v -> target
            e = self.gcs.AddEdge(pair.v, s)

        pose = pair.get_vars_in_vertex(pair.p.get_robot_pose())
        # The only variables in the source/target are the pose variables
        constraint = eq(pose, s.x())
        for c in constraint:
            e.AddConstraint(c)

    def _add_points_as_vertices(
        self, gcs: GraphOfConvexSets, points: List[KnotPoint]
    ) -> List[VertexPointPair]:
        vertices = [gcs.AddVertex(p.get_convex_set(), name=p.name) for p in points]
        pairs = [VertexPointPair(v, p) for v, p in zip(vertices, points)]
        return pairs

    def _add_edges_with_dynamics_constraints(
        self,
        gcs: GraphOfConvexSets,
        edges_to_add: List[Tuple[int, int]],
        pairs: List[VertexPointPair],
        dt: float,
    ) -> None:
        for i, j in edges_to_add:
            u, p_u = pairs[i]
            v, p_v = pairs[j]

            e = gcs.AddEdge(u, v)
            constraint = eq(
                p_u.get_lhs_in_vertex_vars(u.x()),
                p_v.get_rhs_in_vertex_vars(v.x(), dt),
            )
            for c in constraint:
                e.AddConstraint(c)

    def create_graph_diagram(
        self,
        filename: Optional[str] = None,
        result: Optional[MathematicalProgramResult] = None,
    ) -> pydot.Dot:
        """
        Optionally saves the graph to file if a string is given for the 'filepath' argument.
        """
        graphviz = self.gcs.GetGraphvizString(
            precision=2, result=result, show_slacks=False
        )

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        if filename is not None:
            data.write_png(filename + ".png")

        return data

    def plan(self) -> FootstepTrajectory:
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 20
        result = self.gcs.SolveShortestPath(self.source, self.target, options)

        if not result.is_success():
            raise RuntimeError("Could not find a solution!")

        edges_on_sol = self.gcs.GetSolutionPath(self.source, self.target, result)
        names_on_sol = [e.name() for e in edges_on_sol]
        print(f"Path: {' -> '.join(names_on_sol)}")

        # we disregard source and target vertices when we extract the path
        pairs_on_sol = [
            self.vertex_name_to_pairs[e.v().name()] for e in edges_on_sol[:-1]
        ]

        knot_point_vals = [p.get_knot_point_val(result) for p in pairs_on_sol]
        plan = FootstepTrajectory.from_knot_points(knot_point_vals, self.config.dt)

        return plan


def animate_footstep_plan(
    robot: PotatoRobot,
    terrain: InPlaneTerrain,
    plan: FootstepTrajectory,
    title=None,
) -> None:
    # Initialize figure for animation
    fig, ax = plt.subplots()

    # Plot stepping stones
    terrain.plot(title=title, ax=ax, max_height=2.5)

    # Plot robot
    robot_body = Ellipse(
        xy=(0, 0),
        width=robot.width,
        height=robot.height,
        angle=0,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(robot_body)

    # Foot
    base_foot_vertices = np.array(
        [
            [-robot.foot_length / 2, 0],
            [robot.foot_length / 2, 0],
            [0, robot.foot_height],
        ]
    )
    foot = Polygon(base_foot_vertices, closed=True, fill=None, edgecolor="black")
    ax.add_patch(foot)

    # Forces
    FORCE_SCALE = 1e-3
    force_1 = FancyArrowPatch(
        posA=(0, 0),
        posB=(1 * FORCE_SCALE, 1 * FORCE_SCALE),
        arrowstyle="->",
        color="green",
    )
    ax.add_patch(force_1)
    force_2 = FancyArrowPatch(
        posA=(0, 0),
        posB=(1 * FORCE_SCALE, 1 * FORCE_SCALE),
        arrowstyle="->",
        color="green",
    )
    ax.add_patch(force_2)

    # Initial position of the feet
    p_WB = ax.scatter(0, 0, color="r", zorder=3, label="CoM")
    p_WF = ax.scatter(0, 0, color="b", zorder=3, label="Left foot")

    # Misc settings
    plt.close()
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2)

    def animate(n_steps: int) -> None:
        # Robot
        p_WB.set_offsets(plan.knot_points.p_WB[n_steps])
        robot_body.set_center(plan.knot_points.p_WB[n_steps])
        robot_body.angle = plan.knot_points.theta_WB[n_steps]

        # Foot
        foot.set_xy(base_foot_vertices + plan.knot_points.p_WF[n_steps])

        # Scatter feet
        p_WF.set_offsets(plan.knot_points.p_WF[n_steps])

        # Forces
        f1_pos = plan.knot_points.p_WF[n_steps] + base_foot_vertices[0]
        f1_val = plan.knot_points.f_F_1W[n_steps] * FORCE_SCALE
        force_1.set_positions(posA=f1_pos, posB=(f1_pos + f1_val))

        f2_pos = plan.knot_points.p_WF[n_steps] + base_foot_vertices[1]
        f2_val = plan.knot_points.f_F_2W[n_steps] * FORCE_SCALE
        force_2.set_positions(posA=f2_pos, posB=(f2_pos + f2_val))

    # Create and display animation
    n_steps = plan.num_steps
    ani = FuncAnimation(fig, animate, frames=n_steps, interval=1e3)  # type: ignore
    ani.save("footstep_plan.mp4", writer="ffmpeg")


def test_single_point():
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.5, width=1.0, z_pos=0.2, name="initial")
    target_stone = terrain.add_stone(x_pos=1.5, width=1.0, z_pos=0.3, name="target")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    desired_robot_pos = np.array([0, cfg.robot.desired_com_height])

    initial_pose = np.concatenate((initial_stone.center + desired_robot_pos, [0]))
    target_pose = np.concatenate((target_stone.center + desired_robot_pos, [0]))
    planner = FootstepPlanner(cfg, terrain, initial_pose, target_pose)

    planner.create_graph_diagram("footstep_planner")
    plan = planner.plan()

    animate_footstep_plan(robot, terrain, plan)

    # terrain.plot()
    # plt.show()


def test_trajectory_segment() -> None:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=0.5, width=1.5, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot)

    segment = FootstepPlanSegment(stone, "one_foot", robot, cfg, name="First step")

    assert segment.p_WB.shape == (cfg.period_steps, 2)
    assert segment.v_WB.shape == (cfg.period_steps, 2)
    assert segment.theta_WB.shape == (cfg.period_steps,)
    assert segment.omega_WB.shape == (cfg.period_steps,)

    assert segment.p_BF_W.shape == (cfg.period_steps, 2)
    assert segment.f_F_1W.shape == (cfg.period_steps, 2)
    assert segment.f_F_2W.shape == (cfg.period_steps, 2)

    assert segment.tau_F_1.shape == (cfg.period_steps,)
    assert segment.tau_F_2.shape == (cfg.period_steps,)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    initial_pos = np.array([stone.x_pos - 0.1, 0.0]) + desired_robot_pos
    target_pos = np.array([stone.x_pos + 0.1, 0.0]) + desired_robot_pos

    segment.add_pose_constraint(0, initial_pos, 0)  # type: ignore
    segment.add_pose_constraint(cfg.period_steps - 1, target_pos, 0)  # type: ignore

    segment.add_spatial_vel_constraint(0, np.zeros((2,)), 0)
    segment.add_spatial_vel_constraint(cfg.period_steps - 1, np.zeros((2,)), 0)

    debug = True
    solver_options = SolverOptions()
    if debug:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    result = Solve(segment.prog, solver_options=solver_options)
    assert result.is_success()

    segment_value = segment.evaluate_with_result(result)

    traj = FootstepTrajectory.from_segments([segment_value], cfg.dt)
    assert traj.knot_points.p_WB.shape == (cfg.period_steps, 2)
    assert traj.knot_points.theta_WB.shape == (cfg.period_steps, 1)
    assert traj.knot_points.p_BF_W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F_1W.shape == (cfg.period_steps, 2)
    assert traj.knot_points.f_F_2W.shape == (cfg.period_steps, 2)

    a_WB = evaluate_np_expressions_array(segment.a_WB, result)

    cost_vals = segment.evaluate_costs_with_result(result)
    breakpoint()

    animate_footstep_plan(robot, terrain, traj)


def main():
    # test_single_point()
    test_trajectory_segment()


if __name__ == "__main__":
    main()
