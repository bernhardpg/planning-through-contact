from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    HPolyhedron,
    Point,
    Spectrahedron,
)
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
)
from pydrake.symbolic import DecomposeAffineExpressions
from underactuated.exercises.humanoids.footstep_planning_gcs_utils import plot_rectangle

from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray


@dataclass
class InPlaneSteppingStone:
    x_pos: float
    z_pos: float
    width: float
    name: Optional[str] = None

    @property
    def height(self) -> float:
        return self.z_pos

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

    def plot(self, title: Optional[str] = None, **kwargs):
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
        ax.set_ylim((0.0, self.max_height * 2.0))


@dataclass
class Robot:
    # TODO(bernhardpg): set reasonable inertia properties
    # This is from a sphere with mass = 50 kg and all axis = 0.5 meter
    mass: float = 50.0  # kg
    inertia: float = 5.0  # kg m**2
    foot_length: float = 0.3  # m
    step_span: float = 0.8  # m


@dataclass
class KnotPoint:
    def __init__(
        self, stone: InPlaneSteppingStone, robot: Robot, name: Optional[str] = None
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

        # TODO(bernhardpg): Friction cone must be formulated differently
        # when we have tilted ground
        mu = 0.5
        for f in (self.f_F_1W, self.f_F_2W):
            self.prog.AddLinearConstraint(f[1] >= 0)
            self.prog.AddLinearConstraint(f[0] <= mu * f[1])
            self.prog.AddLinearConstraint(f[0] >= -mu * f[1])

        # TODO(bernhardpg): Step span limit

    def get_state(self) -> npt.NDArray:
        return np.concatenate([self.p_WB, [self.theta_WB], self.v_WB, [self.omega_WB]])

    def get_input(self) -> npt.NDArray:
        return np.concatenate([self.f_F_1W, self.f_F_2W, self.p_BF_W])

    def get_vars(self) -> npt.NDArray:
        return np.concatenate(
            (self.get_state(), self.get_input(), (self.tau_F_1, self.tau_F_2))
        )

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


def main():
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.5, width=1.0, z_pos=0.2, name="initial")
    target_stone = terrain.add_stone(x_pos=1.5, width=1.0, z_pos=0.3, name="target")

    dt = 0.3

    robot = Robot()

    point_1 = KnotPoint(initial_stone, robot, name="1")
    point_2 = KnotPoint(initial_stone, robot, name="2")

    points = [point_1, point_2]

    gcs = GraphOfConvexSets()

    vertices = [gcs.AddVertex(p.get_convex_set(), name=p.name) for p in points]
    edges_to_add = [(0, 1)]

    for i, j in edges_to_add:
        u, v = vertices[i], vertices[j]
        p_u, p_v = points[i], points[j]

        e = gcs.AddEdge(u, v)

        constraint = eq(
            p_u.get_lhs_in_vertex_vars(u.x()),
            p_v.get_rhs_in_vertex_vars(v.x(), dt),
            # p_v.get_state() + dt * point_1.get_dynamics(),
        )

        for c in constraint:
            e.AddConstraint(c)

    # terrain.plot()
    # plt.show()


if __name__ == "__main__":
    main()
