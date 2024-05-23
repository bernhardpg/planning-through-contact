import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import GraphOfConvexSets, Spectrahedron
from pydrake.math import eq
from pydrake.solvers import (
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    PositiveSemidefiniteConstraint,
    SnoptSolver,
)
from pydrake.symbolic import DecomposeAffineExpressions, Expression, Variable, Variables

from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.in_plane_terrain import (
    InPlaneSteppingStone,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array

GcsVertex = GraphOfConvexSets.Vertex
GcsEdge = GraphOfConvexSets.Edge


# TODO move this to a utils file
def get_X_from_semidefinite_relaxation(relaxation: MathematicalProgram):
    assert len(relaxation.positive_semidefinite_constraints()) == 1
    X = relaxation.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


def get_X_from_psd_constraint(binding) -> npt.NDArray:
    assert type(binding.evaluator()) == PositiveSemidefiniteConstraint
    X = binding.variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


@dataclass
class FootstepPlanKnotPoints:
    p_WB: npt.NDArray[np.float64]
    theta_WB: npt.NDArray[np.float64]
    p_WF1: npt.NDArray[np.float64]
    f_F1_1W: npt.NDArray[np.float64]
    f_F1_2W: npt.NDArray[np.float64]
    p_WF2: Optional[npt.NDArray[np.float64]] = None
    f_F2_1W: Optional[npt.NDArray[np.float64]] = None
    f_F2_2W: Optional[npt.NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        assert self.p_WB.shape == (self.num_points, 2)
        assert self.theta_WB.shape == (self.num_points,)

        assert self.p_WF1.shape == (self.num_points, 2)
        assert self.f_F1_1W.shape == (self.num_points, 2)
        assert self.f_F1_2W.shape == (self.num_points, 2)

        if self.p_WF2 is not None:
            assert self.p_WF2.shape == (self.num_points, 2)
        if self.f_F2_1W is not None:
            assert self.f_F2_1W.shape == (self.num_points, 2)
        if self.f_F2_2W is not None:
            assert self.f_F2_2W.shape == (self.num_points, 2)

        if self.both_feet:
            assert self.f_F2_1W is not None
            assert self.f_F2_2W is not None

    @property
    def both_feet(self) -> bool:
        return self.p_WF2 is not None

    @property
    def num_points(self) -> int:
        return self.p_WB.shape[0]


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

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            # NOTE: We save the config and path knot points, not this object, as some Drake objects are not serializable
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str) -> "FootstepTrajectory":
        with open(Path(filename), "rb") as file:
            traj = pickle.load(file)
            return traj

    @property
    def end_time(self) -> float:
        return self.num_steps * self.dt

    @property
    def num_steps(self) -> int:
        return self.knot_points.p_WB.shape[0]

    @classmethod
    def from_segments(
        cls,
        segments: List[FootstepPlanKnotPoints],
        dt: float,
    ) -> "FootstepTrajectory":
        both_feet = np.vstack([s.both_feet for s in segments])
        # This is a quick way to check that the bool value changes for each element in the array
        modes_are_alternating = not np.any(both_feet[:-1] & both_feet[1:])
        if not modes_are_alternating:
            raise RuntimeError(
                "The provided segments do not have alternating modes and do not form a coherent footstep plan."
            )

        # NOTE: Here we just pick that we start with the left foot. Could just as well have picked the other foot
        gait_pattern = np.array([[1, 1], [1, 0], [1, 1], [0, 1]])
        gait_schedule = []
        if segments[0].both_feet:
            start_idx = 0
        else:
            start_idx = 1

        gait_idx = start_idx
        for _ in segments:
            gait_schedule.append(gait_pattern[gait_idx % len(gait_pattern)])
            gait_idx += 1

        gait_schedule = np.array(gait_schedule)

        p_WBs = np.vstack([k.p_WB for k in segments])
        theta_WBs = np.vstack([k.theta_WB for k in segments]).flatten()

        p_WF1s = []
        f_F1_1Ws = []
        f_F1_2Ws = []
        p_WF2s = []
        f_F2_1Ws = []
        f_F2_2Ws = []

        # NOTE: This assumes that all the segments have the same lengths!
        empty_shape = segments[0].p_WB.shape

        for segment, (first_active, last_active) in zip(segments, gait_schedule):
            both_active = first_active and last_active
            if both_active:
                p_WF1s.append(segment.p_WF1)
                f_F1_1Ws.append(segment.f_F1_1W)
                f_F1_2Ws.append(segment.f_F1_2W)

                p_WF2s.append(segment.p_WF2)
                f_F2_1Ws.append(segment.f_F2_1W)
                f_F2_2Ws.append(segment.f_F2_2W)
            else:
                # NOTE: These next lines look like they have a typo, but they don't.
                # When there is only one foot active, the values for this foot is
                # always stored in the "first" foot values (to avoid unecessary optimization
                # variables)
                if first_active:
                    p_WF1s.append(segment.p_WF1)
                    f_F1_1Ws.append(segment.f_F1_1W)
                    f_F1_2Ws.append(segment.f_F1_2W)

                    p_WF2s.append(np.full(empty_shape, np.nan))
                    f_F2_1Ws.append(np.full(empty_shape, np.nan))
                    f_F2_2Ws.append(np.full(empty_shape, np.nan))
                else:  # right_active
                    p_WF1s.append(np.full(empty_shape, np.nan))
                    f_F1_1Ws.append(np.full(empty_shape, np.nan))
                    f_F1_2Ws.append(np.full(empty_shape, np.nan))

                    # Notice that here we pick from the "first" values
                    p_WF2s.append(segment.p_WF1)
                    f_F2_1Ws.append(segment.f_F1_1W)
                    f_F2_2Ws.append(segment.f_F1_2W)

        p_WF1s = np.vstack(p_WF1s)
        f_F1_1Ws = np.vstack(f_F1_1Ws)
        f_F1_2Ws = np.vstack(f_F1_2Ws)

        p_WF2s = np.vstack(p_WF2s)
        f_F2_1Ws = np.vstack(f_F2_1Ws)
        f_F2_2Ws = np.vstack(f_F2_2Ws)

        merged_knot_points = FootstepPlanKnotPoints(
            p_WBs, theta_WBs, p_WF1s, f_F1_1Ws, f_F1_2Ws, p_WF2s, f_F2_1Ws, f_F2_2Ws
        )

        return cls(merged_knot_points, dt)


@dataclass
class FootstepPlanSegment:
    def __init__(
        self,
        stone: InPlaneSteppingStone,
        one_or_two_feet: Literal["one_foot", "two_feet"],
        robot: PotatoRobot,
        config: FootstepPlanningConfig,
        name: Optional[str] = None,
        stone_for_last_foot: Optional[InPlaneSteppingStone] = None,
    ) -> None:
        """
        A wrapper class for constructing a nonlinear optimization program for the
        motion within a specified mode.

        @param stones_per_foot: If passed, each foot is restriced to be in contact with their
        respective stone, in the order (L,R). If passed, stone is disregarded.
        """
        if stone_for_last_foot:
            stone_first, stone_last = stone, stone_for_last_foot
            self.name = f"{stone_first.name}_and_{stone_last.name}"
        else:
            self.name = f"{stone.name}"

        if name is not None:
            self.name = f"{self.name}_{name}"

        self.config = config
        self.two_feet = one_or_two_feet == "two_feet"
        if stone_for_last_foot:
            self.stone_first, self.stone_last = stone, stone_for_last_foot
        else:
            self.stone_first = self.stone_last = stone

        self.num_steps = self.config.period_steps

        self.prog = MathematicalProgram()

        # declare states
        self.p_WB = self.prog.NewContinuousVariables(self.num_steps, 2, "p_WB")
        self.v_WB = self.prog.NewContinuousVariables(self.num_steps, 2, "v_WB")
        self.theta_WB = self.prog.NewContinuousVariables(self.num_steps, "theta_WB")
        self.omega_WB = self.prog.NewContinuousVariables(self.num_steps, "omega_WB")

        # declare inputs

        # first foot
        self.p_WF1_x = self.prog.NewContinuousVariables(self.num_steps, "p_WF1_x")
        self.f_F1_1W = self.prog.NewContinuousVariables(self.num_steps, 2, "f_F1_1W")
        self.f_F1_2W = self.prog.NewContinuousVariables(self.num_steps, 2, "f_F1_2W")
        if self.two_feet:
            # second foot
            self.p_WF2_x = self.prog.NewContinuousVariables(self.num_steps, "p_WF2_x")
            self.f_F2_1W = self.prog.NewContinuousVariables(
                self.num_steps, 2, "f_F2_1W"
            )
            self.f_F2_2W = self.prog.NewContinuousVariables(
                self.num_steps, 2, "f_F2_2W"
            )

        self.p_WF1 = np.vstack(
            [self.p_WF1_x, np.full(self.p_WF1_x.shape, self.stone_first.z_pos)]
        ).T  # (num_steps, 2)
        if self.two_feet:
            self.p_WF2 = np.vstack(
                [self.p_WF2_x, np.full(self.p_WF2_x.shape, self.stone_last.z_pos)]
            ).T  # (num_steps, 2)

        # compute the foot position
        self.p_BF1_W = self.p_WF1 - self.p_WB
        if self.two_feet:
            self.p_BF2_W = self.p_WF2 - self.p_WB

        # auxilliary vars
        # TODO(bernhardpg): we might be able to get around this once we
        # have SDP constraints over the edges
        self.tau_F1_1 = self.prog.NewContinuousVariables(self.num_steps, "tau_F1_1")
        self.tau_F1_2 = self.prog.NewContinuousVariables(self.num_steps, "tau_F1_2")
        if self.two_feet:
            self.tau_F2_1 = self.prog.NewContinuousVariables(self.num_steps, "tau_F2_1")
            self.tau_F2_2 = self.prog.NewContinuousVariables(self.num_steps, "tau_F2_2")

        # linear acceleration
        g = np.array([0, -9.81])
        self.a_WB = (1 / robot.mass) * (self.f_F1_1W + self.f_F1_2W) + g
        if self.two_feet:
            self.a_WB += (1 / robot.mass) * (self.f_F2_1W + self.f_F2_2W)

        # angular acceleration
        self.omega_dot_WB = (1 / robot.inertia) * (self.tau_F1_1 + self.tau_F1_2)
        if self.two_feet:
            self.omega_dot_WB += (1 / robot.inertia) * (self.tau_F2_1 + self.tau_F2_2)

        # contact points positions
        self.p_BF1_1W = self.p_BF1_W + np.array([robot.foot_length / 2, 0])
        self.p_BF1_2W = self.p_BF1_W - np.array([robot.foot_length / 2, 0])
        if self.two_feet:
            self.p_BF2_1W = self.p_BF2_W + np.array([robot.foot_length / 2, 0])
            self.p_BF2_2W = self.p_BF2_W - np.array([robot.foot_length / 2, 0])

        self.non_convex_constraints = []
        for k in range(self.num_steps):
            # torque = arm x force
            cs_for_knot_point = []
            c = self.prog.AddQuadraticConstraint(
                self.tau_F1_1[k] - cross_2d(self.p_BF1_1W[k], self.f_F1_1W[k]), 0, 0
            )
            cs_for_knot_point.append(c)
            c = self.prog.AddQuadraticConstraint(
                self.tau_F1_2[k] - cross_2d(self.p_BF1_2W[k], self.f_F1_2W[k]), 0, 0
            )
            cs_for_knot_point.append(c)
            if self.two_feet:
                c = self.prog.AddQuadraticConstraint(
                    self.tau_F2_1[k] - cross_2d(self.p_BF2_1W[k], self.f_F2_1W[k]), 0, 0
                )
                cs_for_knot_point.append(c)
                c = self.prog.AddQuadraticConstraint(
                    self.tau_F2_2[k] - cross_2d(self.p_BF2_2W[k], self.f_F2_2W[k]), 0, 0
                )
                cs_for_knot_point.append(c)

            self.non_convex_constraints.append(cs_for_knot_point)

            # Stay on the stepping stone
            self.prog.AddLinearConstraint(
                self.stone_first.x_min <= self.p_WF1[k][0] - robot.foot_length / 2
            )
            self.prog.AddLinearConstraint(
                self.p_WF1[k][0] + robot.foot_length / 2 <= self.stone_first.x_max
            )
            if self.two_feet:
                self.prog.AddLinearConstraint(
                    self.stone_last.x_min <= self.p_WF2[k][0] - robot.foot_length / 2
                )
                self.prog.AddLinearConstraint(
                    self.p_WF2[k][0] + robot.foot_length / 2 <= self.stone_last.x_max
                )

            # Don't move the feet too far from the robot
            self.prog.AddLinearConstraint(
                self.p_WB[k][0] - self.p_WF1[k][0] <= robot.max_step_dist_from_robot
            )
            self.prog.AddLinearConstraint(
                self.p_WB[k][0] - self.p_WF1[k][0] >= -robot.max_step_dist_from_robot
            )
            if self.two_feet:
                self.prog.AddLinearConstraint(
                    self.p_WB[k][0] - self.p_WF2[k][0] <= robot.max_step_dist_from_robot
                )
                self.prog.AddLinearConstraint(
                    self.p_WB[k][0] - self.p_WF2[k][0]
                    >= -robot.max_step_dist_from_robot
                )

            # constrain feet to not move too far from each other:
            if self.two_feet:
                first_last_foot_distance = self.p_WF1_x[k] - self.p_WF2_x[k]
                self.prog.AddLinearConstraint(
                    first_last_foot_distance <= robot.step_span
                )
                self.prog.AddLinearConstraint(
                    first_last_foot_distance >= -robot.step_span
                )

            # TODO(bernhardpg): Friction cone must be formulated differently
            # when we have tilted ground
            mu = 0.5  # TODO: move friction coeff
            for f in (self.f_F1_1W, self.f_F1_2W):
                self.prog.AddLinearConstraint(f[k][1] >= 0)
                self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])
            if self.two_feet:
                for f in (self.f_F2_1W, self.f_F2_2W):
                    self.prog.AddLinearConstraint(f[k][1] >= 0)
                    self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                    self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])

        # dynamics
        dt = self.config.dt
        for k in range(self.num_steps - 1):
            s_next = self.get_state(k + 1)
            s_curr = self.get_state(k)
            f = self.get_dynamics(k)
            # forward euler
            dynamics = s_next - (s_curr + dt * f)
            self.prog.AddLinearConstraint(eq(dynamics, 0))

            # foot can't move during segment
            const = eq(self.p_WF1[k], self.p_WF1[k + 1])
            for c in const:
                self.prog.AddLinearEqualityConstraint(c)
            if self.two_feet:
                const = eq(self.p_WF2[k], self.p_WF2[k + 1])
                for c in const:
                    self.prog.AddLinearEqualityConstraint(c)

        # TODO(bernhardpg): Step span limit

        self.costs = {
            "sq_forces": [],
            "sq_torques": [],
            "sq_acc_lin": [],
            "sq_acc_rot": [],
            "sq_lin_vel": [],
            "sq_rot_vel": [],
            "sq_nominal_pose": [],
        }

        cost_force = 1e-6
        cost_torque = 1.0
        cost_lin_vel = 10
        cost_ang_vel = 1.0
        cost_nominal_pose = 5

        # cost_force = 1e-5
        # cost_torque = 1e-3
        # cost_lin_vel = 10.0
        # cost_ang_vel = 0.1
        # cost_nominal_pose = 1.0

        # squared forces
        for k in range(self.num_steps):
            f1 = self.f_F1_1W[k]
            f2 = self.f_F1_2W[k]
            sq_forces = f1.T @ f1 + f2.T @ f2
            if self.two_feet:
                f1 = self.f_F2_1W[k]
                f2 = self.f_F2_2W[k]
                sq_forces += f1.T @ f1 + f2.T @ f2
            c = self.prog.AddQuadraticCost(cost_force * sq_forces)
            self.costs["sq_forces"].append(c)

        # squared torques
        for k in range(self.num_steps):
            tau1 = self.tau_F1_1[k]
            tau2 = self.tau_F1_2[k]
            sq_torques = tau1**2 + tau2**2
            if self.two_feet:
                tau3 = self.tau_F2_1[k]
                tau4 = self.tau_F2_2[k]
                sq_torques += tau3**2 + tau4**2
            c = self.prog.AddQuadraticCost(cost_torque * sq_torques)
            self.costs["sq_torques"].append(c)

        # TODO: do we need these? Potentially remove
        # squared accelerations
        # for k in range(self.num_steps):
        #     sq_acc = self.a_WB[k].T @ self.a_WB[k]
        #     c = self.prog.AddQuadraticCost(cost_acc_lin * sq_acc)
        #     self.costs["sq_acc_lin"].append(c)
        #
        # for k in range(self.num_steps):
        #     sq_rot_acc = self.omega_dot_WB[k] ** 2
        #     c = self.prog.AddQuadraticCost(cost_acc_rot * sq_rot_acc)
        #     self.costs["sq_acc_rot"].append(c)
        #
        # squared robot velocity
        for k in range(self.num_steps):
            v = self.v_WB[k]
            sq_lin_vel = v.T @ v
            c = self.prog.AddQuadraticCost(cost_lin_vel * sq_lin_vel)
            self.costs["sq_lin_vel"].append(c)

            sq_rot_vel = self.omega_WB[k] ** 2
            c = self.prog.AddQuadraticCost(cost_ang_vel * sq_rot_vel)
            self.costs["sq_rot_vel"].append(c)

        # squared distance from nominal pose
        # TODO: Use the mean stone height?
        pose_offset = np.array(
            [0, self.stone_first.height, 0]
        )  # offset the stone height
        for k in range(self.num_steps):
            pose = self.get_robot_pose(k) - pose_offset
            diff = pose - robot.get_nominal_pose()
            sq_diff = diff.T @ diff
            c = self.prog.AddQuadraticCost(cost_nominal_pose * sq_diff)  # type: ignore
            self.costs["sq_nominal_pose"].append(c)

    @property
    def dt(self) -> float:
        return self.config.dt

    def get_state(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate(
            [self.p_WB[k], [self.theta_WB[k]], self.v_WB[k], [self.omega_WB[k]]]
        )

    def get_input(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        if self.two_feet:
            return np.concatenate(
                [
                    self.f_F1_1W[k],
                    self.f_F1_2W[k],
                    [self.p_WF1_x[k]],
                    self.f_F2_1W[k],
                    self.f_F2_2W[k],
                    [self.p_WF2_x[k]],
                ]
            )
        else:
            return np.concatenate([self.f_F1_1W[k], self.f_F1_2W[k], [self.p_WF1_x[k]]])

    def get_foot_pos(self, foot: Literal["first", "last"], k: int) -> Variable:
        """
        Returns the decision variable for a given foot for a given knot point idx.
        If the segment has only one foot contact, it returns that one foot always.
        """
        if k == -1:
            k = self.config.period_steps - 1
        if self.two_feet:
            if foot == "first":
                return self.p_WF1_x[k]
            else:  # last
                return self.p_WF2_x[k]
        else:  # if only one foot we return that one foot
            return self.p_WF1_x[k]

    def get_dynamics(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate(
            [self.v_WB[k], [self.omega_WB[k]], self.a_WB[k], [self.omega_dot_WB[k]]]
        )

    def get_var_in_vertex(self, var: Variable, vertex_vars: npt.NDArray) -> Variable:
        idxs = self.relaxed_prog.FindDecisionVariableIndex(var)
        return vertex_vars[idxs]

    def get_vars_in_vertex(
        self, vars: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        shape = vars.shape
        idxs = self.relaxed_prog.FindDecisionVariableIndices(vars.flatten())
        return vertex_vars[idxs].reshape(shape)

    def get_lin_exprs_in_vertex(
        self, exprs: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        original_shape = exprs.shape
        if len(exprs.shape) > 1:
            exprs = exprs.flatten()

        # note: the dynamics are always linear (we introduced some aux vars to achieve this)
        vars = Variables()
        for e in exprs:
            if type(e) == Variable:
                vars.insert(e)
            elif type(e) == Expression:
                vars.insert(e.GetVariables())
            elif type(e) == float:
                continue  # no variables to add
            else:
                raise RuntimeError("Unknown type")
        vars = list(vars)

        A, b = DecomposeAffineExpressions(exprs, vars)
        idxs = self.relaxed_prog.FindDecisionVariableIndices(vars)

        x = vertex_vars[idxs]

        exprs_with_vertex_vars = A @ x + b

        exprs_with_vertex_vars = exprs_with_vertex_vars.reshape(original_shape)
        return exprs_with_vertex_vars

    def get_robot_pose(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate([self.p_WB[k], [self.theta_WB[k]]])

    def get_robot_spatial_vel(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate([self.v_WB[k], [self.omega_WB[k]]])

    def get_robot_spatial_acc(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate([self.a_WB[k], [self.omega_dot_WB[k]]])

    def get_vars(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        if self.two_feet:
            return np.concatenate(
                (
                    self.get_state(k),
                    self.get_input(k),
                    (
                        self.tau_F1_1[k],
                        self.tau_F1_2[k],
                        self.tau_F2_1[k],
                        self.tau_F2_2[k],
                    ),
                )
            )
        else:
            return np.concatenate(
                (
                    self.get_state(k),
                    self.get_input(k),
                    (self.tau_F1_1[k], self.tau_F1_2[k]),
                )
            )

    def add_pose_constraint(
        self, k: int, p_WB: npt.NDArray[np.float64], theta_WB: float
    ) -> None:
        if k == -1:
            k = self.config.period_steps - 1
        self.prog.AddLinearConstraint(eq(self.p_WB[k], p_WB))
        self.prog.AddLinearConstraint(self.theta_WB[k] == theta_WB)

    def add_spatial_vel_constraint(
        self, k: int, v_WB: npt.NDArray[np.float64], omega_WB: float
    ) -> None:
        if k == -1:
            k = self.config.period_steps - 1
        self.prog.AddLinearConstraint(eq(self.v_WB[k], v_WB))
        self.prog.AddLinearConstraint(self.omega_WB[k] == omega_WB)

    def make_relaxed_prog(
        self,
        trace_cost: bool = False,
        use_groups: bool = True,
    ) -> MathematicalProgram:
        if use_groups:
            variable_groups = [
                Variables(np.concatenate([self.get_vars(k), self.get_vars(k + 1)]))
                for k in range(self.num_steps - 1)
            ]
            self.relaxed_prog = MakeSemidefiniteRelaxation(
                self.prog, variable_groups=variable_groups
            )
        else:
            self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)
        if trace_cost:
            X = get_X_from_semidefinite_relaxation(self.relaxed_prog)
            EPS = 1e-6
            self.relaxed_prog.AddLinearCost(EPS * np.trace(X))

        return self.relaxed_prog

    def get_convex_set(self, use_lp_approx: bool = False) -> Spectrahedron:
        relaxed_prog = self.make_relaxed_prog()

        if use_lp_approx:
            for psd_constraint in relaxed_prog.positive_semidefinite_constraints():
                # TODO remove
                # relaxed_prog.RelaxPsdConstraintToDdDualCone(psd_constraint)
                X = get_X_from_psd_constraint(psd_constraint)
                relaxed_prog.RemoveConstraint(psd_constraint)  # type: ignore
                N = X.shape[0]
                for i in range(N):
                    X_i = X[i, i]
                    relaxed_prog.AddLinearConstraint(X_i >= 0)

        spectrahedron = Spectrahedron(relaxed_prog)
        return spectrahedron

    def evaluate_with_vertex_result(
        self,
        result: MathematicalProgramResult,
        vertex_vars: npt.NDArray,
    ) -> FootstepPlanKnotPoints:
        p_WB = result.GetSolution(self.get_vars_in_vertex(self.p_WB, vertex_vars))
        theta_WB = result.GetSolution(
            self.get_vars_in_vertex(self.theta_WB, vertex_vars)
        )
        p_WF1 = evaluate_np_expressions_array(
            self.get_lin_exprs_in_vertex(self.p_WF1, vertex_vars), result
        )
        f_F1_1W = result.GetSolution(self.get_vars_in_vertex(self.f_F1_1W, vertex_vars))
        f_F1_2W = result.GetSolution(self.get_vars_in_vertex(self.f_F1_2W, vertex_vars))

        if self.two_feet:
            p_WF2 = evaluate_np_expressions_array(
                self.get_lin_exprs_in_vertex(self.p_WF2, vertex_vars), result
            )
            f_F2_1W = result.GetSolution(
                self.get_vars_in_vertex(self.f_F2_1W, vertex_vars)
            )
            f_F2_2W = result.GetSolution(
                self.get_vars_in_vertex(self.f_F2_2W, vertex_vars)
            )

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W, p_WF2, f_F2_1W, f_F2_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W)

    def round_with_result(
        self, result: MathematicalProgramResult
    ) -> Tuple[FootstepPlanKnotPoints, MathematicalProgramResult]:
        x = result.GetSolution(self.prog.decision_variables())

        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=x)
        assert rounded_result.is_success()

        p_WB = rounded_result.GetSolution(self.p_WB)
        theta_WB = rounded_result.GetSolution(self.theta_WB)
        p_WF1 = evaluate_np_expressions_array(self.p_WF1, rounded_result)
        f_F1_1W = rounded_result.GetSolution(self.f_F1_1W)
        f_F1_2W = rounded_result.GetSolution(self.f_F1_2W)

        if self.two_feet:
            p_WF1 = evaluate_np_expressions_array(self.p_WF2, rounded_result)
            f_F1_1W = rounded_result.GetSolution(self.f_F2_1W)
            f_F1_2W = rounded_result.GetSolution(self.f_F2_2W)

            return (
                FootstepPlanKnotPoints(
                    p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W, p_WF1, f_F1_1W, f_F1_2W
                ),
                rounded_result,
            )
        else:
            return (
                FootstepPlanKnotPoints(p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W),
                rounded_result,
            )

    def round_with_vertex_result(
        self, result: MathematicalProgramResult, vertex_vars: npt.NDArray
    ) -> FootstepPlanKnotPoints:
        X_var = get_X_from_semidefinite_relaxation(self.relaxed_prog)[:-1, :-1]
        result.GetSolution(self.get_lin_exprs_in_vertex(X_var, vertex_vars))
        x = result.GetSolution(
            self.get_vars_in_vertex(self.prog.decision_variables(), vertex_vars)
        )

        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=x)
        assert rounded_result.is_success()

        p_WB = rounded_result.GetSolution(self.p_WB)
        theta_WB = rounded_result.GetSolution(self.theta_WB)
        p_WF1 = evaluate_np_expressions_array(self.p_WF1, rounded_result)
        f_F1_1W = rounded_result.GetSolution(self.f_F1_1W)
        f_F1_2W = rounded_result.GetSolution(self.f_F1_2W)

        if self.two_feet:
            p_WF2 = evaluate_np_expressions_array(self.p_WF2, rounded_result)
            f_F2_1W = rounded_result.GetSolution(self.f_F2_1W)
            f_F2_2W = rounded_result.GetSolution(self.f_F2_2W)

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W, p_WF2, f_F2_1W, f_F2_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W)

    def evaluate_with_result(
        self, result: MathematicalProgramResult
    ) -> FootstepPlanKnotPoints:
        p_WB = result.GetSolution(self.p_WB)
        theta_WB = result.GetSolution(self.theta_WB)
        p_WF1 = evaluate_np_expressions_array(self.p_WF1, result)
        f_F1_1W = result.GetSolution(self.f_F1_1W)
        f_F1_2W = result.GetSolution(self.f_F1_2W)

        if self.two_feet:
            p_WF2 = evaluate_np_expressions_array(self.p_WF2, result)
            f_F2_1W = result.GetSolution(self.f_F2_1W)
            f_F2_2W = result.GetSolution(self.f_F2_2W)

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W, p_WF2, f_F2_1W, f_F2_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WF1, f_F1_1W, f_F1_2W)

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

    def evaluate_non_convex_constraints_with_result(
        self, result: MathematicalProgramResult
    ) -> npt.NDArray[np.float64]:
        constraint_violations = []
        for cs in self.non_convex_constraints:
            violations_for_knot_point = []
            for c in cs:
                var_vals = result.GetSolution(c.variables())
                violation = np.abs(c.evaluator().Eval(var_vals))
                violations_for_knot_point.append(violation)
            constraint_violations.append(np.concatenate(violations_for_knot_point))

        return np.vstack(constraint_violations).T  # (num_steps, num_torques)
