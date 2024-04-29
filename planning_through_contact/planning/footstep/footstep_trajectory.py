import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import GraphOfConvexSets, Spectrahedron
from pydrake.math import eq
from pydrake.solvers import (
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
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


@dataclass
class FootstepPlanKnotPoints:
    p_WB: npt.NDArray[np.float64]
    theta_WB: npt.NDArray[np.float64]
    p_WFl: npt.NDArray[np.float64]
    f_Fl_1W: npt.NDArray[np.float64]
    f_Fl_2W: npt.NDArray[np.float64]
    p_WFr: Optional[npt.NDArray[np.float64]] = None
    f_Fr_1W: Optional[npt.NDArray[np.float64]] = None
    f_Fr_2W: Optional[npt.NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        assert self.p_WB.shape == (self.num_points, 2)
        assert self.theta_WB.shape == (self.num_points,)

        assert self.p_WFl.shape == (self.num_points, 2)
        assert self.f_Fl_1W.shape == (self.num_points, 2)
        assert self.f_Fl_2W.shape == (self.num_points, 2)

        if self.p_WFr is not None:
            assert self.p_WFr.shape == (self.num_points, 2)
        if self.f_Fr_1W is not None:
            assert self.f_Fr_1W.shape == (self.num_points, 2)
        if self.f_Fr_2W is not None:
            assert self.f_Fr_2W.shape == (self.num_points, 2)

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
        gait_schedule: npt.NDArray[np.bool_],  # (num_steps, 2)
    ) -> "FootstepTrajectory":
        p_WBs = np.vstack([k.p_WB for k in segments])
        theta_WBs = np.vstack([k.theta_WB for k in segments]).flatten()

        if not gait_schedule.shape == (len(segments), 2):
            raise RuntimeError("Gait schedule length must match number of segments")

        p_WFls = []
        f_Fl_1Ws = []
        f_Fl_2Ws = []
        p_WFrs = []
        f_Fr_1Ws = []
        f_Fr_2Ws = []

        # NOTE: This assumes that all the segments have the same lengths!
        empty_shape = segments[0].p_WB.shape

        for segment, (left_active, right_active) in zip(segments, gait_schedule):
            both_active = left_active and right_active
            if both_active:
                p_WFls.append(segment.p_WFl)
                f_Fl_1Ws.append(segment.f_Fl_1W)
                f_Fl_2Ws.append(segment.f_Fl_2W)

                p_WFrs.append(segment.p_WFr)
                f_Fr_1Ws.append(segment.f_Fr_1W)
                f_Fr_2Ws.append(segment.f_Fr_2W)
            else:
                # NOTE: These next lines look like they have a typo, but they don't.
                # When there is only one foot active, the values for this foot is
                # always stored in the "left" foot values (to avoid unecessary optimization
                # variables)
                if left_active:
                    p_WFls.append(segment.p_WFl)
                    f_Fl_1Ws.append(segment.f_Fl_1W)
                    f_Fl_2Ws.append(segment.f_Fl_2W)

                    p_WFrs.append(np.full(empty_shape, np.nan))
                    f_Fr_1Ws.append(np.full(empty_shape, np.nan))
                    f_Fr_2Ws.append(np.full(empty_shape, np.nan))
                else:  # right_active
                    p_WFls.append(np.full(empty_shape, np.nan))
                    f_Fl_1Ws.append(np.full(empty_shape, np.nan))
                    f_Fl_2Ws.append(np.full(empty_shape, np.nan))

                    # Notice that here we pick from the "left" values
                    p_WFrs.append(segment.p_WFl)
                    f_Fr_1Ws.append(segment.f_Fl_1W)
                    f_Fr_2Ws.append(segment.f_Fl_2W)

        p_WFls = np.vstack(p_WFls)
        f_Fl_1Ws = np.vstack(f_Fl_1Ws)
        f_Fl_2Ws = np.vstack(f_Fl_2Ws)

        p_WFrs = np.vstack(p_WFrs)
        f_Fr_1Ws = np.vstack(f_Fr_1Ws)
        f_Fr_2Ws = np.vstack(f_Fr_2Ws)

        merged_knot_points = FootstepPlanKnotPoints(
            p_WBs, theta_WBs, p_WFls, f_Fl_1Ws, f_Fl_2Ws, p_WFrs, f_Fr_1Ws, f_Fr_2Ws
        )

        return cls(merged_knot_points, dt)


@dataclass
class FootstepPlanSegment:
    def __init__(
        self,
        stone: InPlaneSteppingStone,
        active_feet: npt.NDArray[np.bool_],
        robot: PotatoRobot,
        config: FootstepPlanningConfig,
        name: Optional[str] = None,
    ) -> None:
        """ """
        # Assume we always only have one foot in contact
        if name is not None:
            self.name = f"{stone.name}_{name}"

        self.config = config
        self.active_feet = active_feet
        self.two_feet = all(active_feet)

        num_steps = self.config.period_steps

        self.prog = MathematicalProgram()

        # declare states
        self.p_WB = self.prog.NewContinuousVariables(num_steps, 2, "p_WB")
        self.v_WB = self.prog.NewContinuousVariables(num_steps, 2, "v_WB")
        self.theta_WB = self.prog.NewContinuousVariables(num_steps, "theta_WB")
        self.omega_WB = self.prog.NewContinuousVariables(num_steps, "omega_WB")

        # declare inputs

        # WLOG. we call the first foot right foot and the second foot left foot.
        # Planning-wise this does not make a difference, but it makes the naming easier.
        # When executing the plan, the correct foot must be determined from the gait.

        # left foot
        self.p_WFl_x = self.prog.NewContinuousVariables(num_steps, "p_BFl_W_x")
        self.f_Fl_1W = self.prog.NewContinuousVariables(num_steps, 2, "f_Fl_1W")
        self.f_Fl_2W = self.prog.NewContinuousVariables(num_steps, 2, "f_Fl_2W")
        if self.two_feet:
            self.p_WFr_x = self.prog.NewContinuousVariables(num_steps, "p_BFr_W_x")
            self.f_Fr_1W = self.prog.NewContinuousVariables(num_steps, 2, "f_Fr_1W")
            self.f_Fr_2W = self.prog.NewContinuousVariables(num_steps, 2, "f_Fr_2W")

        self.p_WFl = np.vstack(
            [self.p_WFl_x, np.full(self.p_WFl_x.shape, stone.z_pos)]
        ).T  # (num_steps, 2)
        if self.two_feet:
            self.p_WFr = np.vstack(
                [self.p_WFr_x, np.full(self.p_WFr_x.shape, stone.z_pos)]
            ).T  # (num_steps, 2)

        # compute the foot position
        self.p_BFl_W = self.p_WFl - self.p_WB
        if self.two_feet:
            self.p_BFr_W = self.p_WFr - self.p_WB

        # auxilliary vars
        # TODO(bernhardpg): we might be able to get around this once we
        # have SDP constraints over the edges
        self.tau_Fl_1 = self.prog.NewContinuousVariables(num_steps, "tau_Fl_1")
        self.tau_Fl_2 = self.prog.NewContinuousVariables(num_steps, "tau_Fl_2")
        if self.two_feet:
            self.tau_Fr_1 = self.prog.NewContinuousVariables(num_steps, "tau_Fr_1")
            self.tau_Fr_2 = self.prog.NewContinuousVariables(num_steps, "tau_Fr_2")

        # linear acceleration
        g = np.array([0, -9.81])
        self.a_WB = (1 / robot.mass) * (self.f_Fl_1W + self.f_Fl_2W) + g
        if self.two_feet:
            self.a_WB += (1 / robot.mass) * (self.f_Fr_1W + self.f_Fr_2W)

        # angular acceleration
        self.omega_dot_WB = (1 / robot.inertia) * (self.tau_Fl_1 + self.tau_Fl_2)
        if self.two_feet:
            self.omega_dot_WB += (1 / robot.inertia) * (self.tau_Fr_1 + self.tau_Fr_2)

        # contact points positions
        self.p_BFl_1W = self.p_BFl_W + np.array([robot.foot_length / 2, 0])
        self.p_BFl_2W = self.p_BFl_W - np.array([robot.foot_length / 2, 0])
        if self.two_feet:
            self.p_BFr_1W = self.p_BFr_W + np.array([robot.foot_length / 2, 0])
            self.p_BFr_2W = self.p_BFr_W - np.array([robot.foot_length / 2, 0])

        # Start and end in an equilibrium position
        self.prog.AddLinearConstraint(eq(self.a_WB[0], 0))
        self.prog.AddLinearConstraint(eq(self.a_WB[num_steps - 1], 0))
        self.prog.AddLinearEqualityConstraint(self.omega_dot_WB[0], 0)
        self.prog.AddLinearEqualityConstraint(self.omega_dot_WB[num_steps - 1], 0)

        self.non_convex_constraints = []
        for k in range(num_steps):
            # torque = arm x force
            cs_for_knot_point = []
            c = self.prog.AddQuadraticConstraint(
                self.tau_Fl_1[k] - cross_2d(self.p_BFl_1W[k], self.f_Fl_1W[k]), 0, 0
            )
            cs_for_knot_point.append(c)
            c = self.prog.AddQuadraticConstraint(
                self.tau_Fl_2[k] - cross_2d(self.p_BFl_2W[k], self.f_Fl_2W[k]), 0, 0
            )
            cs_for_knot_point.append(c)
            if self.two_feet:
                c = self.prog.AddQuadraticConstraint(
                    self.tau_Fr_1[k] - cross_2d(self.p_BFr_1W[k], self.f_Fr_1W[k]), 0, 0
                )
                cs_for_knot_point.append(c)
                c = self.prog.AddQuadraticConstraint(
                    self.tau_Fr_2[k] - cross_2d(self.p_BFr_2W[k], self.f_Fr_2W[k]), 0, 0
                )
                cs_for_knot_point.append(c)

            self.non_convex_constraints.append(cs_for_knot_point)

            # Stay on the stepping stone
            self.prog.AddLinearConstraint(stone.x_min <= self.p_WFl[k][0])
            self.prog.AddLinearConstraint(self.p_WFl[k][0] <= stone.x_max)
            if self.two_feet:
                self.prog.AddLinearConstraint(stone.x_min <= self.p_WFr[k][0])
                self.prog.AddLinearConstraint(self.p_WFr[k][0] <= stone.x_max)

            # Don't move the feet too far from the robot
            MAX_DIST = 0.4
            # TODO: Add some reasonable bounds here
            # self.prog.AddLinearConstraint(
            #     self.p_WB[k][0] - self.p_WFl[k][0] <= MAX_DIST
            # )
            # self.prog.AddLinearConstraint(
            #     self.p_WB[k][0] - self.p_WFl[k][0] >= -MAX_DIST
            # )
            # if self.two_feet:
            #     self.prog.AddLinearConstraint(
            #         self.p_WB[k][0] - self.p_WFr[k][0] <= MAX_DIST
            #     )
            #     self.prog.AddLinearConstraint(
            #         self.p_WB[k][0] - self.p_WFr[k][0] >= -MAX_DIST
            #     )

            # TODO(bernhardpg): Friction cone must be formulated differently
            # when we have tilted ground
            mu = 0.5  # TODO: move friction coeff
            for f in (self.f_Fl_1W, self.f_Fl_2W):
                self.prog.AddLinearConstraint(f[k][1] >= 0)
                self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])
            if self.two_feet:
                for f in (self.f_Fr_1W, self.f_Fr_2W):
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
            dynamics = s_next - (s_curr + dt * f)
            self.prog.AddLinearConstraint(eq(dynamics, 0))

            # foot can't move during segment
            const = eq(self.p_WFl[k], self.p_WFl[k + 1])
            for c in const:
                self.prog.AddLinearEqualityConstraint(c)
            if self.two_feet:
                const = eq(self.p_WFr[k], self.p_WFr[k + 1])
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
        cost_acc_lin = 100.0
        cost_acc_rot = 1.0
        cost_lin_vel = 10
        cost_ang_vel = 1.0
        cost_nominal_pose = 1.0

        # cost_force = 1e-5
        # cost_torque = 1e-3
        # cost_lin_vel = 10.0
        # cost_ang_vel = 0.1
        # cost_nominal_pose = 1.0

        # squared forces
        for k in range(num_steps):
            f1 = self.f_Fl_1W[k]
            f2 = self.f_Fl_2W[k]
            sq_forces = f1.T @ f1 + f2.T @ f2
            if self.two_feet:
                f1 = self.f_Fr_1W[k]
                f2 = self.f_Fr_2W[k]
                sq_forces += f1.T @ f1 + f2.T @ f2
            c = self.prog.AddQuadraticCost(cost_force * sq_forces)
            self.costs["sq_forces"].append(c)

        # squared torques
        for k in range(num_steps):
            tau1 = self.tau_Fl_1[k]
            tau2 = self.tau_Fl_2[k]
            sq_torques = tau1**2 + tau2**2
            if self.two_feet:
                tau3 = self.tau_Fr_1[k]
                tau4 = self.tau_Fr_2[k]
                sq_torques += tau3**2 + tau4**2
            c = self.prog.AddQuadraticCost(cost_torque * sq_torques)
            self.costs["sq_torques"].append(c)

        # squared accelerations
        for k in range(num_steps):
            sq_acc = self.a_WB[k].T @ self.a_WB[k]
            c = self.prog.AddQuadraticCost(cost_acc_lin * sq_acc)
            self.costs["sq_acc_lin"].append(c)

        for k in range(num_steps):
            sq_rot_acc = self.omega_dot_WB[k] ** 2
            c = self.prog.AddQuadraticCost(cost_acc_rot * sq_rot_acc)
            self.costs["sq_acc_rot"].append(c)

        # squared robot velocity
        for k in range(num_steps):
            v = self.v_WB[k]
            sq_lin_vel = v.T @ v
            c = self.prog.AddQuadraticCost(cost_lin_vel * sq_lin_vel)
            self.costs["sq_lin_vel"].append(c)

            sq_rot_vel = self.omega_WB[k] ** 2
            c = self.prog.AddQuadraticCost(cost_ang_vel * sq_rot_vel)
            self.costs["sq_rot_vel"].append(c)

        # squared distance from nominal pose
        for k in range(num_steps):
            pose = self.get_robot_pose(k)
            diff = pose - robot.get_nominal_pose()
            sq_diff = diff.T @ diff
            c = self.prog.AddQuadraticCost(cost_nominal_pose * sq_diff)
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
                    self.f_Fl_1W[k],
                    self.f_Fl_2W[k],
                    self.p_WFl[k],
                    self.f_Fr_1W[k],
                    self.f_Fr_2W[k],
                    self.p_WFr[k],
                ]
            )
        else:
            return np.concatenate([self.f_Fl_1W[k], self.f_Fl_2W[k], self.p_WFl[k]])

    def get_dynamics(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate(
            [self.v_WB[k], [self.omega_WB[k]], self.a_WB[k], [self.omega_dot_WB[k]]]
        )

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
        return np.concatenate([self.p_WB[k], [self.theta_WB[k]]])

    def get_robot_spatial_vel(self, k: int) -> npt.NDArray:
        return np.concatenate([self.v_WB[k], [self.omega_WB[k]]])

    def get_robot_spatial_acc(self, k: int) -> npt.NDArray:
        return np.concatenate([self.a_WB[k], [self.omega_dot_WB[k]]])

    def get_vars(self, k: int) -> npt.NDArray:
        raise NotImplementedError("This needs to be updated for two feet")
        return np.concatenate(
            (self.get_state(k), self.get_input(k), (self.tau_Fl_1[k], self.tau_Fl_2[k]))
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

    def make_relaxed_prog(self, trace_cost: bool = False) -> MathematicalProgram:
        self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)
        if trace_cost:
            X = get_X_from_semidefinite_relaxation(self.relaxed_prog)
            EPS = 1e-6
            self.relaxed_prog.AddLinearCost(EPS * np.trace(X))

        return self.relaxed_prog

    def get_convex_set(self) -> Spectrahedron:
        relaxed_prog = self.make_relaxed_prog()
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
        p_WFl = evaluate_np_expressions_array(
            self.get_lin_exprs_in_vertex(self.p_WFl, vertex_vars), result
        )
        f_Fl_1W = result.GetSolution(self.get_vars_in_vertex(self.f_Fl_1W, vertex_vars))
        f_Fl_2W = result.GetSolution(self.get_vars_in_vertex(self.f_Fl_2W, vertex_vars))

        if self.two_feet:
            p_WFr = evaluate_np_expressions_array(
                self.get_lin_exprs_in_vertex(self.p_WFr, vertex_vars), result
            )
            f_Fr_1W = result.GetSolution(
                self.get_vars_in_vertex(self.f_Fr_1W, vertex_vars)
            )
            f_Fr_2W = result.GetSolution(
                self.get_vars_in_vertex(self.f_Fr_2W, vertex_vars)
            )

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W, p_WFr, f_Fr_1W, f_Fr_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W)

    def round_with_result(
        self, result: MathematicalProgramResult
    ) -> Tuple[FootstepPlanKnotPoints, MathematicalProgramResult]:
        X = result.GetSolution(
            get_X_from_semidefinite_relaxation(self.relaxed_prog)[:-1, :-1]
        )
        x = result.GetSolution(self.prog.decision_variables())

        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=x)
        assert rounded_result.is_success()

        p_WB = rounded_result.GetSolution(self.p_WB)
        theta_WB = rounded_result.GetSolution(self.theta_WB)
        p_WFl = evaluate_np_expressions_array(self.p_WFl, rounded_result)
        f_Fl_1W = rounded_result.GetSolution(self.f_Fl_1W)
        f_Fl_2W = rounded_result.GetSolution(self.f_Fl_2W)

        if self.two_feet:
            p_WFr = evaluate_np_expressions_array(self.p_WFr, rounded_result)
            f_Fr_1W = rounded_result.GetSolution(self.f_Fr_1W)
            f_Fr_2W = rounded_result.GetSolution(self.f_Fr_2W)

            return (
                FootstepPlanKnotPoints(
                    p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W, p_WFr, f_Fr_1W, f_Fr_2W
                ),
                rounded_result,
            )
        else:
            return (
                FootstepPlanKnotPoints(p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W),
                rounded_result,
            )

    def round_with_vertex_result(
        self, result: MathematicalProgramResult, vertex_vars: npt.NDArray
    ) -> FootstepPlanKnotPoints:
        X_var = get_X_from_semidefinite_relaxation(self.relaxed_prog)[:-1, :-1]
        X = result.GetSolution(self.get_lin_exprs_in_vertex(X_var, vertex_vars))
        x = result.GetSolution(
            self.get_vars_in_vertex(self.prog.decision_variables(), vertex_vars)
        )

        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=x)
        assert rounded_result.is_success()

        p_WB = rounded_result.GetSolution(self.p_WB)
        theta_WB = rounded_result.GetSolution(self.theta_WB)
        p_WFl = evaluate_np_expressions_array(self.p_WFl, rounded_result)
        f_Fl_1W = rounded_result.GetSolution(self.f_Fl_1W)
        f_Fl_2W = rounded_result.GetSolution(self.f_Fl_2W)

        if self.two_feet:
            p_WFr = evaluate_np_expressions_array(self.p_WFr, rounded_result)
            f_Fr_1W = rounded_result.GetSolution(self.f_Fr_1W)
            f_Fr_2W = rounded_result.GetSolution(self.f_Fr_2W)

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W, p_WFr, f_Fr_1W, f_Fr_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W)

    def evaluate_with_result(
        self, result: MathematicalProgramResult
    ) -> FootstepPlanKnotPoints:
        p_WB = result.GetSolution(self.p_WB)
        theta_WB = result.GetSolution(self.theta_WB)
        p_WFl = evaluate_np_expressions_array(self.p_WFl, result)
        f_Fl_1W = result.GetSolution(self.f_Fl_1W)
        f_Fl_2W = result.GetSolution(self.f_Fl_2W)

        if self.two_feet:
            p_WFr = evaluate_np_expressions_array(self.p_WFr, result)
            f_Fr_1W = result.GetSolution(self.f_Fr_1W)
            f_Fr_2W = result.GetSolution(self.f_Fr_2W)

            return FootstepPlanKnotPoints(
                p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W, p_WFr, f_Fr_1W, f_Fr_2W
            )
        else:
            return FootstepPlanKnotPoints(p_WB, theta_WB, p_WFl, f_Fl_1W, f_Fl_2W)

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
