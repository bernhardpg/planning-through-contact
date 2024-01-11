from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import logging
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.common.value import Value
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import (
    BasicVector,
    Context,
    InputPort,
    LeafSystem,
    OutputPort,
    System,
)
from pydrake.systems.primitives import (
    AffineSystem,
)

from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.types import NpVariableArray

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4, suppress=True)

logger = logging.getLogger(__name__)


@dataclass
class HybridMpcConfig:
    horizon: int = 10
    step_size: float = 0.1
    num_sliding_steps: int = 5
    rate_Hz: int = 200
    enforce_hard_end_constraint: bool = False
    Q: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.diag([3, 3, 0.1, 0]) * 10
    )
    Q_N: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.diag([3, 3, 0.1, 0]) * 2000
    )
    R: npt.NDArray[np.float64] = field(default_factory=lambda: np.diag([1, 1, 0]) * 0.5)
    # Max magnitude of control input [c_n, c_f, lam_dot]
    u_max_magnitude: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.3, 0.3, 0.05])
    )


class HybridModes(Enum):
    STICKING = 0
    SLIDING_LEFT = 1
    SLIDING_RIGHT = 2


class HybridMpc:
    def __init__(
        self,
        model: System,
        config: HybridMpcConfig,
        dynamics_config: SliderPusherSystemConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.dynamics_config = dynamics_config

        self.num_states = model.num_continuous_states()
        self.num_inputs = model.get_input_port().size()

        self.A_sym, self.B_sym, self.sym_vars = self._calculate_symbolic_system()

        self.control_log: List[npt.NDArray[np.float64]] = []
        self.cost_log: List[float] = []
        self.desired_velocity_log: List[npt.NDArray[np.float64]] = []
        self.commanded_velocity_log: List[npt.NDArray[np.float64]] = []

    def _calculate_symbolic_system(
        self,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # TODO(bernhardpg): This causes a segfault. Look into this
        # self.model_context.SetContinuousState(state)
        # self.model.get_input_port().FixValue(self.model_context, input)
        # lin_sys = FirstOrderTaylorApproximation(self.model, self.model_context)

        # TODO(bernhardpg): Replace all of the following once symbolic computation works
        sym_model = self.model.ToSymbolic()
        x = sym.Variable("x")
        y = sym.Variable("y")
        theta = sym.Variable("theta")
        lam = sym.Variable("lam")
        state_sym = np.array([x, y, theta, lam])

        c_n = sym.Variable("c_n")
        c_f = sym.Variable("c_f")
        lam_dot = sym.Variable("lam_dot")
        control_sym = np.array([c_n, c_f, lam_dot])

        x_dot = sym_model.calc_dynamics(state_sym, control_sym)
        A_sym = sym.Jacobian(x_dot, state_sym)
        B_sym = sym.Jacobian(x_dot, control_sym)

        sym_vars = np.concatenate([state_sym, control_sym])

        return A_sym, B_sym, sym_vars  # type: ignore

    def _create_env(
        self, state: npt.NDArray[np.float64], control: npt.NDArray[np.float64]
    ) -> Dict[sym.Variable, float]:
        var_vals = np.concatenate([state, control])
        env = {sym: val for sym, val in zip(self.sym_vars, var_vals)}
        return env

    def _get_linear_system(
        self, state: npt.NDArray[np.float64], control: npt.NDArray[np.float64]
    ) -> AffineSystem:
        """
        Linearizes around 'state' and 'control', and returns an affine system

        x_dot = A x + B u + f_0

        where f_0 = x_dot_0 - A x_0 - B u_u
        where _0 denotes the nominal state and input

        """

        env = self._create_env(state, control)
        A = sym.Evaluate(self.A_sym, env)
        B = sym.Evaluate(self.B_sym, env)

        x_dot_desired = self.model.calc_dynamics(state, control).flatten()  # type: ignore
        f = x_dot_desired - A.dot(state) - B.dot(control)

        return AffineSystem(A, B, f)

    def _setup_QP(
        self,
        x_curr: npt.NDArray[np.float64],
        x_traj: List[npt.NDArray[np.float64]],
        u_traj: List[npt.NDArray[np.float64]],
        mode: HybridModes,
    ) -> Tuple[MathematicalProgram, NpVariableArray, NpVariableArray]:
        N = len(x_traj)
        h = self.config.step_size
        num_sliding_steps = self.config.num_sliding_steps

        prog = MathematicalProgram()

        # Formulate the problem in the local coordinates around the nominal trajectory
        x_bar = prog.NewContinuousVariables(self.num_states, N, "x_bar")
        u_bar = prog.NewContinuousVariables(self.num_inputs, N - 1, "u_bar")

        # Initial value constraint
        x_bar_curr = x_curr - x_traj[0]
        prog.AddLinearConstraint(eq(x_bar[:, 0], x_bar_curr))

        # Dynamic constraints
        lin_systems = [
            self._get_linear_system(state, control)
            for state, control in zip(x_traj, u_traj)
        ][
            : N - 1
        ]  # we only have N-1 controls

        As = [sys.A() for sys in lin_systems]
        Bs = [sys.B() for sys in lin_systems]
        for i, (A, B) in enumerate(zip(As, Bs)):
            x_bar_dot = A.dot(x_bar[:, i]) + B.dot(u_bar[:, i])
            forward_euler = x_bar[:, i] + h * x_bar_dot
            prog.AddLinearConstraint(eq(x_bar[:, i + 1], forward_euler))

        # Last error state should be exactly 0
        if self.config.enforce_hard_end_constraint:
            prog.AddLinearConstraint(eq(x_bar[:, N - 1], np.zeros(x_traj[-1].shape)))

        # x_bar = x - x_traj
        x = x_bar + np.vstack(x_traj).T

        if len(u_traj) == N:  # make sure u_traj is not too long
            u_traj = u_traj[: N - 1]
        # u_bar = u - u_traj
        u = u_bar + np.vstack(u_traj).T

        # Control constraints
        mu = self.dynamics_config.friction_coeff_slider_pusher
        # Control limits:
        lb = np.array(
            [0, -self.config.u_max_magnitude[1], -self.config.u_max_magnitude[2]]
        )
        ub = self.config.u_max_magnitude
        for i, u_i in enumerate(u.T):
            c_n = u_i[0]
            c_f = u_i[1]
            lam_dot = u_i[2]

            prog.AddLinearConstraint(c_n >= 0)

            if mode == HybridModes.STICKING or i > num_sliding_steps:
                prog.AddLinearConstraint(c_f <= mu * c_n)
                prog.AddLinearConstraint(c_f >= -mu * c_n)
                prog.AddLinearEqualityConstraint(lam_dot == 0)

            elif mode == HybridModes.SLIDING_LEFT:
                prog.AddLinearConstraint(c_f == mu * c_n)
                prog.AddLinearConstraint(lam_dot <= 0)

            else:  # SLIDING_RIGHT
                prog.AddLinearConstraint(c_f == -mu * c_n)
                prog.AddLinearConstraint(lam_dot >= 0)

            # Control Limits:
            prog.AddLinearConstraint(c_n <= ub[0])
            prog.AddLinearConstraint(c_f >= lb[1])
            prog.AddLinearConstraint(c_f <= ub[1])
            prog.AddLinearConstraint(lam_dot >= lb[2])
            prog.AddLinearConstraint(lam_dot <= ub[2])

        # State constraints
        for state in x.T:
            lam = state[3]
            prog.AddLinearConstraint(lam >= 0)
            prog.AddLinearConstraint(lam <= 1)

        Q = self.config.Q
        R = self.config.R
        Q_N = self.config.Q_N

        state_running_cost = sum(
            [x_bar[:, i].T.dot(Q).dot(x_bar[:, i]) for i in range(N - 1)]
        )
        input_running_cost = sum(
            [u_bar[:, i].T.dot(R).dot(u_bar[:, i]) for i in range(N - 1)]
        )
        terminal_cost = x_bar[:, N - 1].T.dot(Q_N).dot(x_bar[:, N - 1])
        prog.AddCost(terminal_cost + state_running_cost + input_running_cost)

        return prog, x, u  # type: ignore

    def compute_control(
        self,
        x_curr: npt.NDArray[np.float64],
        x_traj: List[npt.NDArray[np.float64]],
        u_traj: List[npt.NDArray[np.float64]],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Solve one prog per contact mode
        progs, states, controls = zip(
            *[self._setup_QP(x_curr, x_traj, u_traj, mode) for mode in HybridModes]
        )
        results = [Solve(prog) for prog in progs]  # type: ignore

        costs = [
            result.get_optimal_cost() if result.is_success() else np.inf
            for result in results
        ]  # we do not allow infeasible results
        best_idx = np.argmin(costs)
        state = states[best_idx]
        control = controls[best_idx]
        result = results[best_idx]
        lowest_cost = costs[best_idx]
        self.cost_log.append(lowest_cost)
        state_sol = sym.Evaluate(result.GetSolution(state))  # type: ignore

        x_next = state_sol[:, 1]

        # Finite difference method to get x_dot_curr
        # Uses the linear approximation of the dynamics
        x_dot_curr = (x_next - x_curr) / self.config.step_size

        control_sol = sym.Evaluate(result.GetSolution(control))  # type: ignore

        if len(control_sol.T) == self.config.horizon:
            self.control_log.append(control_sol.T)
        else:
            # Padding because the solution length gets smaller as the prediction horizon decreases
            # towards the end of the trajectory segment
            padding = self.config.horizon - control_sol.shape[1]
            padded_array = np.pad(control_sol, ((0, 0), (0, padding)), mode="constant")
            self.control_log.append(padded_array.T)

        if lowest_cost == np.inf:
            logger.debug(
                f"Infeasible: x_dot_curr:{x_dot_curr}, u_next:{control_sol[:, 0]} "
            )
        u_next = control_sol[:, 0]

        # Finite difference method to get velocity of pusher
        v_WP_W = (
            self.model.get_p_WP_from_state(x_next)
            - self.model.get_p_WP_from_state(x_curr)
        ) / self.config.step_size

        return x_dot_curr.flatten(), u_next, v_WP_W


# Not used in pusher pose controller
class HybridModelPredictiveControlSystem(LeafSystem):
    def __init__(
        self, model: System, config: HybridMpcConfig = HybridMpcConfig()
    ) -> None:
        super().__init__()

        self.mpc = HybridMpc(model, config, model.config)  # type: ignore
        self.config = self.mpc.config

        self.state_port = self.DeclareVectorInputPort("state", self.mpc.num_states)

        self.desired_state_port = self.DeclareAbstractInputPort(
            "desired_state",
            Value([np.array([])]),
        )

        self.control_port = self.DeclareVectorOutputPort(
            "control", self.mpc.num_inputs, self.CalcControl
        )
        self.desired_control_port = self.DeclareAbstractInputPort(
            "desired_control",
            Value([np.array([])]),
        )

    def CalcControl(self, context: Context, output: BasicVector):
        x_curr: npt.NDArray[np.float64] = self.state_port.Eval(context)  # type: ignore
        x_traj: List[npt.NDArray[np.float64]] = self.desired_state_port.Eval(context)  # type: ignore
        u_traj: List[npt.NDArray[np.float64]] = self.desired_control_port.Eval(context)  # type: ignore
        if len(u_traj) > 1:
            # Closed loop control
            _, control_next, _ = self.mpc.compute_control(x_curr, x_traj, u_traj)

            # Open loop control
            # control_next = u_traj[0]
        else:
            control_next = np.array([0, 0, 0])
        output.SetFromVector(control_next)  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_control_port(self) -> InputPort:
        return self.desired_control_port
