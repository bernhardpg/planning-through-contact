from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.common.value import Value
from pydrake.math import ContinuousAlgebraicRiccatiEquation, eq
from pydrake.planning import MultipleShooting
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
    FirstOrderTaylorApproximation,
    Linearize,
    LinearSystem,
)

from planning_through_contact.tools.types import NpVariableArray


@dataclass
class HybridMpcConfig:
    horizon: int = 10
    step_size: float = 0.1
    num_sliding_steps: int = 5
    rate_Hz: int = 200
    pusher_radius: float = 0.01


class HybridModes(Enum):
    STICKING = 0
    SLIDING_LEFT = 1
    SLIDING_RIGHT = 2


class HybridMpc:
    def __init__(
        self, model: System, config: HybridMpcConfig = HybridMpcConfig()
    ) -> None:
        self.model = model
        self.cfg = config

        self.num_states = model.num_continuous_states()
        self.num_inputs = model.get_input_port().size()

    def _get_linear_system(
        self, state: npt.NDArray[np.float64], control: npt.NDArray[np.float64]
    ) -> AffineSystem:
        """
        Linearizes around 'state' and 'control', and returns an affine system

        x_dot = A x + B u + f_0

        where f_0 = x_dot_0 - A x_0 - B u_u
        where _0 denotes the nominal state and input

        """
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

        env = {
            x: state[0],
            y: state[1],
            theta: state[2],
            lam: state[3],
            c_n: control[0],
            c_f: control[1],
            lam_dot: control[2],
        }
        A = sym.Evaluate(A_sym, env)
        B = sym.Evaluate(B_sym, env)

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
        N = self.cfg.horizon
        h = self.cfg.step_size
        num_sliding_steps = self.cfg.num_sliding_steps

        prog = MathematicalProgram()

        # Formulate the problem in the local coordinates around the nominal trajectory
        x_bar = prog.NewContinuousVariables(self.num_states, N, "x_bar")
        u_bar = prog.NewContinuousVariables(self.num_inputs, N, "u_bar")

        # Initial value constraint
        x_bar_curr = x_curr - x_traj[0]
        prog.AddLinearConstraint(eq(x_bar[:, 0], x_bar_curr))

        # Dynamic constraints
        lin_systems = [
            self._get_linear_system(state, control)
            for state, control in zip(x_traj, u_traj)
        ][
            :-1
        ]  # only need N-1 linear systems

        assert len(lin_systems) == N - 1

        As = [sys.A() for sys in lin_systems]
        Bs = [sys.B() for sys in lin_systems]
        for i, (A, B) in enumerate(zip(As, Bs)):
            x_bar_dot = A.dot(x_bar[:, i]) + B.dot(u_bar[:, i])
            forward_euler = x_bar[:, i] + h * x_bar_dot
            prog.AddLinearConstraint(eq(x_bar[:, i + 1], forward_euler))

        # x_bar = x - x_traj
        x = x_bar + np.vstack(x_traj).T
        # u_bar = u - u_traj
        u = u_bar + np.vstack(u_traj).T

        # Control constraints
        FRICTION_COEFF = 0.5
        for i, u_i in enumerate(u.T):
            c_n = u_i[0]
            c_f = u_i[1]
            lam_dot = u_i[2]

            prog.AddLinearConstraint(c_n >= 0)

            if mode == HybridModes.STICKING or i > num_sliding_steps:
                prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
                prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

            elif mode == HybridModes.SLIDING_LEFT:
                prog.AddLinearConstraint(c_f == FRICTION_COEFF * c_n)

            else:  # SLIDING_RIGHT
                prog.AddLinearConstraint(c_f == -FRICTION_COEFF * c_n)

        # State constraints
        for state in x.T:
            lam = state[3]
            prog.AddLinearConstraint(lam >= 0)
            prog.AddLinearConstraint(lam <= 1)

        # Cost
        Q = np.diag([1, 1, 1, 0]) * 10
        R = np.diag([1, 1, 1]) * 0.01
        Q_N = Q

        terminal_cost = x_bar[:, -1].T.dot(Q_N).dot(x_bar[:, -1])
        state_running_cost = sum(
            [x_bar[:, i].T.dot(Q).dot(x_bar[:, i]) for i in range(N)]
        )
        input_running_cost = sum(
            [u_bar[:, i].T.dot(R).dot(u_bar[:, i]) for i in range(N)]
        )
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

        state_sol = sym.Evaluate(result.GetSolution(state))  # type: ignore
        x_next = state_sol[:, 1]

        x_dot_curr = (x_next - x_curr) / self.cfg.step_size

        control_sol = sym.Evaluate(result.GetSolution(control))  # type: ignore
        u_next = control_sol[:, 0]
        return x_dot_curr, u_next


class HybridModelPredictiveControlSystem(LeafSystem):
    def __init__(
        self, model: System, config: HybridMpcConfig = HybridMpcConfig()
    ) -> None:
        super().__init__()

        self.mpc = HybridMpc(model, config)
        self.cfg = self.mpc.cfg

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

        _, control_next = self.mpc.compute_control(x_curr, x_traj, u_traj)
        output.SetFromVector(control_next)  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_control_port(self) -> InputPort:
        return self.desired_control_port
