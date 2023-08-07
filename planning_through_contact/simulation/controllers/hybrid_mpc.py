from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.common.value import Value
from pydrake.math import eq
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


class HybridModelPredictiveControl(LeafSystem):
    def __init__(
        self, model: System, config: HybridMpcConfig = HybridMpcConfig()
    ) -> None:
        super().__init__()

        self.model = model
        self.model_context = self.model.CreateDefaultContext()
        self.cfg = config

        self.num_states = model.num_continuous_states()
        self.state_port = self.DeclareVectorInputPort("state", self.num_states)

        self.desired_state_port = self.DeclareAbstractInputPort(
            "desired_state",
            Value([np.array([])]),
        )

        assert model.num_input_ports() == 1
        self.num_inputs = model.get_input_port().size()
        self.control_port = self.DeclareVectorOutputPort(
            "control", self.num_inputs, self.CalcControl
        )
        self.desired_control_port = self.DeclareAbstractInputPort(
            "desired_control",
            Value([np.array([])]),
        )

    def _get_linear_system(
        self, state: npt.NDArray[np.float64], control: npt.NDArray[np.float64]
    ) -> AffineSystem:
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
        current_state: npt.NDArray[np.float64],
        desired_state: List[npt.NDArray[np.float64]],
        desired_control: List[npt.NDArray[np.float64]],
    ) -> Tuple[MathematicalProgram, NpVariableArray, NpVariableArray]:
        N = self.cfg.horizon
        Q = np.eye(self.num_states) * 10
        Q_N = np.eye(self.num_states)
        R = np.eye(self.num_inputs)

        prog = MathematicalProgram()
        error_state = prog.NewContinuousVariables(self.num_states, N, "error_state")
        # u = prog.NewContinuousVariables(self.num_inputs, N - 1, "u")
        error_control = prog.NewContinuousVariables(self.num_inputs, N, "error_control")

        # Initial value constraint
        curr_error_state = current_state - desired_state[0]
        prog.AddLinearConstraint(eq(error_state[:, 0], curr_error_state))

        # Dynamic constraints
        lin_systems = [
            self._get_linear_system(state, control)
            for state, control in zip(desired_state, desired_control)
        ][
            :-1
        ]  # only need N-1 linear systems
        As = [sys.A() for sys in lin_systems]
        Bs = [sys.B() for sys in lin_systems]
        for i, (A, B) in enumerate(zip(As, Bs)):
            error_state_dot = A.dot(error_state[:, i]) + B.dot(error_control[:, i])
            forward_euler = error_state[:, i] + self.cfg.step_size * error_state_dot
            prog.AddLinearConstraint(eq(error_state[:, i + 1], forward_euler))

        # Cost
        terminal_cost = error_state[:, -1].T.dot(Q_N).dot(error_state[:, -1])
        state_running_cost = sum(
            [error_state[:, i].T.dot(Q).dot(error_state[:, i]) for i in range(N)]
        )
        input_running_cost = sum(
            [error_control[:, i].T.dot(R).dot(error_control[:, i]) for i in range(N)]
        )
        prog.AddCost(terminal_cost + state_running_cost + input_running_cost)

        # error_state = state - desired_state
        state = error_state + np.vstack(desired_state).T
        control = error_control + np.vstack(desired_control).T

        return prog, state, control  # type: ignore

    def CalcControl(self, context: Context, output: BasicVector):
        state = self.state_port.Eval(context)
        desired_state_traj = self.desired_state_port.Eval(context)
        desired_control_traj = self.desired_control_port.Eval(context)
        self._setup_QP(state, desired_state_traj, desired_control_traj)  # type: ignore

        result = Solve(self.prog)
        assert result.is_success()

        u_bar_next = result.GetSolution(self.u[:, 0])
        u_next = desired_control_traj[0] + u_bar_next  # type: ignore

        output.SetFromVector(u_next)  # type: ignore
        # output.SetFromVector(result.GetSolution(self.u[:, 0]))  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_control_port(self) -> InputPort:
        return self.desired_control_port
