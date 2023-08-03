from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
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
        self, state: npt.NDArray[np.float64], input: npt.NDArray[np.float64]
    ) -> AffineSystem:
        model_context = self.model.CreateDefaultContext()
        model_context.SetContinuousState(state)
        self.model.get_input_port().FixValue(model_context, input)
        return FirstOrderTaylorApproximation(self.model, model_context)

    def _setup_QP(
        self,
        current_state: npt.NDArray[np.float64],
        desired_state: List[npt.NDArray[np.float64]],
        desired_control: List[npt.NDArray[np.float64]],
    ) -> None:
        N = self.cfg.horizon
        Q = np.eye(self.num_states)
        Q_N = np.eye(self.num_states)
        R = np.eye(self.num_inputs)

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.num_states, N, "x")
        u = prog.NewContinuousVariables(self.num_inputs, N - 1, "u")

        # Initial value constraint
        prog.AddLinearConstraint(eq(x[:, 0], current_state))

        # Dynamic constraints
        lin_systems = [
            self._get_linear_system(state, control)
            for state, control in zip(desired_state, desired_control)
        ][
            :-1
        ]  # only need N-1 linear systems
        As = [sys.A() for sys in lin_systems]
        Bs = [sys.B() for sys in lin_systems]
        f0s = [sys.f0() for sys in lin_systems]
        for i, (A, B, f0) in enumerate(zip(As, Bs, f0s)):
            x_dot = A.dot(x[:, i]) + B.dot(u[:, i]) + f0
            forward_euler = x[:, i] + self.cfg.step_size * x_dot
            prog.AddLinearConstraint(eq(x[:, i + 1], forward_euler))

        # Cost
        terminal_cost = x[:, -1].T.dot(Q_N).dot(x[:, -1])
        state_running_cost = sum([x[:, i].T.dot(Q).dot(x[:, i]) for i in range(N - 1)])
        input_running_cost = sum([u[:, i].T.dot(R).dot(u[:, i]) for i in range(N - 1)])
        prog.AddCost(terminal_cost + state_running_cost + input_running_cost)

        self.prog = prog
        self.x = x
        self.u = u

    def CalcControl(self, context: Context, output: BasicVector):
        state = self.state_port.Eval(context)
        desired_state_traj = self.desired_state_port.Eval(context)
        desired_control_traj = self.desired_control_port.Eval(context)
        self._setup_QP(state, desired_state_traj, desired_control_traj)  # type: ignore

        result = Solve(self.prog)
        assert result.is_success()

        u_next = result.GetSolution(self.u[:, 0])

        output.SetFromVector(u_next)  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_control_port(self) -> InputPort:
        return self.desired_control_port
