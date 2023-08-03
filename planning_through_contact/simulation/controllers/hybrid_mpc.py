from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydrake.common.value import Value
from pydrake.systems.framework import (
    BasicVector,
    Context,
    InputPort,
    LeafSystem,
    OutputPort,
    System,
)
from pydrake.systems.primitives import Linearize, LinearSystem


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
        self.model_context = model.CreateDefaultContext()
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
    ) -> LinearSystem:
        self.model_context.SetContinuousState(state)
        self.model.get_input_port().FixValue(self.model_context, np.array([0, 0, 0]))
        return Linearize(self.model, self.model_context)

    def _setup_QP(self) -> None:
        Q = np.eye(self.num_states)
        R = np.eye(self.num_inputs)

        state_desired = []
        input_desired = []

        lin_systems = [
            self._get_linear_system(state, input)
            for state, input in zip(state_desired, input_desired)
        ]
        As = [sys.A() for sys in lin_systems]
        Bs = [sys.B() for sys in lin_systems]

    def CalcControl(self, context: Context, output: BasicVector):
        curr_t = context.get_time()

        desired_state_traj = self.desired_state_port.Eval(context)
        desired_control_traj = self.desired_control_port.Eval(context)

        breakpoint()

        # TODO(bernhardpg): we need the entire desired trajectory, not just one point

        output.SetFromVector(desired_control)  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_control_port(self) -> InputPort:
        return self.desired_control_port
