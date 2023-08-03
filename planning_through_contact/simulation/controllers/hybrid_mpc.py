import numpy as np
import numpy.typing as npt
from pydrake.systems.framework import (
    BasicVector,
    Context,
    InputPort,
    LeafSystem,
    OutputPort,
    System,
)
from pydrake.systems.primitives import Linearize, LinearSystem


class HybridModelPredictiveControl(LeafSystem):
    def __init__(self, model: System) -> None:
        super().__init__()

        self.model = model
        self.model_context = model.CreateDefaultContext()

        self.state_port = self.DeclareVectorInputPort(
            "state", model.num_continuous_states()
        )

        self.desired_state_port = self.DeclareVectorInputPort(
            "desired_state", model.num_continuous_states()
        )

        assert model.num_input_ports() == 1
        self.control_port = self.DeclareVectorOutputPort(
            "control", model.get_input_port().size(), self.CalcControl
        )
        self.desired_input_port = self.DeclareVectorInputPort(
            "desired_inpout", model.get_input_port().size()
        )

    def _get_linear_system(
        self, state: npt.NDArray[np.float64], input: npt.NDArray[np.float64]
    ) -> LinearSystem:
        self.model_context.SetContinuousState(state)
        self.model.get_input_port().FixValue(self.model_context, np.array([0, 0, 0]))
        return Linearize(self.model, self.model_context)

    def CalcControl(self, context: Context, output: BasicVector):
        desired_input = self.desired_input_port.Eval(context)
        desired_state = self.desired_state_port.Eval(context)
        output.SetFromVector(desired_input)  # type: ignore

    def get_control_port(self) -> OutputPort:
        return self.control_port

    def get_state_port(self) -> InputPort:
        return self.state_port

    def get_desired_state_port(self) -> InputPort:
        return self.desired_state_port

    def get_desired_input_port(self) -> InputPort:
        return self.desired_input_port
