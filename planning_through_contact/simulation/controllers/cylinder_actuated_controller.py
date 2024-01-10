from abc import abstractmethod
from typing import Dict, Any

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    System,
    Diagram,
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)

from .position_controller_base import PositionControllerBase
from planning_through_contact.simulation.sim_utils import GetParser


class CylinderActuatedController(PositionControllerBase):
    """Base controller class for an actuated floating cylinder robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
    ):
        self._sim_config = sim_config
        self._meshcat = None
        self._pid_gains = dict(kp=800, ki=100, kd=50)
        self._num_positions = 2  # Number of dimensions for robot position

    def AddToBuilder(
        self,
        builder: DiagramBuilder,
        state_estimator: Diagram,
        station_plant: MultibodyPlant,
    ) -> System:
        """Note: could bundle everything up into a diagram, but for now actually
        returning the desired_state_source with its inputs unconnected."""
        if self._meshcat is None:
            raise RuntimeError(
                "Need to call `add_meshcat` before calling `setup` of the teleop controller."
            )

        robot_controller_plant = MultibodyPlant(time_step=self._sim_config.time_step)
        parser = GetParser(robot_controller_plant)
        parser.AddModelsFromUrl(
            "package://planning_through_contact/pusher_floating_hydroelastic_actuated.sdf"
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        robot_controller = builder.AddSystem(
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._pid_gains["kp"]] * self._num_positions,
                ki=[self._pid_gains["ki"]] * self._num_positions,
                kd=[self._pid_gains["kd"]] * self._num_positions,
                has_reference_acceleration=False,
            )
        )
        robot_controller.set_name("robot_controller")

        robot_model_instance_name = "pusher"
        robot_model_instance = station_plant.GetModelInstanceByName(
            robot_model_instance_name
        )
        builder.Connect(
            state_estimator.GetOutputPort(f"{robot_model_instance_name}_state"),
            robot_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            robot_controller.get_output_port_control(),
            station_plant.get_actuation_input_port(robot_model_instance),
        )

        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("desired_state_source")
        builder.Connect(
            desired_state_source.get_output_port(),
            robot_controller.get_input_port_desired_state(),
        )

        return desired_state_source
