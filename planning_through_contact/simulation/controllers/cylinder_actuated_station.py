import numpy as np

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RollPitchYaw,
    AddDefaultVisualization,
    Meshcat,
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

from .robot_system_base import RobotSystemBase
from planning_through_contact.simulation.sim_utils import (
    GetParser,
    AddSliderAndConfigureContact,
)


class CylinderActuatedStation(RobotSystemBase):
    """Base controller class for an actuated floating cylinder robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        self._pid_gains = dict(kp=800, ki=100, kd=50)
        self._num_positions = 2  # Number of dimensions for robot position

        builder = DiagramBuilder()

        # "Internal" plant for the robot controller
        robot_controller_plant = MultibodyPlant(time_step=self._sim_config.time_step)
        parser = GetParser(robot_controller_plant)
        parser.AddModelsFromUrl(
            "package://planning_through_contact/pusher_floating_hydroelastic_actuated.sdf"
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        # "External" station plant
        self.station_plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._sim_config.time_step
        )
        self.slider = AddSliderAndConfigureContact(
            sim_config, self.station_plant, self._scene_graph
        )

        self._meshcat.SetTransform(
            path="/Cameras/default",
            matrix=RigidTransform(
                RollPitchYaw([0.0, 0.0, np.pi / 2]),  # type: ignore
                np.array([1, 0, 0]),
            ).GetAsMatrix4(),
        )
        AddDefaultVisualization(builder, self._meshcat)

        ## Add Leaf systems

        robot_controller = builder.AddNamedSystem(
            "RobotController",
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._pid_gains["kp"]] * self._num_positions,
                ki=[self._pid_gains["ki"]] * self._num_positions,
                kd=[self._pid_gains["kd"]] * self._num_positions,
                has_reference_acceleration=False,
            ),
        )

        # Add system to convert desired position to desired position and velocity.
        desired_state_source = builder.AddNamedSystem(
            "DesiredStateSource",
            StateInterpolatorWithDiscreteDerivative(
                self._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            ),
        )

        ## Connect systems

        self._robot_model_instance = self.station_plant.GetModelInstanceByName(
            self.robot_model_name
        )
        builder.Connect(
            robot_controller.get_output_port_control(),
            self.station_plant.get_actuation_input_port(self._robot_model_instance),
        )

        builder.Connect(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            robot_controller.get_input_port_estimated_state(),
        )

        builder.Connect(
            desired_state_source.get_output_port(),
            robot_controller.get_input_port_desired_state(),
        )

        ## Export inputs and outputs

        builder.ExportInput(
            desired_state_source.get_input_port(),
            "planar_position_command",
        )

        builder.ExportOutput(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            "robot_state_measured",
        )

        # Only relevant when use_hardware=False
        # If use_hardware=True, this info will be updated by the optitrack system in the state estimator directly
        builder.ExportOutput(
            self.station_plant.get_state_output_port(self.slider),
            "object_state_measured",
        )

        builder.BuildInto(self)

        ## Set default position for the robot
        self.station_plant.SetDefaultPositions(
            self._robot_model_instance, self._sim_config.pusher_start_pose.pos()
        )

    @property
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        return "pusher"
