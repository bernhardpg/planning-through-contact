import numpy as np
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pyparsing import Path

from planning_through_contact.simulation.hardware.hardware_interface import (
    ManipulationHardwareInterface,
)


class PlanarPushingPositionControl:
    """
    An LCM node that implements an IK position controller for the end-effector of the Iiwa,
    following a pre-computed trajectory.
    """

    def __init__(self):
        builder = DiagramBuilder()

        self.station = builder.AddSystem(ManipulationHardwareInterface())
        self.station.Connect()

        self._load_robot()
        self._add_ik_system(builder)

        # TODO(bernhardpg): Make traj publisher
        self._connect_traj_publisher(builder)

        diagram = builder.Build()
        self.simulator = Simulator(diagram)

        # This is important to avoid duplicate publishes to the hardware interface:
        self.simulator.set_publish_every_time_step(False)

        self.simulator.set_target_realtime_rate(1.0)

        station_context = diagram.GetMutableSubsystemContext(
            self.station, self.simulator.get_mutable_context()
        )

        # TODO(bernhardpg): Do we want to compute a feedforward torque?
        self.station.GetInputPort("iiwa_feedforward_torque").FixValue(
            station_context, np.zeros(7)
        )

        # If the diagram is only the hardware interface, then we must advance it a
        # little bit so that first LCM messages get processed. A simulated plant is
        # already publishing correct positions even without advancing, and indeed
        # we must not advance a simulated plant until the sliders and filters have
        # been initialized to match the plant.
        self.simulator.AdvanceTo(1e-6)

        self._initialize_ik()

    def _load_robot(self) -> None:
        self.robot = MultibodyPlant(1e-3)
        parser = Parser(self.robot)
        models_folder = Path(__file__).parents[1] / "models"
        parser.package_map().PopulateFromFolder(str(models_folder))

        # Load the controller plant, i.e. the plant without the box
        controller_plant_file = "iiwa_controller_plant.yaml"
        directives = LoadModelDirectives(str(models_folder / controller_plant_file))
        ProcessModelDirectives(directives, self.robot, parser)  # type: ignore
        self.robot.Finalize()

    def _add_ik_system(self, builder: DiagramBuilder) -> None:
        ik_params = DifferentialInverseKinematicsParameters(
            self.robot.num_positions(), self.robot.num_velocities()
        )

        time_step = 0.005
        ik_params.set_time_step(time_step)
        # True velocity limits for the IIWA14 (in rad, rounded down to the first
        # decimal)
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        velocity_limit_factor = 1.0
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * iiwa14_velocity_limits,
                velocity_limit_factor * iiwa14_velocity_limits,
            )
        )

        self.differential_ik = builder.AddSystem(
            DifferentialInverseKinematicsIntegrator(
                self.robot,
                self.robot.GetFrameByName("iiwa_link_7"),
                time_step,
                ik_params,
            )
        )

        builder.Connect(
            self.differential_ik.GetOutputPort("joint_positions"),
            self.station.GetInputPort("iiwa_position"),
        )

    def _connect_traj_publisher(self, builder: DiagramBuilder) -> None:
        # TODO(bernhardpg)
        builder.Connect(
            to_pose.get_output_port(), self.differential_ik.GetInputPort("X_WE_desired")
        )

    def _initialize_ik(self) -> None:
        q0 = self.station.GetOutputPort("iiwa_position_measured").Eval(station_context)
        self.differential_ik.get_mutable_parameters().set_nominal_joint_position(q0)
        self.differential_ik.SetPositions(
            self.differential_ik.GetMyMutableContextFromRoot(
                self.simulator.get_mutable_context()
            ),
            q0,
        )

    def run(self, end_time: float = 1e8) -> None:
        self.simulator.AdvanceTo(end_time)
