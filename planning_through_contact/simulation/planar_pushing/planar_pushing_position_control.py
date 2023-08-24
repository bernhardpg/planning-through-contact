import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
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
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem
from pyparsing import Path

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.hardware.hardware_interface import (
    ManipulationHardwareInterface,
)


class PusherPosePublisher(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        z_dist_to_table: float = 0.5,
        delay_before_start: float = 10,
    ):
        super().__init__()
        self.traj = traj
        self.z_dist = z_dist_to_table
        self.delay = delay_before_start

        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    def _calc_pose(self, t: float) -> RigidTransform:
        p_c_W = self.traj.get_value(t - self.delay, "p_c_W")
        assert isinstance(p_c_W, type(np.array([])))

        planar_pose = PlanarPose(p_c_W[0].item(), p_c_W[1].item(), theta=0)
        return planar_pose.to_pose(object_height=self.z_dist)

    def DoCalcOutput(self, context: Context, output):
        curr_t = context.get_time()
        end_effector_pose = self._calc_pose(curr_t)
        print(end_effector_pose)
        output.set_value(end_effector_pose)


class PlanarPushingPositionControl:
    """
    An LCM node that implements an IK position controller for the end-effector of the Iiwa,
    following a pre-computed trajectory.
    """

    def __init__(self, traj: PlanarPushingTrajectory):
        builder = DiagramBuilder()

        self.hardware_interface = builder.AddNamedSystem(
            "hardware_interface",
            ManipulationHardwareInterface(),
        )
        self.hardware_interface.Connect()

        self._load_robot()
        self._add_ik_system(builder)

        self._add_pose_publisher(builder, traj)

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)

        # This is important to avoid duplicate publishes to the hardware interface:
        self.simulator.set_publish_every_time_step(False)

        self.simulator.set_target_realtime_rate(1.0)

        self.hardware_interface_context = self.diagram.GetMutableSubsystemContext(
            self.hardware_interface, self.simulator.get_mutable_context()
        )

        # TODO(bernhardpg): Do we want to compute a feedforward torque?
        self.hardware_interface.GetInputPort("iiwa_feedforward_torque").FixValue(
            self.hardware_interface_context, np.zeros(7)
        )

        # If the diagram is only the hardware interface, then we must advance it a
        # little bit so that first LCM messages get processed. A simulated plant is
        # already publishing correct positions even without advancing, and indeed
        # we must not advance a simulated plant until the sliders and filters have
        # been initialized to match the plant.
        real_hardware = True  # TODO(bernhardpg)
        if real_hardware:
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
        # True velocity limits for the IIWA14
        # (in rad, rounded down to the first decimal)
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        velocity_limit_factor = 1.0
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * iiwa14_velocity_limits,
                velocity_limit_factor * iiwa14_velocity_limits,
            )
        )

        self.differential_ik = builder.AddNamedSystem(
            "diff_ik",
            DifferentialInverseKinematicsIntegrator(
                self.robot,
                self.robot.GetFrameByName("iiwa_link_7"),
                time_step,
                ik_params,
            ),
        )

        builder.Connect(
            self.differential_ik.GetOutputPort("joint_positions"),
            self.hardware_interface.GetInputPort("iiwa_position"),
        )

    def _add_pose_publisher(
        self, builder: DiagramBuilder, traj: PlanarPushingTrajectory
    ) -> None:
        # TODO(bernhardpg): Should not hardcode this
        PUSHER_HEIGHT = 0.15
        BUFFER = 0.05  # TODO(bernhardpg): Turn down
        self.pose_publisher = builder.AddNamedSystem(
            "pusher_pose_publisher",
            PusherPosePublisher(traj, PUSHER_HEIGHT + BUFFER),
        )
        builder.Connect(
            self.pose_publisher.get_output_port(),
            self.differential_ik.GetInputPort("X_WE_desired"),
        )

    def _initialize_ik(self) -> None:
        q0 = self.hardware_interface.GetOutputPort("iiwa_position_measured").Eval(
            self.hardware_interface_context
        )
        self.differential_ik.get_mutable_parameters().set_nominal_joint_position(q0)
        self.differential_ik.SetPositions(
            self.differential_ik.GetMyMutableContextFromRoot(
                self.simulator.get_mutable_context()
            ),
            q0,
        )

    def run(self, end_time: float = 1e8) -> None:
        self.simulator.AdvanceTo(end_time)

    def export_diagram(self, filename: str):
        import pydot

        gviz_string = self.diagram.GetGraphvizString()
        pydot.graph_from_dot_data(gviz_string)[0].write_png(filename)  # type: ignore
        print(f"Saved diagram to: {filename}")
