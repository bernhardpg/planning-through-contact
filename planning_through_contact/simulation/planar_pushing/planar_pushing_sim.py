import numpy as np
import numpy.typing as npt
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, InputPort
from pydrake.systems.primitives import ConstantVectorSource

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingDiagram,
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_to_joint_pos import (
    PusherPoseToJointPos,
)
from planning_through_contact.simulation.systems.pusher_pose_publisher import (
    PusherPosePublisher,
)


class PlanarPushingSimulation:
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        config: PlanarPushingSimConfig = PlanarPushingSimConfig(),
        delay_before_execution: float = 8.0,
    ):
        self.TABLE_BUFFER_DIST = 0.05

        builder = DiagramBuilder()
        self.station = builder.AddNamedSystem(
            "PlanarPushingDiagram",
            PlanarPushingDiagram(add_visualizer=True, config=config),
        )

        # TODO(bernhardpg): Do we want to compute a feedforward torque?
        constant_source = builder.AddNamedSystem(
            "const", ConstantVectorSource(np.zeros(7))
        )
        builder.Connect(
            constant_source.get_output_port(),
            self.station.GetInputPort("iiwa_feedforward_torque"),
        )

        self.pusher_pose_to_joint_pos = PusherPoseToJointPos.add_to_builder(
            builder,
            self.station.GetInputPort("iiwa_position"),
        )

        self.pusher_pose_pub = PusherPosePublisher.add_to_builder(
            builder,
            traj,
            delay_before_execution,
            self.pusher_pose_to_joint_pos.get_input_port(),
        )

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_mutable_context()
        self.mbp_context = self.station.mbp.GetMyContextFromRoot(self.context)

        self._set_joint_positions(config.default_joint_positions)
        self.set_slider_planar_pose(config.start_pose)

        self.config = config

        q0 = self.station.GetOutputPort("iiwa_position_measured").Eval(
            self.station.GetMyContextFromRoot(self.context)
        )
        self.pusher_pose_to_joint_pos.init_diff_ik(q0, self.context)

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_png(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def reset(self) -> None:
        self.simulator.Initialize()

    def run(self, timeout=1e8):
        self.simulator.AdvanceTo(timeout)

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = min([shape.height() for shape in self.station.get_slider_shapes()])

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self.station.mbp.SetPositions(self.mbp_context, self.station.slider, q)

    # TODO(bernhardpg): This will not work on the real system!
    def set_pusher_planar_pose(
        self, planar_pose: PlanarPose, disregard_angle: bool = True
    ):
        """
        Sets the planar pose of the pusher.

        @param planar_pose: Desired end-effector planar pose.
        @param disregard_angle: Whether or not to enforce the z-axis rotation specified by the planar_pose.
        """

        ik = InverseKinematics(self.station.mbp, self.mbp_context)
        pusher_shape = self.station.get_pusher_shape()
        pose = planar_pose.to_pose(
            z_value=pusher_shape.length() + self.TABLE_BUFFER_DIST
        )

        ik.AddPositionConstraint(
            self.station.pusher_frame,
            np.zeros(3),
            self.station.mbp.world_frame(),
            pose.translation(),
            pose.translation(),
        )

        if disregard_angle:
            z_unit_vec = np.array([0, 0, 1])
            ik.AddAngleBetweenVectorsConstraint(
                self.station.pusher_frame,
                z_unit_vec,
                self.station.mbp.world_frame(),
                -z_unit_vec,  # The pusher object has z-axis pointing up
                0,
                0,
            )

        else:
            ik.AddOrientationConstraint(
                self.station.pusher_frame,
                RotationMatrix(),
                self.station.mbp.world_frame(),
                pose.rotation(),
                0.0,
            )

        # Non-penetration
        ik.AddMinimumDistanceConstraint(0.001, 0.1)

        # Cost on deviation from default joint positions
        prog = ik.get_mutable_prog()
        q = ik.q()

        slider_position = self.station.mbp.GetPositions(
            self.mbp_context, self.station.slider
        )
        q0 = np.concatenate([self.config.default_joint_positions, slider_position])
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)

        # Will automatically set the positions of the objects
        Solve(ik.prog())

    def _set_joint_positions(self, joint_positions: npt.NDArray[np.float64]):
        self.station.mbp.SetPositions(
            self.mbp_context, self.station.iiwa, joint_positions
        )
