from typing import Optional

import numpy as np
import numpy.typing as npt
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.planar_pushing.planar_pose_traj_publisher import (
    PlanarPoseTrajPublisher,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingDiagram,
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_controller import (
    PusherPoseController,
)
from planning_through_contact.simulation.planar_pushing.pusher_pose_to_joint_pos import (
    PusherPoseInverseKinematics,
    PusherPoseToJointPosDiffIk,
    solve_ik,
)


class PlanarPushingSimulation:
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        slider: RigidBody,
        config: PlanarPushingSimConfig = PlanarPushingSimConfig(),
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

        self.planar_pose_pub = builder.AddNamedSystem(
            "PlanarPoseTrajPublisher",
            PlanarPoseTrajPublisher(
                traj, config.mpc_config, config.delay_before_execution
            ),
        )

        if config.use_diff_ik:
            self.pusher_pose_to_joint_pos = PusherPoseToJointPosDiffIk.add_to_builder(
                builder,
                self.station.GetInputPort("iiwa_position"),
                self.station.GetOutputPort("iiwa_state_measured"),
                time_step=config.time_step,
                use_diff_ik_feedback=False,
            )
        else:
            # TODO: outdated, fix
            ik = PusherPoseInverseKinematics.AddTobuilder(
                builder,
                self.pusher_pose_controller.get_output_port(),
                self.station.GetOutputPort("iiwa_position_measured"),
                self.station.GetOutputPort("slider_pose"),
                self.station.GetInputPort("iiwa_position"),
                config.default_joint_positions,
            )

        self.pusher_pose_controller = PusherPoseController.AddToBuilder(
            builder,
            slider,
            self.planar_pose_pub.GetOutputPort("contact_mode"),
            self.planar_pose_pub.GetOutputPort("slider_planar_pose_traj"),
            self.planar_pose_pub.GetOutputPort("pusher_planar_pose_traj"),
            self.station.GetOutputPort("pusher_pose"),
            self.station.GetOutputPort("slider_pose"),
            self.pusher_pose_to_joint_pos.get_pose_input_port(),
        )

        if config.save_plots:
            builder.AddNamedSystem("theta_source")

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        if config.use_realtime:
            self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_mutable_context()
        self.mbp_context = self.station.mbp.GetMyContextFromRoot(self.context)

        self.config = config
        self.set_slider_planar_pose(config.slider_start_pose)

        BUFFER = 0.02
        start_joint_positions = solve_ik(
            self.diagram,
            self.station,
            config.pusher_start_pose.to_pose(BUFFER),
            config.slider_start_pose.to_pose(self.station.get_slider_min_height()),
            config.default_joint_positions,
        )
        self._set_joint_positions(start_joint_positions)

        if not config.use_diff_ik:
            ik.init(self.diagram, self.station)

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def reset(self) -> None:
        self.simulator.Initialize()

    def run(self, timeout=1e8, save_recording_as: Optional[str] = None) -> None:
        if save_recording_as:
            self.station.meshcat.StartRecording()
        self.simulator.AdvanceTo(timeout)
        if save_recording_as:
            self.station.meshcat.StopRecording()
            self.station.meshcat.PublishRecording()
            res = self.station.meshcat.StaticHtml()
            with open(save_recording_as, "w") as f:
                f.write(res)

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = min([shape.height() for shape in self.station.get_slider_shapes()])

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self.station.mbp.SetPositions(self.mbp_context, self.station.slider, q)

    def _set_joint_positions(self, joint_positions: npt.NDArray[np.float64]):
        self.station.mbp.SetPositions(
            self.mbp_context, self.station.iiwa, joint_positions
        )
