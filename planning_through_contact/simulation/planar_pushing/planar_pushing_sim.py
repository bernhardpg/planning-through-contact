from typing import Optional

import numpy as np
import numpy.typing as npt
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.all import LogVectorOutput

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import (
    PlanarTranslationToRigidTransformSystem,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
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

from planning_through_contact.visualize.analysis import (
    plot_planar_pushing_logs_from_pose_vectors,
)


class PlanarPushingSimulation:
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        sim_config: PlanarPushingSimConfig,
    ):
        self.TABLE_BUFFER_DIST = 0.05

        builder = DiagramBuilder()
        self.station = builder.AddNamedSystem(
            "PlanarPushingDiagram",
            PlanarPushingDiagram(add_visualizer=True, sim_config=sim_config),
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
                traj, sim_config.mpc_config, sim_config.delay_before_execution
            ),
        )

        if sim_config.use_diff_ik:
            self.pusher_pose_to_joint_pos = PusherPoseToJointPosDiffIk.add_to_builder(
                builder,
                self.station.GetInputPort("iiwa_position"),
                self.station.GetOutputPort("iiwa_state_measured"),
                time_step=sim_config.time_step,
                use_diff_ik_feedback=False,
            )
            planar_translation_to_rigid_tranform = builder.AddSystem(
                PlanarTranslationToRigidTransformSystem()
            )
            builder.Connect(
                planar_translation_to_rigid_tranform.get_output_port(),
                self.pusher_pose_to_joint_pos.get_pose_input_port(),
            )
        else:
            # TODO: outdated, fix
            ik = PusherPoseInverseKinematics.AddTobuilder(
                builder,
                self.pusher_pose_controller.get_output_port(),
                self.station.GetOutputPort("iiwa_position_measured"),
                self.station.GetOutputPort("slider_pose"),
                self.station.GetInputPort("iiwa_position"),
                sim_config.default_joint_positions,
            )

        self.pusher_pose_controller = PusherPoseController.AddToBuilder(
            builder,
            sim_config.dynamics_config,
            sim_config.mpc_config,
            self.planar_pose_pub.GetOutputPort("contact_mode_traj"),
            self.planar_pose_pub.GetOutputPort("slider_planar_pose_traj"),
            self.planar_pose_pub.GetOutputPort("pusher_planar_pose_traj"),
            self.planar_pose_pub.GetOutputPort("contact_force_traj"),
            planar_translation_to_rigid_tranform.get_input_port(),
            closed_loop=sim_config.closed_loop,
            pusher_planar_pose_measured=self.station.GetOutputPort("pusher_pose"),
            slider_pose_measured=self.station.GetOutputPort("slider_pose"),
        )

        if sim_config.save_plots:
            # state logger
            pusher_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self.station.GetOutputPort("pusher_pose"),
                pusher_pose_to_vector.get_input_port(),
            )
            pusher_pose_logger = LogVectorOutput(
                pusher_pose_to_vector.get_output_port(), builder
            )

            slider_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self.station.GetOutputPort("slider_pose"),
                slider_pose_to_vector.get_input_port(),
            )
            slider_pose_logger = LogVectorOutput(
                slider_pose_to_vector.get_output_port(), builder
            )
            pusher_pose_desired_logger = LogVectorOutput(
                self.planar_pose_pub.GetOutputPort("desired_pusher_planar_pose_vector"),
                builder,
            )
            slider_pose_desired_logger = LogVectorOutput(
                self.planar_pose_pub.GetOutputPort("desired_slider_planar_pose_vector"),
                builder,
            )

            self._pusher_pose_logger = pusher_pose_logger
            self._slider_pose_logger = slider_pose_logger
            self._pusher_pose_desired_logger = pusher_pose_desired_logger
            self._slider_pose_desired_logger = slider_pose_desired_logger

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        if sim_config.use_realtime:
            self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_mutable_context()
        self.mbp_context = self.station.mbp.GetMyContextFromRoot(self.context)

        self.config = sim_config
        self.set_slider_planar_pose(sim_config.slider_start_pose)

        desired_pusher_pose = sim_config.pusher_start_pose.to_pose(
            self.TABLE_BUFFER_DIST
        )
        desired_slider_pose = sim_config.slider_start_pose.to_pose(
            self.station.get_slider_min_height()
        )
        start_joint_positions = solve_ik(
            self.diagram,
            self.station,
            pose=desired_pusher_pose,
            current_slider_pose=desired_slider_pose,
            default_joint_positions=sim_config.default_joint_positions,
        )
        self._set_joint_positions(start_joint_positions)

        if not sim_config.use_diff_ik:
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
            self.station.meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self.station.meshcat.PublishRecording()
            res = self.station.meshcat.StaticHtml()
            with open(save_recording_as, "w") as f:
                f.write(res)
        if self.config.save_plots:
            pusher_pose_log = self._pusher_pose_logger.FindLog(self.context)
            slider_pose_log = self._slider_pose_logger.FindLog(self.context)
            pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
                self.context
            )
            slider_pose_desired_log = self._slider_pose_desired_logger.FindLog(
                self.context
            )
            plot_planar_pushing_logs_from_pose_vectors(
                pusher_pose_log,
                slider_pose_log,
                pusher_pose_desired_log,
                slider_pose_desired_log,
            )

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = min([shape.height() for shape in self.station.get_slider_shapes()])

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self.station.mbp.SetPositions(self.mbp_context, self.station.slider, q)

    def _set_joint_positions(self, joint_positions: npt.NDArray[np.float64]):
        self.station.mbp.SetPositions(
            self.mbp_context, self.station.iiwa, joint_positions
        )
