import lcm
import numpy as np
import numpy.typing as npt
from drake import lcmt_iiwa_command, lcmt_iiwa_status
from pydrake.all import AbstractValue, LeafSystem, Quaternion
from pydrake.lcm import DrakeLcm
from pydrake.manipulation import IiwaCommandReceiver, IiwaStatusSender
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import (
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingDiagram,
    PlanarPushingSimConfig,
)


# TODO: remove
def get_keypoints(x):
    """
    Get keypoints given pose x.
    Optionally accepts whether to visualize, the path to meshcat,
    and injected noise.
    """
    canon_points = np.array(
        [
            [1, -1, -1, 1, 0],
            [1, 1, -1, -1, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    # TODO: Remove
    box_dim = np.array([0.0867, 0.1703, 0.0391])
    # These dimensions come from the ycb dataset on 004_sugar_box.sdf
    # Make homogeneous coordinates
    keypoints = np.zeros((4, 5))
    keypoints[0, :] = box_dim[0] * canon_points[0, :] / 2
    keypoints[1, :] = box_dim[1] * canon_points[1, :] / 2
    keypoints[2, :] = box_dim[2]
    keypoints[3, :] = 1

    # transform according to pose x.
    X_WB = RigidTransform(
        RollPitchYaw(np.array([0.0, 0.0, x[2]])), np.array([x[0], x[1], 0.0])  # type: ignore
    ).GetAsMatrix4()

    X_WK = X_WB @ keypoints
    X_WK[0:2] = X_WK[0:2]
    return X_WK[0:2, :]


class KeyptsLCM(LeafSystem):
    def __init__(self, slider_index):
        LeafSystem.__init__(self)
        # self.Abstract
        self.slider_index = slider_index
        self.lc = lcm.LCM()
        self.input_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclarePeriodicPublishEvent(1.0 / 200.0, 0.0, self.publish)

    def publish(self, context):
        X_WB = self.get_input_port().Eval(context)[self.slider_index]  # type: ignore
        q_wxyz = Quaternion(X_WB.rotation().matrix()).wxyz()
        p_xyz = X_WB.translation()
        qp_WB = np.concatenate((q_wxyz, p_xyz))
        planar_pose = PlanarPose.from_generalized_coords(qp_WB).vector()

        # TODO: Fix keypoint handling with octotrack

        # keypts = get_keypoints(xyt_WB)
        # keypts = np.concatenate((keypts, np.zeros((1, 5))))
        #
        # sub_msg = optitrack.optitrack_marker_set_t()
        # sub_msg.num_markers = 5
        # sub_msg.xyz = keypts.T
        # msg = optitrack.optitrack_frame_t()
        # msg.utime = int(time.time() * 1e6)
        # msg.num_marker_sets = 1
        # msg.marker_sets = [sub_msg]
        # self.lc.publish("KEYPTS", msg.encode())


class PlanarPushingMockSimulation:
    """
    Planar pushing dynamical system, implemented in Drake.
    x: position of box and pusher, [x_box, y_box, theta_box, x_pusher, y_pusher]
    u: delta position command on the pusher.
    """

    def __init__(self, config: PlanarPushingSimConfig = PlanarPushingSimConfig()):
        self.TABLE_BUFFER_DIST = 0.05

        builder = DiagramBuilder()
        self.station = builder.AddSystem(
            PlanarPushingDiagram(add_visualizer=True, config=config)
        )

        self.connect_lcm(builder, self.station)
        self.keypts_lcm = builder.AddSystem(
            KeyptsLCM(self.station.get_slider_body().index())
        )
        builder.Connect(
            self.station.GetOutputPort("body_poses"), self.keypts_lcm.get_input_port()
        )

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_mutable_context()
        self.mbp_context = self.station.mbp.GetMyContextFromRoot(self.context)

        self._set_joint_positions(config.default_joint_positions)
        self.set_slider_planar_pose(config.start_pose)

        self.config = config

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

    def connect_lcm(self, builder, station):
        # Set up LCM publisher subscribers.
        lcm = DrakeLcm()
        lcm_system = builder.AddSystem(LcmInterfaceSystem(lcm))
        iiwa_command = builder.AddSystem(IiwaCommandReceiver())
        iiwa_command_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(  # type: ignore
                channel="IIWA_COMMAND",
                lcm_type=lcmt_iiwa_command,
                lcm=lcm,
                use_cpp_serializer=True,
            )
        )
        builder.Connect(
            iiwa_command_subscriber.get_output_port(),
            iiwa_command.get_message_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_position_measured"),
            iiwa_command.get_position_measured_input_port(),
        )
        builder.Connect(
            iiwa_command.get_commanded_position_output_port(),
            station.GetInputPort("iiwa_position"),
        )
        builder.Connect(
            iiwa_command.get_commanded_torque_output_port(),
            station.GetInputPort("iiwa_feedforward_torque"),
        )

        iiwa_status = builder.AddSystem(IiwaStatusSender())
        builder.Connect(
            station.GetOutputPort("iiwa_position_commanded"),
            iiwa_status.get_position_commanded_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_position_measured"),
            iiwa_status.get_position_measured_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_velocity_estimated"),
            iiwa_status.get_velocity_estimated_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_commanded"),
            iiwa_status.get_torque_commanded_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_measured"),
            iiwa_status.get_torque_measured_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_external"),
            iiwa_status.get_torque_external_input_port(),
        )

        iiwa_status_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                "IIWA_STATUS",  # type: ignore
                lcm_type=lcmt_iiwa_status,
                lcm=lcm,
                publish_period=0.005,
                use_cpp_serializer=True,
            )
        )
        builder.Connect(
            iiwa_status.get_output_port(), iiwa_status_publisher.get_input_port()
        )

    def get_slider_planar_pose(self):
        return self.station.get_slider_planar_pose(self.mbp_context)

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = min([shape.height() for shape in self.station.get_slider_shapes()])

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self.station.mbp.SetPositions(self.mbp_context, self.station.slider, q)

    def get_pusher_planar_pose(self):
        return self.station.get_pusher_planar_pose(self.mbp_context)

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
