from pathlib import Path

import lcm
import numpy as np
import numpy.typing as npt
from drake import lcmt_iiwa_command, lcmt_iiwa_status
from pydrake.all import (
    AbstractValue,
    InverseDynamicsController,
    LeafSystem,
    MultibodyPlant,
    Quaternion,
    StateInterpolatorWithDiscreteDerivative,
)
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.lcm import DrakeLcm
from pydrake.manipulation.kuka_iiwa import IiwaCommandReceiver, IiwaStatusSender
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import Frame
from pydrake.multibody.tree import RigidBody as DrakeRigidBody
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context, Diagram, DiagramBuilder
from pydrake.systems.lcm import (
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
)
from pydrake.systems.primitives import Adder, Demultiplexer, PassThrough

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody


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


class PlanarPushingDiagram(Diagram):
    def __init__(self, add_visualizer: bool = False):
        Diagram.__init__(self)

        self.h_mbp = 1e-3

        builder = DiagramBuilder()
        self.mbp, self.sg = AddMultibodyPlantSceneGraph(builder, time_step=self.h_mbp)
        self.parser = Parser(self.mbp, self.sg)

        # Load ROS packages starting from current folder
        # NOTE: Depends on there being a `package.xml` file in the models/ folder in order to load all the models.
        # TODO: Is this really the best way to do something like this?
        self.models_folder = Path(__file__).parents[1] / "models"
        self.parser.package_map().PopulateFromFolder(str(self.models_folder))
        directives = LoadModelDirectives(
            str(self.models_folder / "planar_pushing_iiwa_plant.yaml")
        )
        ProcessModelDirectives(directives, self.mbp, self.parser)  # type: ignore
        self.mbp.Finalize()

        # TODO: rename these
        # Get model instances for box and pusher.
        self.box = self.mbp.GetModelInstanceByName("box")
        self.pusher = self.mbp.GetModelInstanceByName("pusher")
        self.iiwa = self.mbp.GetModelInstanceByName("iiwa")

        if add_visualizer:
            self.meshcat = StartMeshcat()  # type: ignore
            self.meshcat.SetTransform(
                path="/Cameras/default",
                matrix=RigidTransform(
                    RollPitchYaw([-np.pi / 8, 0.0, np.pi / 2]),  # type: ignore
                    0.01 * np.array([0.05, 0.0, 0.1]),
                ).GetAsMatrix4(),
            )
            self.visualizer = MeshcatVisualizer.AddToBuilder(builder, self.sg, self.meshcat)  # type: ignore

        self.add_controller(builder)

        # Export states
        builder.ExportOutput(self.mbp.get_body_poses_output_port(), "body_poses")
        builder.BuildInto(self)

    def add_controller(self, builder):
        self.controller_plant = MultibodyPlant(1e-3)
        parser = Parser(self.controller_plant)
        parser.package_map().PopulateFromFolder(str(self.models_folder))
        directives = LoadModelDirectives(
            str(self.models_folder / "iiwa_controller_plant.yaml")
        )
        ProcessModelDirectives(directives, self.controller_plant, parser)  # type: ignore
        self.controller_plant.Finalize()
        kp = 800 * np.ones(7)
        ki = 100 * np.ones(7)
        kd = 2 * np.sqrt(kp)
        arm_controller = builder.AddSystem(
            InverseDynamicsController(self.controller_plant, kp, ki, kd, False)
        )
        adder = builder.AddSystem(Adder(2, 7))
        state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(7, 1e-3, True)
        )
        arm_command = builder.AddSystem(PassThrough(7))
        state_split = builder.AddSystem(Demultiplexer(14, 7))

        # Export positions command
        builder.ExportInput(arm_command.get_input_port(0), "iiwa_position")
        builder.ExportOutput(arm_command.get_output_port(0), "iiwa_position_commanded")

        # Export arm state ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa), state_split.get_input_port(0)
        )
        builder.ExportOutput(state_split.get_output_port(0), "iiwa_position_measured")
        builder.ExportOutput(state_split.get_output_port(1), "iiwa_velocity_estimated")
        builder.ExportOutput(
            self.mbp.get_state_output_port(self.iiwa), "iiwa_state_measured"
        )

        # Export controller stack ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa),
            arm_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            arm_controller.get_output_port_control(), adder.get_input_port(0)
        )
        builder.Connect(
            adder.get_output_port(0),
            self.mbp.get_actuation_input_port(self.iiwa),
        )
        builder.Connect(
            state_from_position.get_output_port(0),
            arm_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            arm_command.get_output_port(0), state_from_position.get_input_port(0)
        )

        builder.ExportInput(adder.get_input_port(1), "iiwa_feedforward_torque")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_commanded")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_measured")

        builder.ExportOutput(
            self.mbp.get_generalized_contact_forces_output_port(self.iiwa),
            "iiwa_torque_external",
        )

    def get_box_body(self) -> DrakeRigidBody:
        box_body = self.mbp.GetUniqueFreeBaseBodyOrThrow(self.box)
        return box_body

    def get_box_shape(self) -> DrakeBox:
        box_body = self.get_box_body()
        collision_geometries_ids = self.mbp.GetCollisionGeometriesForBody(box_body)

        inspector = self.sg.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]
        box_shape = next(shape for shape in shapes if isinstance(shape, DrakeBox))

        return box_shape

    def get_box(self) -> RigidBody:
        box_body = self.get_box_body()
        box_shape = self.get_box_shape()

        box = RigidBody.from_drake(box_shape, box_body, "box")
        return box

    def get_box_planar_pose(self, context: Context):
        pose = self.mbp.GetFreeBodyPose(context, self.get_box_body())
        planar_pose = PlanarPose.from_pose(pose)
        return planar_pose

    def get_pusher_shape(self) -> DrakeBox:
        pusher_body = self.mbp.GetBodyByName("pusher")
        collision_geometry_id = self.mbp.GetCollisionGeometriesForBody(pusher_body)[0]

        inspector = self.sg.model_inspector()
        pusher_shape = inspector.GetShape(collision_geometry_id)

        return pusher_shape

    @property
    def pusher_frame(self) -> Frame:
        P = self.mbp.GetFrameByName("pusher_base")
        return P

    def get_pusher_planar_pose(self, context: Context):
        pose = self.mbp.CalcRelativeTransform(
            context, self.mbp.world_frame(), self.pusher_frame
        )
        planar_pose = PlanarPose.from_pose(pose)
        return planar_pose


class KeyptsLCM(LeafSystem):
    def __init__(self, box_index):
        LeafSystem.__init__(self)
        # self.Abstract
        self.box_index = box_index
        self.lc = lcm.LCM()
        self.input_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclarePeriodicPublishEvent(1.0 / 200.0, 0.0, self.publish)

    def publish(self, context):
        X_WB = self.get_input_port().Eval(context)[self.box_index]  # type: ignore
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


class PlanarPushingSimulation:
    """
    Planar pushing dynamical system, implemented in Drake.
    x: position of box and pusher, [x_box, y_box, theta_box, x_pusher, y_pusher]
    u: delta position command on the pusher.
    """

    def __init__(self):
        self.DEFAULT_JOINT_POSITIONS = np.array(
            [0.666, 1.039, -0.7714, -2.0497, 1.3031, 0.6729, -1.0252]
        )
        self.DEFAULT_BOX_POSITION = PlanarPose(x=0.5, y=0.0, theta=0.0)

        builder = DiagramBuilder()
        self.station = builder.AddSystem(PlanarPushingDiagram(add_visualizer=True))

        self.connect_lcm(builder, self.station)
        self.keypts_lcm = builder.AddSystem(
            KeyptsLCM(self.station.get_box_body().index())
        )
        builder.Connect(
            self.station.GetOutputPort("body_poses"), self.keypts_lcm.get_input_port()
        )

        self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.context = self.simulator.get_mutable_context()
        self.mbp_context = self.station.mbp.GetMyContextFromRoot(self.context)

        self._set_joint_positions(self.DEFAULT_JOINT_POSITIONS)
        self.set_box_planar_pose(self.DEFAULT_BOX_POSITION)

    def export_diagram(self, target_folder: str):
        import pydot

        filename = target_folder + "/diagram.png"
        pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_png(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

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

        self.TABLE_BUFFER_DIST = 0.05

    def get_box(self) -> RigidBody:
        return self.station.get_box()

    def get_box_planar_pose(self):
        return self.station.get_box_planar_pose(self.mbp_context)

    def set_box_planar_pose(self, box_pose: PlanarPose):
        box_height = self.station.get_box_shape().height()
        q = box_pose.to_generalized_coords(box_height + self.TABLE_BUFFER_DIST)
        self.station.mbp.SetPositions(self.mbp_context, self.station.box, q)

    def get_pusher_planar_pose(self):
        return self.station.get_pusher_planar_pose(self.mbp_context)

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
            object_height=pusher_shape.length() + self.TABLE_BUFFER_DIST
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

        box_position = self.station.mbp.GetPositions(self.mbp_context, self.station.box)
        q0 = np.concatenate([self.DEFAULT_JOINT_POSITIONS, box_position])
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)

        # Will automatically set the positions of the objects
        Solve(ik.prog())

    def _set_joint_positions(self, joint_positions: npt.NDArray[np.float64]):
        self.station.mbp.SetPositions(
            self.mbp_context, self.station.iiwa, joint_positions
        )