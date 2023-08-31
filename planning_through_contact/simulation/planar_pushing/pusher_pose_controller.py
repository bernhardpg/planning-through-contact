import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem, OutputPort

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle


class PusherPoseController(LeafSystem):
    def __init__(
        self,
        object_geometry: CollisionGeometry,
        z_dist_to_table: float = 0.5,
    ):
        super().__init__()
        self.z_dist = z_dist_to_table
        self.object_geometry = object_geometry

        self.pusher_planar_pose_traj = self.DeclareAbstractInputPort(
            "pusher_planar_pose_traj",
            AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
        )
        self.slider_planar_pose_traj = self.DeclareAbstractInputPort(
            "slider_planar_pose_traj",
            AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
        )
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.contact_mode_desired = self.DeclareAbstractInputPort(
            "contact_mode_desired",
            AbstractValue.Make(PlanarPushingContactMode(0)),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose",
            AbstractValue.Make(RigidTransform()),
        )
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    @classmethod
    def AddToBuilder(
        cls,
        builder: DiagramBuilder,
        slider: RigidBody,
        contact_mode_desired: OutputPort,
        slider_planar_pose_traj: OutputPort,
        pusher_planar_pose_traj: OutputPort,
        pusher_planar_pose_measured: OutputPort,
        slider_pose_measured: OutputPort,
    ) -> "PusherPoseController":
        pusher_pose_controller = builder.AddNamedSystem(
            "PusherPoseController",
            PusherPoseController(slider.geometry, z_dist_to_table=0.02),
        )
        builder.Connect(
            contact_mode_desired,
            pusher_pose_controller.GetInputPort("contact_mode_desired"),
        )
        builder.Connect(
            pusher_planar_pose_measured,
            pusher_pose_controller.GetInputPort("pusher_pose_measured"),
        )
        builder.Connect(
            pusher_planar_pose_traj,
            pusher_pose_controller.GetInputPort("pusher_planar_pose_traj"),
        )
        builder.Connect(
            slider_planar_pose_traj,
            pusher_pose_controller.GetInputPort("slider_planar_pose_traj"),
        )
        builder.Connect(
            slider_pose_measured,
            pusher_pose_controller.GetInputPort("slider_pose"),
        )
        return pusher_pose_controller

    def _compute_control(
        self,
        theta: float,
        theta_desired: float,
        p_W_c: npt.NDArray[np.float64],
        p_WB: npt.NDArray[np.float64],
        loc: PolytopeContactLocation,
    ) -> npt.NDArray[np.float64]:
        theta_error = theta_desired - theta

        K_P = 0.5

        # Commanded difference in position along contact face
        delta_lam = -K_P * theta_error

        pv1, pv2 = self.object_geometry.get_proximate_vertices_from_location(loc)

        # TODO(bernhardpg): consider not normalizing this
        dir_vector = (pv2 - pv1) / np.linalg.norm(pv2 - pv1)
        delta_p_B_c = delta_lam * dir_vector

        R_WB = two_d_rotation_matrix_from_angle(theta)
        p_B_c = R_WB.T.dot(p_W_c - p_WB)

        p_B_c_cmd = p_B_c + delta_p_B_c
        cmd_between_pv1_and_pv2 = (
            np.cross(pv2.flatten(), p_B_c_cmd.flatten()) >= 0
            and np.cross(pv1.flatten(), p_B_c_cmd.flatten()) <= 0
        )
        if not cmd_between_pv1_and_pv2:  # saturate control
            print("saturate")
            delta_p_B_c = np.zeros((2, 1))

        # We only care about the diffs here, i.e. no constant translation
        R_WB_desired = two_d_rotation_matrix_from_angle(theta_desired)
        delta_p_W_c = R_WB_desired.dot(delta_p_B_c)

        return delta_p_W_c

    def DoCalcOutput(self, context: Context, output):
        mode_desired: PlanarPushingContactMode = self.contact_mode_desired.Eval(context)  # type: ignore
        pusher_planar_pose_traj: List[PlanarPose] = self.pusher_planar_pose_traj.Eval(context)  # type: ignore

        if mode_desired == PlanarPushingContactMode.NO_CONTACT:
            curr_planar_pose = pusher_planar_pose_traj[0]
            pusher_pose_desired = curr_planar_pose.to_pose(z_value=self.z_dist)
            output.set_value(pusher_pose_desired)
        else:  # do control of angle
            slider_pose: RigidTransform = self.slider_pose.Eval(context)  # type: ignore

            slider_planar_pose = PlanarPose.from_pose(slider_pose)
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore

            pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
            pusher_planar_pose = PlanarPose.from_pose(pusher_pose)

            delta_p_W_c = self._compute_control(
                slider_planar_pose.theta,
                slider_planar_pose_traj[0].theta,
                pusher_planar_pose.pos(),
                slider_planar_pose.pos(),
                mode_desired.to_contact_location(),
            )
            pusher_planar_pose_adjustment = PlanarPose(
                delta_p_W_c[0, 0], delta_p_W_c[1, 0], theta=0
            )

            pusher_pose_planar_command = (
                pusher_planar_pose_traj[0] + pusher_planar_pose_adjustment
            )
            pusher_pose_command = pusher_pose_planar_command.to_pose(
                z_value=self.z_dist
            )
            output.set_value(pusher_pose_command)
