import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem, OutputPort

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import (
    PlanarPose,
    PlanarVelocity,
)
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

        self.pusher_planar_pose_desired = self.DeclareAbstractInputPort(
            "pusher_planar_pose_desired",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.slider_planar_pose_desired = self.DeclareAbstractInputPort(
            "slider_planar_pose_desired",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.slider_theta_dot_desired = self.DeclareAbstractInputPort(
            "slider_theta_dot_desired",
            AbstractValue.Make(PlanarVelocity(v_x=0, v_y=0, omega=0)),
        )
        self.contact_mode_desired = self.DeclareAbstractInputPort(
            "contact_mode_desired",
            AbstractValue.Make(PlanarPushingContactMode(0)),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose",
            AbstractValue.Make(RigidTransform()),
        )
        self.slider_spatial_velocity = self.DeclareAbstractInputPort(
            "slider_spatial_velocity",
            AbstractValue.Make(SpatialVelocity()),
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
        pusher_planar_pose_desired: OutputPort,
        slider_planar_pose_desired: OutputPort,
        slider_theta_dot_desired: OutputPort,
        slider_pose_measured: OutputPort,
        slider_spatial_velocity_measured: OutputPort,
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
            pusher_planar_pose_desired,
            pusher_pose_controller.GetInputPort("pusher_planar_pose_desired"),
        )
        builder.Connect(
            slider_planar_pose_desired,
            pusher_pose_controller.GetInputPort("slider_planar_pose_desired"),
        )
        builder.Connect(
            slider_theta_dot_desired,
            pusher_pose_controller.GetInputPort("slider_theta_dot_desired"),
        )
        builder.Connect(
            slider_pose_measured,
            pusher_pose_controller.GetInputPort("slider_pose"),
        )
        builder.Connect(
            slider_spatial_velocity_measured,
            pusher_pose_controller.GetInputPort("slider_spatial_velocity"),
        )
        return pusher_pose_controller

    def _compute_control(
        self,
        theta: float,
        theta_desired: float,
        theta_dot: float,
        theta_dot_desired: float,
        loc: PolytopeContactLocation,
    ) -> npt.NDArray[np.float64]:
        theta_error = theta_desired - theta
        theta_dot_error = theta_dot_desired - theta_dot

        # K_P = 0.3
        K_P = 0.6
        # K_D = 2 * np.sqrt(K_P)
        K_D = 0.01
        # K_D = 0

        print(f"theta: {theta_error}, theta_dot: {theta_dot_error}")

        # Commanded difference in position along contact face
        delta_lam = -(K_P * theta_error + K_D * theta_dot_error)

        pv1, pv2 = self.object_geometry.get_proximate_vertices_from_location(loc)

        # TODO(bernhardpg): consider not normalizing this
        dir_vector = (pv2 - pv1) / np.linalg.norm(pv2 - pv1)
        delta_p_B_c = delta_lam * dir_vector

        R_WB = two_d_rotation_matrix_from_angle(theta_desired)
        # We only care about the diffs here, i.e. no constant translation
        delta_p_W_c = R_WB.dot(delta_p_B_c)
        return delta_p_W_c

    def DoCalcOutput(self, context: Context, output):
        mode_desired: PlanarPushingContactMode = self.contact_mode_desired.Eval(context)  # type: ignore
        # print(mode_desired)
        pusher_planar_pose_desired: PlanarPose = self.pusher_planar_pose_desired.Eval(context)  # type: ignore

        if mode_desired == PlanarPushingContactMode.NO_CONTACT:
            pusher_pose_desired = pusher_planar_pose_desired.to_pose(
                z_value=self.z_dist
            )
            output.set_value(pusher_pose_desired)
        else:  # do control of angle
            slider_pose: RigidTransform = self.slider_pose.Eval(context)  # type: ignore

            slider_planar_pose = PlanarPose.from_pose(slider_pose)
            slider_planar_pose_desired: PlanarPose = self.slider_planar_pose_desired.Eval(context)  # type: ignore

            slider_spatial_velocity: SpatialVelocity = self.slider_spatial_velocity.Eval(context)  # type: ignore
            Z_AXIS = 2
            slider_theta_dot = slider_spatial_velocity.rotational()[Z_AXIS]
            slider_theta_dot_desired: float = self.slider_theta_dot_desired.Eval(context)  # type: ignore

            delta_p_W_c = self._compute_control(
                slider_planar_pose.theta,
                slider_planar_pose_desired.theta,
                slider_theta_dot,
                slider_theta_dot_desired,
                mode_desired.to_contact_location(),
            )
            pusher_planar_pose_adjustment = PlanarPose(
                delta_p_W_c[0, 0], delta_p_W_c[1, 0], theta=0
            )

            pusher_pose_planar_command = (
                pusher_planar_pose_desired + pusher_planar_pose_adjustment
            )
            pusher_pose_command = pusher_pose_planar_command.to_pose(
                z_value=self.z_dist
            )
            output.set_value(pusher_pose_command)
