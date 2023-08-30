import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem, OutputPort

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
)
from planning_through_contact.geometry.rigid_body import RigidBody


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
            slider_pose_measured,
            pusher_pose_controller.GetInputPort("slider_pose"),
        )
        builder.Connect(
            slider_spatial_velocity_measured,
            pusher_pose_controller.GetInputPort("slider_spatial_velocity"),
        )
        return pusher_pose_controller

    def DoCalcOutput(self, context: Context, output):
        pusher_planar_pose_desired: PlanarPose = self.pusher_planar_pose_desired.Eval(context)  # type: ignore
        slider_planar_pose_desired: PlanarPose = self.slider_planar_pose_desired.Eval(context)  # type: ignore

        mode_desired: PlanarPushingContactMode = self.contact_mode_desired.Eval(context)  # type: ignore
        slider_pose: RigidTransform = self.slider_pose.Eval(context)  # type: ignore
        slider_spatial_velocity: SpatialVelocity = self.slider_spatial_velocity.Eval(context)  # type: ignore

        slider_planar_pose = PlanarPose.from_pose(slider_pose)
        breakpoint()

        pose_desired = pusher_planar_pose_desired.to_pose(z_value=self.z_dist)
        output.set_value(pose_desired)
