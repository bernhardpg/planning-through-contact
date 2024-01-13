from typing import List

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import ModelInstanceIndex, SpatialVelocity
from pydrake.systems.framework import Context, LeafSystem


class PusherSliderPoseSelector(LeafSystem):
    """
    Select the slider pose and spatial velocity and output these.
    """

    def __init__(
        self, slider_idx: ModelInstanceIndex, pusher_idx: ModelInstanceIndex
    ) -> None:
        super().__init__()

        self.slider_idx = slider_idx
        self.pusher_idx = pusher_idx

        self.body_poses = self.DeclareAbstractInputPort(
            "body_poses",
            AbstractValue.Make([RigidTransform()]),
        )
        self.body_spatial_velocities = self.DeclareAbstractInputPort(
            "body_spatial_velocities",
            AbstractValue.Make([SpatialVelocity()]),
        )
        self.DeclareAbstractOutputPort(
            "slider_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcSliderPose,
        )
        self.DeclareAbstractOutputPort(
            "slider_spatial_velocity",
            lambda: AbstractValue.Make(SpatialVelocity()),
            self.DoCalcSliderSpatialVelocity,
        )
        self.DeclareAbstractOutputPort(
            "pusher_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcPusherPose,
        )

    def DoCalcSliderPose(self, context: Context, output):
        body_poses: List[RigidTransform] = self.body_poses.Eval(context)  # type: ignore
        slider_pose = body_poses[self.slider_idx]
        output.set_value(slider_pose)

    def DoCalcSliderSpatialVelocity(self, context: Context, output):
        spatial_velocities: List[SpatialVelocity] = self.body_spatial_velocities.Eval(context)  # type: ignore
        slider_spatial_vel = spatial_velocities[self.slider_idx]
        output.set_value(slider_spatial_vel)

    def DoCalcPusherPose(self, context: Context, output):
        body_poses: List[RigidTransform] = self.body_poses.Eval(context)  # type: ignore
        pusher_pose = body_poses[self.pusher_idx]
        output.set_value(pusher_pose)
