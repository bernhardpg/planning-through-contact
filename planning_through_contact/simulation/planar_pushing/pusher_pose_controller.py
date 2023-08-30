import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class PusherPoseController(LeafSystem):
    def __init__(
        self,
        object_geometry: CollisionGeometry,
        z_dist_to_table: float = 0.5,
    ):
        super().__init__()
        self.z_dist = z_dist_to_table
        self.object_geometry = object_geometry

        self.planar_pose_desired = self.DeclareAbstractInputPort(
            "planar_pose_desired",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    def DoCalcOutput(self, context: Context, output):
        planar_pose: PlanarPose = self.planar_pose_desired.Eval(context)  # type: ignore
        pose = planar_pose.to_pose(z_value=self.z_dist)
        output.set_value(pose)
