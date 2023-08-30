import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class PusherPoseController(LeafSystem):
    def __init__(
        self,
        z_dist_to_table: float = 0.5,
    ):
        super().__init__()
        self.z_dist = z_dist_to_table

        self.input_port = self.DeclareAbstractInputPort(
            "planar_pose",
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0)),
        )
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    def DoCalcOutput(self, context: Context, output):
        planar_pose: PlanarPose = self.input_port.Eval(context)  # type: ignore
        pose = planar_pose.to_pose(z_value=self.z_dist)
        output.set_value(pose)
