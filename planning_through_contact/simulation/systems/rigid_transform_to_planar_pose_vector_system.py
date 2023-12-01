from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.all import (RigidTransform, AbstractValue)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose

class RigidTransformToPlanrPoseVectorSystem(LeafSystem):
    """Converts a RigidTransform to a [x,y,theta] vector"""
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("rigid_transform_input", AbstractValue.Make(RigidTransform()))
        self.DeclareVectorOutputPort("vector_output", 3, self.DoCalcVectorOutput)

    def DoCalcVectorOutput(self, context, output):
        rigid_transform = self.EvalAbstractInput(context, 0).get_value()
        planar_pose = PlanarPose.from_pose(rigid_transform)
        output.SetFromVector(planar_pose.vector())
