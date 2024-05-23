from pydrake.all import AbstractValue, RigidTransform
from pydrake.systems.framework import LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class PlanarTranslationToRigidTransformSystem(LeafSystem):
    """Converts [x,y] vector into a rigid transform"""

    def __init__(self, z_dist: float = 0.02):
        super().__init__()
        self._z_dist = z_dist
        self.DeclareVectorInputPort("vector_input", 2)
        self.DeclareAbstractOutputPort(
            "rigid_transform_output",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output):
        planar_translation = self.EvalVectorInput(context, 0).get_value()
        planar_pose = PlanarPose(planar_translation[0], planar_translation[1], 0.0)
        output.set_value(planar_pose.to_pose(z_value=self._z_dist))
