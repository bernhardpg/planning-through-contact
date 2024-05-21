from pydrake.systems.framework import LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class PlanarPoseToGeneralizedCoords(LeafSystem):
    """
    Converts Planar Pose ([x, y, theta]) to generalized coords
    """

    def __init__(self, z_value: float, z_axis_is_positive: bool):
        super().__init__()
        self._z_value = z_value
        self._z_axis_is_positive = z_axis_is_positive
        self.DeclareVectorInputPort("planar_pose_input", 3)
        self.DeclareVectorOutputPort("generalized_coords_output", 7, self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        planar_pose_input = self.EvalVectorInput(context, 0).get_value()
        planar_pose = PlanarPose(
            planar_pose_input[0], planar_pose_input[1], planar_pose_input[2]
        )
        output.set_value(
            planar_pose.to_generalized_coords(self._z_value, self._z_axis_is_positive)
        )
