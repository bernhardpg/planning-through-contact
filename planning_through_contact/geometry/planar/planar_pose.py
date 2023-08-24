from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix


@dataclass
class PlanarPose:
    x: float
    y: float
    theta: float

    @classmethod
    def from_pose(cls, pose: RigidTransform) -> "PlanarPose":
        """
        Creates a planar pose from a RigidTransform.

        The z-position value is disregarded, and theta is set to the rotation about the z-axis.
        """
        x = pose.translation()[0]
        y = pose.translation()[1]

        Z_AXIS = 2
        theta = RollPitchYaw(pose.rotation()).vector()[Z_AXIS]
        return cls(x, y, theta)

    def to_pose(
        self, object_height: float, pos_along_z_axis: bool = False
    ) -> RigidTransform:
        """
        Creates a RigidTransform from a planar pose, with the z-axis pointing downwards.

        @param object_height: Height of the object. This is required to set the z-value of the pose correctly.
        @param pos_along_z_axis: Set to true to point z-axis upwards

        """
        roll = 0 if pos_along_z_axis else np.pi

        pose = RigidTransform(
            RollPitchYaw(np.array([roll, 0.0, self.theta])),  # type: ignore
            np.array([self.x, self.y, object_height]),
        )
        return pose

    @classmethod
    def from_generalized_coords(cls, q: npt.NDArray[np.float64]) -> "PlanarPose":
        """
        Creates a planar pose from a vector q of generalized coordinates: [quaternion, translation]'

        The z-position value is disregarded, and theta is set to the rotation about the z-axis.
        """
        q_wxyz = q[0:4] / np.linalg.norm(q[0:4])
        x = q[4]
        y = q[5]

        Z_AXIS = 2
        theta = RollPitchYaw(Quaternion(q_wxyz)).vector()[Z_AXIS]
        return cls(x, y, theta)

    def to_generalized_coords(self, object_height: float) -> npt.NDArray[np.float64]:
        """
        Returns the full RigidBody pose as generalized coordinates: [quaternion, translation]'

        """
        pose = self.to_pose(object_height)
        quat = pose.rotation().ToQuaternion().wxyz()
        trans = pose.translation()
        gen_coords = np.concatenate((quat, trans))
        return gen_coords

    def vector(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y, self.theta])

    def pos(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y]).reshape((2, 1))

    def full_vector(self) -> npt.NDArray[np.float64]:
        """
        Returns a vector where theta is represented by two variables, cos(theta) and sin(theta).
        """
        return np.array([self.x, self.y, np.cos(self.theta), np.sin(self.theta)])

    def two_d_rot_matrix(self) -> npt.NDArray[np.float64]:
        R = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )
        return R

    def cos(self) -> float:
        return np.cos(self.theta)

    def sin(self) -> float:
        return np.sin(self.theta)
