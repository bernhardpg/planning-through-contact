import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.all import (
    RigidTransform,
    AbstractValue,
    MultibodyPlant,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing.inverse_kinematics import solve_ik

class DiffIKSystem(LeafSystem):
    """Solves inverse kinematics"""

    def __init__(
            self, plant: MultibodyPlant,
            time_step: float,
            default_joint_positions: np.ndarray = None,
            disregard_angle: bool = False # TODO: implement this
        ):
        super().__init__()
        if default_joint_positions is not None:
            assert len(default_joint_positions) == plant.num_positions()
        
        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._time_step = time_step
        self._default_joint_positions = default_joint_positions
        self._disregard_angle = disregard_angle
        self._paramters = self._get_diff_ik_params()
        self._pusher_frame = self._plant.GetFrameByName("pusher_end")
        self._prev_v = np.zeros(plant.num_velocities())

        # Declare I/O ports
        self.DeclareAbstractInputPort(
            "rigid_transform_input", AbstractValue.Make(RigidTransform())
        )
        self.DeclareVectorInputPort(
            "state", plant.num_positions() + plant.num_velocities()
        )

        self.DeclareVectorOutputPort("q", plant.num_positions(), self.DoCalcVectorOutput)

    def _get_diff_ik_params(self):
        # Initialize parameters
        param = DifferentialInverseKinematicsParameters(
            num_positions=self._plant.num_positions(),
            num_velocities=self._plant.num_velocities()
        )

        # Set parameters
        param.set_time_step(self._time_step)
        if self._default_joint_positions is not None:
            param.set_nominal_joint_position(self._default_joint_positions)
        param.set_joint_position_limits(
            (self._plant.GetPositionLowerLimits(), self._plant.GetPositionUpperLimits())
        )
        param.set_joint_velocity_limits(
            (self._plant.GetVelocityLowerLimits(), self._plant.GetVelocityUpperLimits())
        )
        param.set_joint_acceleration_limits(
            (self._plant.GetAccelerationLowerLimits(), self._plant.GetAccelerationUpperLimits())
        )
                
        return param

    def DoCalcVectorOutput(self, context, output):
        # Read input ports
        rigid_transform = self.EvalAbstractInput(context, 0).get_value()
        state = self.EvalVectorInput(context, 1).get_mutable_value()
        if np.allclose(state, np.zeros_like(state)):
            state[:self._plant.num_positions()] = self._default_joint_positions

        # Update plant context
        self._plant.SetPositionsAndVelocities(self._plant_context, state)

        diff_ik_result = DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            rigid_transform,
            self._pusher_frame,
            self._paramters,
        )

        if diff_ik_result.status == DifferentialInverseKinematicsStatus.kSolutionFound:
            v = diff_ik_result.joint_velocities
            q = state[:self._plant.num_positions()] + v * self._time_step
            # print("Solution Found")
        elif diff_ik_result.status == DifferentialInverseKinematicsStatus.kStuck:
            v = diff_ik_result.joint_velocities
            q = state[:self._plant.num_positions()] + v * self._time_step
            # print("Stuck")
        else:
            # TODO: implement in house ik that returns the result with solution flags
            q = solve_ik(self._plant,
                     rigid_transform,
                     self._default_joint_positions,
                     self._disregard_angle
            )
            # print("No solution found")
        
        output.SetFromVector(q)
