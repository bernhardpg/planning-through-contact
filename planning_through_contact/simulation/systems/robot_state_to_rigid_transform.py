from pydrake.all import (
    AbstractValue,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
import numpy as np


class RobotStateToRigidTransform(LeafSystem):
    def __init__(self, plant: MultibodyPlant, robot_model_name: str, offset=None):
        super().__init__()
        
        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._offset = offset
        self._robot_model_name = robot_model_name
        self._robot_model_instance_index = plant.GetModelInstanceByName(
            robot_model_name
        )
        self._num_positions = self._plant.num_positions(self._robot_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._robot_model_instance_index)

        # Input ports
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "state",
            self._num_positions + self._num_velocities,
        )

        self._pose_output_ports = self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )
    
    def DoCalcOutput(self, context, output):
        robot_state = self.EvalVectorInput(context, 0).get_value()
        q = robot_state[:self._num_positions]
        self._plant.SetPositions(self._plant_context, self._robot_model_instance_index, q)
        pose = self._plant.EvalBodyPoseInWorld(
            self._plant_context,
            self._plant.GetBodyByName("pusher")
        )
        if self._offset:
            pose.set_translation(pose.translation() + pose.rotation()*self._offset)
        output.set_value(pose)