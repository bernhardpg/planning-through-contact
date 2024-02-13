from pydrake.all import (
    AbstractValue,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
import time


class StateToRigidTransform(LeafSystem):
    def __init__(self, plant: MultibodyPlant, robot_model_name: str, z_value: float):
        super().__init__()
        
        self._plant = plant
        self._z_value = z_value
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
        robot_position = robot_state[:self._num_positions]
        if self._num_positions == 7:
            planar_pose = PlanarPose.from_generalized_coords(robot_position)
        else:
            planar_pose = PlanarPose(robot_position[0], robot_position[1], 0.0)
        output.set_value(planar_pose.to_pose(z_value=self._z_value))