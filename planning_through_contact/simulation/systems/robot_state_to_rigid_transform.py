from pydrake.all import (
    AbstractValue,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose



class RobotStateToRigidTransform(LeafSystem):
    def __init__(self, plant: MultibodyPlant, robot_model_name: str, z_value: float):
        super().__init__()
        
        self._plant = plant
        self._z_value = z_value
        self._robot_model_instance_index = plant.GetModelInstanceByName(
            robot_model_name
        )

        # Input ports
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "robot_state",
            self._plant.num_positions(self._robot_model_instance_index)
            + self._plant.num_velocities(self._robot_model_instance_index),
        )

        self._pose_output_ports = self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )
    
    def DoCalcOutput(self, context, output):
        robot_state = self.EvalVectorInput(context, 0).get_value()
        planar_pose = PlanarPose(robot_state[0], robot_state[1], 0.0)
        output.set_value(planar_pose.to_pose(z_value=self._z_value))