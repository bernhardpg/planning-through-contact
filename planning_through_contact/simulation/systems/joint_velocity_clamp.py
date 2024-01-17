from pydrake.all import LeafSystem, QueryObject, AbstractValue
import logging

logger = logging.getLogger(__name__)

class JointVelocityClamp(LeafSystem):
    def __init__(self, num_positions, joint_velocity_limits, time_step):
        LeafSystem.__init__(self)
        self._num_positions = num_positions
        self._joint_velocity_limits = joint_velocity_limits
        self._time_step = time_step
        # self._joint_positions_measured = self.DeclareVectorInputPort("joint_positions_measured", num_positions)
        self._joint_positions_commanded = self.DeclareVectorInputPort("joint_positions_commanded", num_positions)
        self.DeclareVectorOutputPort("joint_positions_clamped", num_positions, self.DoCalcOutput)

        self._last_command = self.DeclareAbstractState(AbstractValue.Make([-999]*num_positions))
    
    def DoCalcOutput(self, context, output):
        joint_positions_commanded = self._joint_positions_commanded.Eval(context)
        last_command_state = context.get_abstract_state(self._last_command)
        last_command = last_command_state.get_value()
        if last_command == [-999]*self._num_positions:
            logger.debug(f"Initializing JointVelocityClamp with joint_positions_commanded {joint_positions_commanded}")
            last_command = joint_positions_commanded

        joint_positions_clamped = [0.0]*self._num_positions
        for i in range(self._num_positions):
            sign = 1.0 if joint_positions_commanded[i] > last_command[i] else -1.0
            delta = abs((joint_positions_commanded[i] - last_command[i])/self._time_step)
            if delta > self._joint_velocity_limits[i]:
                joint_positions_clamped[i] = last_command[i] + self._joint_velocity_limits[i] * self._time_step * sign
            else:
                joint_positions_clamped[i] = joint_positions_commanded[i]
        if (joint_positions_clamped != joint_positions_commanded).any():
            logger.warn(f"JointVelocityClamp clamped joint_positions_commanded deltas {abs((joint_positions_commanded - last_command)/self._time_step)}")

        output.set_value(joint_positions_clamped)
        last_command_state.set_value(joint_positions_clamped)