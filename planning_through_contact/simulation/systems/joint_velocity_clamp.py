import logging

from pydrake.all import LeafSystem

logger = logging.getLogger(__name__)


class JointVelocityClamp(LeafSystem):
    def __init__(self, num_positions, joint_velocity_limits):
        LeafSystem.__init__(self)
        self._num_positions = num_positions
        self._joint_velocity_limits = joint_velocity_limits
        self._joint_acceleration_limits = joint_velocity_limits * 0.1
        self._joint_positions_commanded = self.DeclareVectorInputPort(
            "joint_positions_commanded", num_positions
        )
        self.DeclareVectorOutputPort(
            "joint_positions_clamped", num_positions, self.DoCalcOutput
        )

        self._last_velocity = self.DeclareDiscreteState([0.0] * num_positions)
        self._last_command = self.DeclareDiscreteState([-999] * num_positions)
        self._last_time = self.DeclareDiscreteState([0.0])

    def DoCalcOutput(self, context, output):
        joint_positions_commanded = self._joint_positions_commanded.Eval(context)
        last_velocity_state = context.get_discrete_state(self._last_velocity)
        last_command_state = context.get_discrete_state(self._last_command)
        last_time_state = context.get_discrete_state(self._last_time)
        time_step = context.get_time() - last_time_state.get_value()[0]

        last_command = last_command_state.get_value()
        last_velocity_state.get_value()
        if time_step == 0 or (last_command == [-999] * self._num_positions).all():
            joint_positions_clamped = joint_positions_commanded
            velocity_clamped = [0.0] * self._num_positions
        else:
            joint_positions_clamped = [0.0] * self._num_positions
            velocity_clamped = [0.0] * self._num_positions
            for i in range(self._num_positions):
                vel_sign = (
                    1.0 if joint_positions_commanded[i] > last_command[i] else -1.0
                )
                velocity = (joint_positions_commanded[i] - last_command[i]) / time_step
                speed = abs(velocity)
                # accel_sign = 1.0 if velocity > last_velocity[i] else -1.0
                # acceleration_mag = abs(velocity - last_velocity[i]) / time_step
                # if acceleration_mag > self._joint_acceleration_limits[i]:
                #     speed = (
                #         last_velocity[i]
                #         + self._joint_acceleration_limits[i] * time_step * accel_sign
                #     )
                #     # logger.warn(
                #     #     f"({context.get_time()}) clamped acceleration {acceleration_mag} to {self._joint_acceleration_limits[i]}"
                #     # )
                if speed > self._joint_velocity_limits[i]:
                    # Above speed limit
                    joint_positions_clamped[i] = (
                        last_command[i]
                        + self._joint_velocity_limits[i] * time_step * vel_sign
                    )
                    velocity_clamped[i] = self._joint_velocity_limits[i] * vel_sign
                else:
                    joint_positions_clamped[i] = joint_positions_commanded[i]
                    velocity_clamped[i] = speed * vel_sign
            if (joint_positions_clamped != joint_positions_commanded).any():
                logger.warn(
                    f"({context.get_time()}) clamped joint_positions_commanded deltas {abs((joint_positions_commanded - last_command)/time_step)}"
                )

        output.set_value(joint_positions_clamped)
        last_command_state.set_value(joint_positions_clamped)
        last_time_state.set_value([context.get_time()])
        last_velocity_state.set_value(velocity_clamped)
