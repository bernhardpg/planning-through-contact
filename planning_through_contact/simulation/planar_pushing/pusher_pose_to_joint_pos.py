from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import Context, DiagramBuilder, InputPort, OutputPort
from pydrake.systems.primitives import ConstantValueSource, Multiplexer


class PusherPoseToJointPos:
    def __init__(
        self,
        time_step: float,
        robot: MultibodyPlant,
        diff_ik: DifferentialInverseKinematicsIntegrator,
    ) -> None:
        self.time_step = time_step
        self.robot = robot
        # TODO(bernhardpg): This is where we can replace diff IK with something else
        self.converter = diff_ik

    @staticmethod
    def _load_robot(time_step: float) -> MultibodyPlant:
        robot = MultibodyPlant(time_step)
        parser = Parser(robot)
        models_folder = Path(__file__).parents[1] / "models"
        parser.package_map().PopulateFromFolder(str(models_folder))

        # Load the controller plant, i.e. the plant without the box
        CONTROLLER_PLANT_FILE = "iiwa_controller_plant.yaml"
        directives = LoadModelDirectives(str(models_folder / CONTROLLER_PLANT_FILE))
        ProcessModelDirectives(directives, robot, parser)  # type: ignore
        robot.Finalize()
        return robot

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        pusher_pose_output_port: OutputPort,
        iiwa_joint_position_input: InputPort,
        iiwa_state_measured: OutputPort,
        time_step: float = 1 / 200,  # 200 Hz
        use_diff_ik_feedback: bool = False,
    ) -> "PusherPoseToJointPos":
        robot = cls._load_robot(time_step)

        ik_params = DifferentialInverseKinematicsParameters(
            robot.num_positions(), robot.num_velocities()
        )
        ik_params.set_time_step(time_step)

        # True velocity limits for the IIWA14
        # (in rad, rounded down to the first decimal)
        IIWA14_VELOCITY_LIMITS = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        velocity_limit_factor = 1.0
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * IIWA14_VELOCITY_LIMITS,
                velocity_limit_factor * IIWA14_VELOCITY_LIMITS,
            )
        )

        EE_FRAME = "pusher_end"
        differential_ik = builder.AddNamedSystem(
            "DiffIk",
            DifferentialInverseKinematicsIntegrator(
                robot,
                robot.GetFrameByName(EE_FRAME),
                time_step,
                ik_params,
            ),
        )
        pusher_pose_to_joint_pos = cls(time_step, robot, differential_ik)

        builder.Connect(
            pusher_pose_output_port, pusher_pose_to_joint_pos.get_pose_input_port()
        )
        builder.Connect(
            differential_ik.GetOutputPort("joint_positions"),
            iiwa_joint_position_input,
        )

        if use_diff_ik_feedback:
            const = builder.AddNamedSystem(
                "true", ConstantValueSource(AbstractValue.Make(True))
            )
        else:
            const = builder.AddNamedSystem(
                "false", ConstantValueSource(AbstractValue.Make(False))
            )

        builder.Connect(
            const.get_output_port(),
            differential_ik.GetInputPort("use_robot_state"),
        )
        builder.Connect(
            iiwa_state_measured, differential_ik.GetInputPort("robot_state")
        )

        return pusher_pose_to_joint_pos

    def get_pose_input_port(self) -> InputPort:
        assert isinstance(self.converter, DifferentialInverseKinematicsIntegrator)
        return self.converter.GetInputPort("X_WE_desired")

    def init_diff_ik(self, q0: npt.NDArray[np.float64], root_context: Context) -> None:
        assert isinstance(self.converter, DifferentialInverseKinematicsIntegrator)
        diff_ik = self.converter
        diff_ik.get_mutable_parameters().set_nominal_joint_position(q0)
        diff_ik.SetPositions(
            diff_ik.GetMyMutableContextFromRoot(root_context),
            q0,
        )
