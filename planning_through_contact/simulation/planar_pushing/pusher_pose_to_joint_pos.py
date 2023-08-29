from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
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
from pydrake.systems.framework import Context, DiagramBuilder, InputPort


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
        iiwa_joint_position_input: InputPort,
        time_step: float = 1e-3,
        converter: Literal["diff_ik"] = "diff_ik",
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

        if converter == "diff_ik":
            EE_FRAME = "iiwa_link_7"
            differential_ik = builder.AddNamedSystem(
                "DiffIk",
                DifferentialInverseKinematicsIntegrator(
                    robot,
                    robot.GetFrameByName(EE_FRAME),
                    time_step,
                    ik_params,
                ),
            )
            builder.Connect(
                differential_ik.GetOutputPort("joint_positions"),
                iiwa_joint_position_input,
            )
        else:
            raise NotImplementedError("")

        pusher_pose_to_joint_pos = cls(time_step, robot, differential_ik)
        return pusher_pose_to_joint_pos

    def get_input_port(self) -> InputPort:
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
