from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.geometry import Cylinder
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.all import InverseKinematics
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
from pydrake.solvers import Solve
from pydrake.systems.framework import (
    Context,
    Diagram,
    DiagramBuilder,
    InputPort,
    LeafSystem,
    OutputPort,
)
from pydrake.systems.primitives import ConstantValueSource, Multiplexer, ZeroOrderHold

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingDiagram,
)


def _load_robot(time_step: float = 1e-3) -> MultibodyPlant:
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


def solve_ik(
    plant: MultibodyPlant,
    pose: RigidTransform,
    default_joint_positions: npt.NDArray[np.float64],
    disregard_angle: bool = False,
) -> npt.NDArray[np.float64]:
    # Plant needs to be just the robot without other objects
    # Need to create a new context that the IK can use for solving the problem

    ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
    pusher_frame = plant.GetFrameByName("pusher_end")
    EPS = 1e-3
    # print("pose.translation()", pose.translation())
    # print("ub", pose.translation() + np.ones(3) * EPS)
    # print("lb", pose.translation() - np.ones(3) * EPS)
    ik.AddPositionConstraint(
        pusher_frame,
        np.zeros(3),
        plant.world_frame(),
        pose.translation() - np.ones(3) * EPS,
        pose.translation() + np.ones(3) * EPS,
    )

    if disregard_angle:
        z_unit_vec = np.array([0, 0, 1])
        ik.AddAngleBetweenVectorsConstraint(
            pusher_frame,
            z_unit_vec,
            plant.world_frame(),
            -z_unit_vec,  # The pusher object has z-axis pointing up
            0 - EPS,
            0 + EPS,
        )

    else:
        ik.AddOrientationConstraint(
            pusher_frame,
            RotationMatrix(),
            plant.world_frame(),
            pose.rotation(),
            EPS,
        )

    # Cost on deviation from default joint positions
    prog = ik.get_mutable_prog()
    q = ik.q()

    q0 = default_joint_positions
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)

    result = Solve(ik.prog())
    assert result.is_success()

    q_sol = result.GetSolution(q)
    return q_sol

# Deprecated
class PusherPoseInverseKinematics(LeafSystem):
    def __init__(
        self,
        default_joint_positions: npt.NDArray[np.float64],
    ):
        super().__init__()

        robot = _load_robot()  # don't need the timestep
        self.default_joint_positions = default_joint_positions

        self.pusher_pose_desired = self.DeclareAbstractInputPort(
            "pusher_pose_desired",
            AbstractValue.Make(RigidTransform()),
        )
        self.joint_positions_measured = self.DeclareVectorInputPort(
            "joint_positions_measured",
            robot.num_positions(),
        )
        self.slider_pose_measured = self.DeclareAbstractInputPort(
            "slider_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.DeclareVectorOutputPort(
            "joint_positions_cmd",
            robot.num_positions(),
            self.DoCalcOutput,
        )

    def init(self, diagram: Diagram, station: PlanarPushingDiagram) -> None:
        self.diagram = diagram
        self.station = station

    def DoCalcOutput(self, context: Context, output):
        pose_desired: RigidTransform = self.pusher_pose_desired.Eval(context)  # type: ignore
        slider_pose: RigidTransform = self.slider_pose_measured.Eval(context)  # type: ignore
        current_joint_pos: npt.NDArray[np.float64] = self.joint_positions_measured.Eval(
            context
        )  # type: ignore
        joint_positions = solve_ik(
            self.diagram,
            self.station,
            pose_desired,
            slider_pose,
            self.default_joint_positions,
            current_joint_pos,
        )
        output.set_value(joint_positions)

    @classmethod
    def AddTobuilder(
        cls,
        builder: DiagramBuilder,
        pusher_pose_desired: OutputPort,
        iiwa_joint_position_measured: OutputPort,
        slider_pose_desired: OutputPort,
        iiwa_joint_position_cmd: InputPort,
        default_joint_positions: npt.NDArray[np.float64],
        rate: float = 100,
    ) -> "PusherPoseInverseKinematics":
        ik = builder.AddNamedSystem(
            "PusherPoseInverseKinematics",
            PusherPoseInverseKinematics(default_joint_positions),
        )
        builder.Connect(
            pusher_pose_desired,
            ik.GetInputPort("pusher_pose_desired"),
        )
        builder.Connect(
            iiwa_joint_position_measured,
            ik.GetInputPort("joint_positions_measured"),
        )
        builder.Connect(
            slider_pose_desired,
            ik.slider_pose_measured,
        )

        period_sec = 1 / rate
        zero_order_hold = builder.AddNamedSystem(
            "zero_order_hold",
            ZeroOrderHold(period_sec, ik.get_output_port().size()),
        )
        builder.Connect(ik.get_output_port(), zero_order_hold.get_input_port())
        builder.Connect(zero_order_hold.get_output_port(), iiwa_joint_position_cmd)

        return ik


# Deprecated
class PusherPoseToJointPosDiffIk:
    def __init__(
        self,
        time_step: float,
        robot: MultibodyPlant,
        diff_ik: DifferentialInverseKinematicsIntegrator,
    ) -> None:
        self.time_step = time_step
        self.robot = robot
        self.diff_ik = diff_ik

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        iiwa_joint_position_input: Optional[InputPort] = None,
        iiwa_state_measured: Optional[OutputPort] = None,
        time_step: float = 1 / 200,  # 200 Hz
        use_diff_ik_feedback: bool = False,
    ) -> "PusherPoseToJointPosDiffIk":
        robot = _load_robot(time_step)

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

        if iiwa_joint_position_input is not None:
            builder.Connect(
                pusher_pose_to_joint_pos.get_output_port(),
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

        if iiwa_state_measured is not None:
            # TODO set open loop!
            builder.Connect(
                iiwa_state_measured, differential_ik.GetInputPort("robot_state")
            )

        return pusher_pose_to_joint_pos

    def get_output_port(self) -> OutputPort:
        return self.diff_ik.GetOutputPort("joint_positions")

    def get_pose_input_port(self) -> InputPort:
        return self.diff_ik.GetInputPort("X_WE_desired")

    def init_diff_ik(self, q0: npt.NDArray[np.float64], root_context: Context) -> None:
        diff_ik = self.diff_ik
        diff_ik.get_mutable_parameters().set_nominal_joint_position(q0)
        diff_ik.SetPositions(
            diff_ik.GetMyMutableContextFromRoot(root_context),
            q0,
        )
