import logging
from copy import copy
from enum import Enum

import numpy as np
from pydrake.all import (
    AbstractValue,
    GcsTrajectoryOptimization,
    HPolyhedron,
    InputPortIndex,
    LeafSystem,
    MultibodyPlant,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    Point,
    RigidTransform,
    Toppra,
)

from planning_through_contact.simulation.planar_pushing.inverse_kinematics import (
    solve_ik,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

logger = logging.getLogger(__name__)


class IiwaPlannerMode(Enum):
    PLAN_GO_PUSH_START = 0
    GO_PUSH_START = 1
    WAIT_PUSH = 2
    PUSHING = 3


class IiwaPlanner(LeafSystem):
    """Planner that manages the iiwa going to the start position, waiting and then pushing according to the desired planar position source."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        robot_plant: MultibodyPlant,
        initial_delay=2,
        wait_push_delay=2,
    ):
        LeafSystem.__init__(self)
        self._wait_push_delay = wait_push_delay
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(IiwaPlannerMode.PLAN_GO_PUSH_START)
        )

        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": initial_delay})
        )

        # For GoPushStart mode:
        num_positions = robot_plant.num_positions()
        self._iiwa_position_measured_index = self.DeclareVectorInputPort(
            "iiwa_position_measured", robot_plant.num_positions()
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )

        # This output port is not currently being used
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", num_positions, self.CalcIiwaPosition
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self._internal_model = robot_plant

        self._sim_config = sim_config

    def Update(self, context, state):
        # FSM Logic for planner
        mode = context.get_abstract_state(self._mode_index).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(self._times_index).get_value()

        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            if context.get_time() > times["initial"]:
                self.PlanGoPushStart(context, state)
            return
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()

            if current_time > times["go_push_start_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IiwaPlannerMode.WAIT_PUSH
                )
                logger.debug(f"Switching to WAIT_PUSH mode at time {current_time}.")
                current_pos = self.get_input_port(
                    self._iiwa_position_measured_index
                ).Eval(context)
                logger.debug(f"Current position: {current_pos}")
            return
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            if current_time > times["wait_push_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IiwaPlannerMode.PUSHING
                )
                logger.debug(f"Switching to PUSHING mode at time {current_time}.")
                current_pos = self.get_input_port(
                    self._iiwa_position_measured_index
                ).Eval(context)
                logger.debug(f"Current position: {current_pos}")
                global time_pushing_transition
                time_pushing_transition = current_time
            return
        # elif mode == IiwaPlannerMode.PUSHING:
        #     current_pos = self.get_input_port(
        #         self._iiwa_position_measured_index
        #     ).Eval(context)
        #     logger.debug(f"PUSHING: time {context.get_time()} Current position: {current_pos}")

    def PlanGoPushStart(self, context, state):
        logger.debug(f"PlanGoPushStart at time {context.get_time()}.")
        q_start = copy(context.get_discrete_state(self._q0_index).get_value())
        q_goal = self.get_desired_start_pos()

        q_traj = self.create_go_push_start_traj(q_goal, q_start)
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)
        times = state.get_mutable_abstract_state(int(self._times_index)).get_value()

        total_delay = q_traj.end_time() + context.get_time() + self._wait_push_delay
        assert (
            self._sim_config.delay_before_execution >= total_delay
        ), f"Not enough time to execute plan. Required time: {total_delay}s."
        times["go_push_start_initial"] = context.get_time()
        times["go_push_start_final"] = q_traj.end_time() + context.get_time()
        times["wait_push_final"] = times["go_push_start_final"] + self._wait_push_delay
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            IiwaPlannerMode.GO_PUSH_START
        )
        self.push_start_pos = q_goal

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PUSHING:
            output.set_value(InputPortIndex(2))  # Pushing (DiffIK)
        else:
            output.set_value(InputPortIndex(1))  # Wait/GoPushStart

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PUSHING:
            output.set_value(False)  # Pushing (DiffIK)
        else:
            output.set_value(True)  # Wait/GoPushStart

    def CalcIiwaPosition(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            q_start = copy(context.get_discrete_state(self._q0_index).get_value())
            output.SetFromVector(q_start)
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()

            times = context.get_mutable_abstract_state(
                int(self._times_index)
            ).get_value()

            traj_curr_time = context.get_time() - times["go_push_start_initial"]

            output.SetFromVector(traj_q.value(traj_curr_time))
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            output.SetFromVector(self.push_start_pos)
        elif mode == IiwaPlannerMode.PUSHING:
            assert (
                False
            ), "Planner CalcIiwaPosition should not be called in PUSHING mode."
        else:
            assert False, "Invalid mode."

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_measured_index)).Eval(context),
        )

    def get_desired_start_pos(self):
        # Set iiwa starting position
        global desired_pose
        desired_pose = self._sim_config.pusher_start_pose.to_pose(
            self._sim_config.pusher_z_offset
        )
        start_joint_positions = solve_ik(
            self._internal_model,
            pose=desired_pose,
            default_joint_positions=self._sim_config.default_joint_positions,
        )

        return start_joint_positions

    @staticmethod
    def make_traj_toppra(traj, plant, vel_limits, accel_limits, num_grid_points=1000):
        toppra = Toppra(
            traj,
            plant,
            np.linspace(traj.start_time(), traj.end_time(), num_grid_points),
        )
        toppra.AddJointVelocityLimit(-vel_limits, vel_limits)
        toppra.AddJointAccelerationLimit(-accel_limits, accel_limits)
        time_traj = toppra.SolvePathParameterization()
        return PathParameterizedTrajectory(traj, time_traj)

    def create_go_push_start_traj(self, q_goal, q_start):
        plant = self._internal_model
        num_positions = plant.num_positions()

        gcs = GcsTrajectoryOptimization(plant.num_positions())

        workspace = gcs.AddRegions(
            [
                HPolyhedron.MakeBox(
                    plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
                )
            ],
            5,
            1,
            60,
        )

        logger.debug(f"q_start = {q_start}")
        logger.debug(f"q_goal = {q_goal}")

        vel_limits = 1 * np.ones(7)  # 0.15
        accel_limits = 1 * np.ones(7)
        # Set non-zero h_min for start and goal to enforce zero velocity.
        start = gcs.AddRegions([Point(q_start)], order=1, h_min=0.1)
        goal = gcs.AddRegions([Point(q_goal)], order=1, h_min=0.1)
        goal.AddVelocityBounds([0] * num_positions, [0] * num_positions)
        gcs.AddEdges(start, workspace)
        gcs.AddEdges(workspace, goal)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        gcs.AddVelocityBounds(-vel_limits, vel_limits)

        traj, result = gcs.SolvePath(start, goal)

        traj_toppra = IiwaPlanner.make_traj_toppra(
            traj, plant, vel_limits=vel_limits, accel_limits=accel_limits
        )

        return traj_toppra
