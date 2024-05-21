import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import AbstractStateIndex, Context, LeafSystem, State

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import (
    HybridMpc,
    HybridMpcConfig,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)

logger = logging.getLogger(__name__)

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


class PusherPoseControllerState(Enum):
    """FSM states for the PusherPoseController"""

    OPEN_LOOP = 0
    CFREE_MPC = 1
    HYBRID_MPC = 2
    RETURN_TO_CONTACT = 3


class PusherPoseController(LeafSystem):
    def __init__(
        self,
        dynamics_config: SliderPusherSystemConfig,
        mpc_config: HybridMpcConfig,
        closed_loop: bool = True,
    ):
        super().__init__()
        self.object_geometry = dynamics_config.slider.geometry
        self.dynamics_config = dynamics_config
        self.mpc_config = mpc_config
        self.closed_loop = closed_loop

        self._clamped_last_step = False
        self._last_hybrid_mpc_time = 0

        # Internal state
        self._fsm_state_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(
                    PusherPoseControllerState.CFREE_MPC
                    if closed_loop
                    else PusherPoseControllerState.OPEN_LOOP
                )
            )
        )

        self._pusher_slider_contact = self.DeclareVectorInputPort(
            "pusher_slider_contact",
            1,
        )

        self.systems = {
            loc: SliderPusherSystem(loc, dynamics_config)
            for loc in self.object_geometry.contact_locations
        }
        # one controller per face
        self.mpc_controllers = {
            loc: HybridMpc(system, mpc_config, dynamics_config)
            for loc, system in self.systems.items()
        }
        self.pusher_pose_cmd_index = self.DeclareAbstractState(
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0))
        )

        self.pusher_planar_pose_traj = self.DeclareAbstractInputPort(
            "pusher_planar_pose_traj",
            AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
        )
        self.slider_planar_pose_traj = self.DeclareAbstractInputPort(
            "slider_planar_pose_traj",
            AbstractValue.Make([PlanarPose(x=0, y=0, theta=0)]),
        )
        self.contact_force_traj = self.DeclareAbstractInputPort(
            "contact_force_traj",
            AbstractValue.Make([np.array([])]),
        )
        self.contact_mode_traj = self.DeclareAbstractInputPort(
            "contact_mode_traj",
            AbstractValue.Make([PlanarPushingContactMode(0)]),
        )

        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_estimated",
            AbstractValue.Make(RigidTransform()),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose_estimated",
            AbstractValue.Make(RigidTransform()),
        )

        self.output = self.DeclareVectorOutputPort("translation", 2, self.DoCalcOutput)

        # For logging MPC control outputs
        self._mpc_control_index = self.DeclareDiscreteState(3)
        self.DeclareVectorOutputPort("mpc_control", 3, self.RetrieveMpcControl)
        self._mpc_control_desired_index = self.DeclareDiscreteState(3)
        self.DeclareVectorOutputPort(
            "mpc_control_desired", 3, self.RetrieveDesiredMpcControl
        )
        # Run FSM logic before every trajectory-advancing step
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    def _run_fsm_logic(self, context: Context, state: State) -> None:
        """FSM state transition logic."""

        time = context.get_time()

        mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
        fsm_state_value: PlanarPushingContactMode = context.get_abstract_state(
            self._fsm_state_idx
        ).get_value()

        mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
        curr_mode_desired = mode_traj[0]
        EPS = 5e-3

        if fsm_state_value == PusherPoseControllerState.CFREE_MPC:
            signed_dist = self._pusher_slider_contact.Eval(context)
            if curr_mode_desired != PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.HYBRID_MPC)
                self._hybrid_mpc_count = 0
                logger.debug(f"Transitioning to HYBRID_MPC state at time {time}")

            # Unintended collision detection
            in_very_close_contact = signed_dist <= -1e-3
            time_since_last_hybrid_mpc = time - self._last_hybrid_mpc_time
            if (
                all(
                    [
                        current_mode == PlanarPushingContactMode.NO_CONTACT
                        for current_mode in mode_traj
                    ]
                )
                and time_since_last_hybrid_mpc > 0.2
                and in_very_close_contact
            ):
                logger.warn(f"PUSHER SLIDER UNINTENDED COLLISION at time {time}")

        elif fsm_state_value == PusherPoseControllerState.HYBRID_MPC:
            self._last_hybrid_mpc_time = time
            if curr_mode_desired == PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.CFREE_MPC)
                logger.debug(f"Transitioning to CFREE_MPC state at time {time}")
                return
            # Enable/Disable return to contact
            # signed_dist = self._pusher_slider_contact.Eval(context)
            # in_contact = signed_dist <= EPS
            # if not in_contact:
            #     mutable_fsm_state.set_value(PusherPoseControllerState.RETURN_TO_CONTACT)
            #     logger.debug(f"Transitioning to RETURN_TO_CONTACT state at time {time}")
            #     return

        elif fsm_state_value == PusherPoseControllerState.RETURN_TO_CONTACT:
            if curr_mode_desired == PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.CFREE_MPC)
                logger.debug(f"Transitioning to CFREE_MPC state at time {time}")
                return

            signed_dist = self._pusher_slider_contact.Eval(context)
            in_contact = signed_dist <= EPS
            if in_contact:
                mutable_fsm_state.set_value(PusherPoseControllerState.HYBRID_MPC)
                self._hybrid_mpc_count = 0
                logger.debug(f"Transitioning to HYBRID_MPC state at time {time}")
                return

    def _get_mpc_for_mode(self, mode: PlanarPushingContactMode) -> HybridMpc:
        loc = mode.to_contact_location()
        return self.mpc_controllers[loc]

    def _get_system_for_mode(
        self, mode: PlanarPushingContactMode
    ) -> SliderPusherSystem:  # type: ignore
        loc = mode.to_contact_location()
        return self.systems[loc]

    def _call_mpc(
        self,
        curr_slider_pose: PlanarPose,
        curr_pusher_pose: PlanarPose,
        slider_pose_traj: List[PlanarPose],
        pusher_pose_traj: List[PlanarPose],
        contact_force_traj: List[npt.NDArray[np.float64]],
        mode_traj: List[PlanarPushingContactMode],
        pusher_pose_cmd_state: AbstractStateIndex,
        mpc_control_state: AbstractStateIndex,
        mpc_control_desired_state: AbstractStateIndex,
    ) -> PlanarPose:
        mode = mode_traj[0]
        controller = self._get_mpc_for_mode(mode)
        system = self._get_system_for_mode(mode)

        x_traj = [
            system.get_state_from_planar_poses_by_projection(slider_pose, pusher_pose)
            for slider_pose, pusher_pose in zip(slider_pose_traj, pusher_pose_traj)
        ]
        u_traj = [
            system.get_control_from_contact_force(force, slider_pose)
            for force, slider_pose in zip(contact_force_traj, slider_pose_traj)
        ][:-1]
        x_curr = system.get_state_from_planar_poses_by_projection(
            curr_slider_pose, curr_pusher_pose
        )

        modes_eq_to_curr = [m == mode for m in mode_traj]
        if not all(modes_eq_to_curr):
            next_mode_idx = modes_eq_to_curr.index(False)

            # repeat last element of the trajectory that is still in contact
            for idx in range(next_mode_idx, len(x_traj)):
                x_traj[idx] = x_traj[next_mode_idx - 1]

            for idx in range(next_mode_idx, len(u_traj)):
                u_traj[idx] = u_traj[next_mode_idx - 1]
        # else:
        N = len(x_traj)

        # Period between MPC steps
        h = 1 / self.mpc_config.rate_Hz

        # Finite difference method based on pusher position
        pusher_pose_acc = pusher_pose_cmd_state.get_value()
        x_dot_curr, u_input, pusher_vel = controller.compute_control(
            x_curr, x_traj[:N], u_traj[: N - 1]
        )

        next_pusher_pose = PlanarPose(*(pusher_pose_acc.pos() + h * pusher_vel), 0)
        pusher_pose_cmd_state.set_value(next_pusher_pose)
        mpc_control_state.set_value(u_input)
        mpc_control_desired_state.set_value(u_traj[0])

        return next_pusher_pose

    def _call_return_to_contact_controller(
        self,
        curr_slider_pose: PlanarPose,
        curr_pusher_pose: PlanarPose,
        mode_traj: List[PlanarPushingContactMode],
    ) -> PlanarPose:
        mode = mode_traj[0]
        system = self._get_system_for_mode(mode)
        x_desired = system.get_state_from_planar_poses_by_projection(
            slider_pose=curr_slider_pose, pusher_pose=curr_pusher_pose
        )
        next_pusher_pos = system.get_p_WP_from_state(x_desired, buffer=-1e-4)
        next_pusher_pose = PlanarPose(next_pusher_pos[0], next_pusher_pos[1], 0)
        return next_pusher_pose

    def DoCalcOutput(self, context: Context, output):
        pusher_planar_pose_traj: List[PlanarPose] = self.pusher_planar_pose_traj.Eval(context)  # type: ignore
        state = context.get_abstract_state(int(self._fsm_state_idx)).get_value()
        mpc_control_state = context.get_mutable_discrete_state(self._mpc_control_index)
        mpc_control_desired_state = context.get_mutable_discrete_state(
            self._mpc_control_desired_index
        )
        if state == PusherPoseControllerState.OPEN_LOOP:
            curr_planar_pose = pusher_planar_pose_traj[0]
            output.set_value(curr_planar_pose.pos())
            return
        else:
            # Reset the MPC controller integrator (set commanded position to current position)
            pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
            pusher_planar_pose = PlanarPose.from_pose(pusher_pose)
            slider_pose: RigidTransform = self.slider_pose.Eval(context)  # type: ignore
            slider_planar_pose = PlanarPose.from_pose(slider_pose)
            pusher_pose_cmd_state = context.get_mutable_abstract_state(
                self.pusher_pose_cmd_index
            )

        if state == PusherPoseControllerState.CFREE_MPC:
            curr_planar_pose = pusher_planar_pose_traj[0]
            output.set_value(curr_planar_pose.pos())
            mpc_control_state.SetFromVector([0, 0, 0])
            mpc_control_desired_state.SetFromVector([0, 0, 0])

        elif state == PusherPoseControllerState.HYBRID_MPC:
            mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore
            contact_force_traj: List[npt.NDArray[np.float64]] = self.contact_force_traj.Eval(context)  # type: ignore
            if self._hybrid_mpc_count == 0:
                logger.debug(f"Resetting accumulator at time {context.get_time()}")
                pusher_pose_cmd_state.set_value(pusher_planar_pose)
            self._hybrid_mpc_count += 1

            next_pusher_pose = self._call_mpc(
                slider_planar_pose,
                pusher_planar_pose,
                slider_planar_pose_traj,
                pusher_planar_pose_traj,
                contact_force_traj,
                mode_traj,
                pusher_pose_cmd_state=pusher_pose_cmd_state,
                mpc_control_state=mpc_control_state,
                mpc_control_desired_state=mpc_control_desired_state,
            )

            output.set_value(next_pusher_pose.pos())

        elif state == PusherPoseControllerState.RETURN_TO_CONTACT:
            mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore
            contact_force_traj: List[npt.NDArray[np.float64]] = self.contact_force_traj.Eval(context)  # type: ignore
            next_pusher_pose = self._call_return_to_contact_controller(
                slider_planar_pose,
                pusher_planar_pose,
                mode_traj,
            )
            mpc_control_state.set_value([0, 0, 0])

            output.set_value(next_pusher_pose.pos())

    def RetrieveMpcControl(self, context: Context, output):
        mpc_control_state = context.get_discrete_state(self._mpc_control_index)
        output.SetFromVector(mpc_control_state.get_value())

    def RetrieveDesiredMpcControl(self, context: Context, output):
        mpc_control_desired_state = context.get_discrete_state(
            self._mpc_control_desired_index
        )
        output.SetFromVector(mpc_control_desired_state.get_value())
