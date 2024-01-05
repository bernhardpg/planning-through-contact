from enum import Enum
from typing import List, Optional
import logging

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import (
    AbstractStateIndex,
    Context,
    DiagramBuilder,
    InputPort,
    LeafSystem,
    OutputPort,
    AbstractStateIndex,
    State,
)
from pydrake.systems.primitives import ZeroOrderHold

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

        # Internal state
        self._fsm_state_idx = int(
            self.DeclareAbstractState(
                AbstractValue.Make(PusherPoseControllerState.CFREE_MPC if closed_loop 
                                   else PusherPoseControllerState.OPEN_LOOP)
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
        self.slider_pose_cmd_index = self.DeclareAbstractState(
            AbstractValue.Make(PlanarPose(x=0, y=0, theta=0))
        )
        self.x_acc_index = self.DeclareAbstractState(
            AbstractValue.Make(np.zeros(4))
        ) # state: x = [x, y, theta, lam]

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
        self.output = self.DeclareVectorOutputPort("translation", 2,  self.DoCalcOutput)

        self.closed_loop = closed_loop
        if self.closed_loop:
            self.pusher_pose_measured = self.DeclareAbstractInputPort(
                "pusher_pose_measured",
                AbstractValue.Make(RigidTransform()),
            )
            self.slider_pose = self.DeclareAbstractInputPort(
                "slider_pose",
                AbstractValue.Make(RigidTransform()),
            )

        # Run FSM logic before every trajectory-advancing step
        self.DeclarePerStepUnrestrictedUpdateEvent(self._run_fsm_logic)

    @classmethod
    def AddToBuilder(
        cls,
        builder: DiagramBuilder,
        dynamics_config: SliderPusherSystemConfig,
        mpc_config: HybridMpcConfig,
        contact_mode_traj: OutputPort,
        slider_planar_pose_traj: OutputPort,
        pusher_planar_pose_traj: OutputPort,
        contact_force_traj: OutputPort,
        pose_cmd: Optional[InputPort]=None,
        closed_loop: bool = True,
        pusher_planar_pose_measured: Optional[OutputPort] = None,
        slider_pose_measured: Optional[OutputPort] = None,
    ) -> "PusherPoseController":
        pusher_pose_controller = builder.AddNamedSystem(
            "PusherPoseController",
            cls(
                dynamics_config,
                mpc_config,
                closed_loop=closed_loop,
            ),
        )

        builder.Connect(
            contact_mode_traj,
            pusher_pose_controller.GetInputPort("contact_mode_traj"),
        )
        builder.Connect(
            pusher_planar_pose_traj,
            pusher_pose_controller.GetInputPort("pusher_planar_pose_traj"),
        )
        builder.Connect(
            slider_planar_pose_traj,
            pusher_pose_controller.GetInputPort("slider_planar_pose_traj"),
        )
        builder.Connect(
            contact_force_traj,
            pusher_pose_controller.GetInputPort("contact_force_traj"),
        )

        if closed_loop:
            assert pusher_planar_pose_measured is not None
            assert slider_pose_measured is not None

            builder.Connect(
                slider_pose_measured,
                pusher_pose_controller.GetInputPort("slider_pose"),
            )
            builder.Connect(
                pusher_planar_pose_measured,
                pusher_pose_controller.GetInputPort("pusher_pose_measured"),
            )

        period = 1 / mpc_config.rate_Hz
        zero_order_hold = builder.AddNamedSystem(
            "ZeroOrderHold", ZeroOrderHold(period, vector_size=2) # Just the x and y positions
        )
        builder.Connect(
            pusher_pose_controller.get_output_port(), zero_order_hold.get_input_port()
        )
        if pose_cmd is not None:
            builder.Connect(zero_order_hold.get_output_port(), pose_cmd)
            # Otherwise connect the output somewhere else
        return pusher_pose_controller

    def _run_fsm_logic(self, context: Context, state: State) -> None:
        """FSM state transition logic."""
        
        time = context.get_time()

        mutable_fsm_state = state.get_mutable_abstract_state(self._fsm_state_idx)
        fsm_state_value: PlanarPushingContactMode = context.get_abstract_state(
            self._fsm_state_idx
        ).get_value()

        mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
        curr_mode_desired = mode_traj[0]

        if fsm_state_value == PusherPoseControllerState.CFREE_MPC:
            if curr_mode_desired != PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.HYBRID_MPC)
                self._hybrid_mpc_count = 0
                logger.debug(f"Transitioning to HYBRID_MPC state at time {time}")
        
        elif fsm_state_value == PusherPoseControllerState.HYBRID_MPC:
            if curr_mode_desired == PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.CFREE_MPC)
                logger.debug(f"Transitioning to CFREE_MPC state at time {time}")
                return
            
            in_contact = self._pusher_slider_contact.Eval(context)
            if not in_contact:
                mutable_fsm_state.set_value(PusherPoseControllerState.RETURN_TO_CONTACT)
                logger.debug(f"Transitioning to RETURN_TO_CONTACT state at time {time}")
                return
            
        
        elif fsm_state_value == PusherPoseControllerState.RETURN_TO_CONTACT:
            if curr_mode_desired == PlanarPushingContactMode.NO_CONTACT:
                mutable_fsm_state.set_value(PusherPoseControllerState.CFREE_MPC)
                logger.debug(f"Transitioning to CFREE_MPC state at time {time}")
                return
            
            in_contact = self._pusher_slider_contact.Eval(context)
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
        pusher_pose_cmd_state: Optional[AbstractStateIndex] = None,
        slider_pose_cmd_state: Optional[AbstractStateIndex] = None,
        x_acc_state: Optional[AbstractStateIndex] = None,
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
            N = modes_eq_to_curr.index(False)

            # repeat last element of the trajectory that is still in contact
            for idx in range(N, len(x_traj)):
                x_traj[idx] = x_traj[N - 1]

            for idx in range(N, len(u_traj)):
                u_traj[idx] = u_traj[N - 1]
        else:
            N = len(x_traj)

        x_dot_curr, u_input = controller.compute_control(
            x_curr, x_traj[: N + 1], u_traj[:N]
        )

        h = 1 / self.mpc_config.rate_Hz
        x_acc = system.get_state_from_planar_poses_by_projection(
            slider_pose_cmd_state.get_value(), pusher_pose_cmd_state.get_value()
        )
        x_at_next_mpc_step = x_acc + h * x_dot_curr
        # For now this is used in _call_return_to_contact_controller, we just want to save the last lambda.
        x_acc_state.set_value(x_at_next_mpc_step)
        next_slider_pose = PlanarPose(*(x_at_next_mpc_step[0:3]))
        next_pusher_pos = system.get_p_WP_from_state(x_at_next_mpc_step).flatten()
        next_pusher_pose = PlanarPose(next_pusher_pos[0], next_pusher_pos[1], 0)
        pusher_pose_cmd_state.set_value(next_pusher_pose)
        slider_pose_cmd_state.set_value(next_slider_pose)
        return next_pusher_pose
    
    def _call_return_to_contact_controller(
        self,
        curr_slider_pose: PlanarPose,
        curr_pusher_pose: PlanarPose,
        x_acc_state: AbstractStateIndex,
        slider_pose_traj: List[PlanarPose],
        pusher_pose_traj: List[PlanarPose],
        mode_traj: List[PlanarPushingContactMode],
    ) -> PlanarPose:
        mode = mode_traj[0]
        system = self._get_system_for_mode(mode)
        x_desired = x_acc_state.get_value() # The only value we want is the last lambda
        x_desired[0:3] = curr_slider_pose.vector() # Set the rest of the values to the current slider pose
        next_pusher_pose = system.get_pusher_planar_pose_from_state(x_desired, buffer=-1e-3)
        diff = next_pusher_pose.pos()-curr_pusher_pose.pos()
        magnitude = np.linalg.norm(diff)
        if magnitude > 0.1:
            logger.warn(f"magnitude of pusher pose change: {magnitude}")
        return next_pusher_pose

    def DoCalcOutput(self, context: Context, output):
        
        pusher_planar_pose_traj: List[PlanarPose] = self.pusher_planar_pose_traj.Eval(context)  # type: ignore
        state = context.get_abstract_state(int(self._fsm_state_idx)).get_value()

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
            slider_pose_cmd_state = context.get_mutable_abstract_state(
                self.slider_pose_cmd_index
            )
            x_acc_state = context.get_mutable_abstract_state(self.x_acc_index)
        if state == PusherPoseControllerState.CFREE_MPC:
            curr_planar_pose = pusher_planar_pose_traj[0]
            output.set_value(curr_planar_pose.pos())
            # Reset the MPC controller integrator (set commanded position to current position)
            pusher_pose_cmd_state.set_value(pusher_planar_pose)
            slider_pose_cmd_state.set_value(slider_planar_pose)
            
        elif state == PusherPoseControllerState.HYBRID_MPC:
            # Reset accumulator after every 10 steps:
            if self._hybrid_mpc_count % 10 == 0:
                logger.debug(f"Resetting accumulator at time {context.get_time()}")
                pusher_pose_cmd_state.set_value(pusher_planar_pose)
                slider_pose_cmd_state.set_value(slider_planar_pose)
            self._hybrid_mpc_count += 1
            mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore
            contact_force_traj: List[npt.NDArray[np.float64]] = self.contact_force_traj.Eval(context)  # type: ignore
            next_pusher_pose = self._call_mpc(
                slider_planar_pose,
                pusher_planar_pose,
                slider_planar_pose_traj,
                pusher_planar_pose_traj,
                contact_force_traj,
                mode_traj,
                pusher_pose_cmd_state=pusher_pose_cmd_state,
                slider_pose_cmd_state=slider_pose_cmd_state,
                x_acc_state=x_acc_state,
            )
            output.set_value(next_pusher_pose.pos())
        
        elif state == PusherPoseControllerState.RETURN_TO_CONTACT:
            pusher_pose_cmd_state.set_value(pusher_planar_pose)
            slider_pose_cmd_state.set_value(slider_planar_pose)

            mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore
            contact_force_traj: List[npt.NDArray[np.float64]] = self.contact_force_traj.Eval(context)  # type: ignore
            # next_pusher_pose = self._call_mpc(
            #     slider_planar_pose,
            #     pusher_planar_pose,
            #     slider_planar_pose_traj,
            #     pusher_planar_pose_traj,
            #     contact_force_traj,
            #     mode_traj,
            #     pusher_pose_cmd_state=pusher_pose_cmd_state,
            #     slider_pose_cmd_state=slider_pose_cmd_state,
            #     x_acc_state=x_acc_state,
            # )
            # Does not work well
            next_pusher_pose = self._call_return_to_contact_controller(
                slider_planar_pose,
                pusher_planar_pose,
                x_acc_state,
                slider_planar_pose_traj,
                pusher_planar_pose_traj,
                mode_traj,
            )
            output.set_value(next_pusher_pose.pos())
        
            
