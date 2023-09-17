from typing import List, Optional

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import (
    Context,
    DiagramBuilder,
    InputPort,
    LeafSystem,
    OutputPort,
)
from pydrake.systems.primitives import ZeroOrderHold

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
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


class PusherPoseController(LeafSystem):
    def __init__(
        self,
        dynamics_config: SliderPusherSystemConfig,
        mpc_config: HybridMpcConfig,
        z_dist_to_table: float = 0.5,
        closed_loop: bool = True,
    ):
        super().__init__()
        self.z_dist = z_dist_to_table
        self.object_geometry = dynamics_config.slider.geometry
        self.dynamics_config = dynamics_config
        self.mpc_config = mpc_config

        self.systems = {
            loc: SliderPusherSystem(loc, dynamics_config)
            for loc in self.object_geometry.contact_locations
        }
        # one controller per face
        self.mpc_controllers = {
            loc: HybridMpc(system, mpc_config, dynamics_config)
            for loc, system in self.systems.items()
        }

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
        self.output = self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

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
        pose_cmd: InputPort,
        closed_loop: bool = True,
        pusher_planar_pose_measured: Optional[OutputPort] = None,
        slider_pose_measured: Optional[OutputPort] = None,
    ) -> "PusherPoseController":
        pusher_pose_controller = builder.AddNamedSystem(
            "PusherPoseController",
            cls(
                dynamics_config,
                mpc_config,
                z_dist_to_table=0.02,
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
            "ZeroOrderHold", ZeroOrderHold(period, AbstractValue.Make(RigidTransform()))
        )
        builder.Connect(
            pusher_pose_controller.get_output_port(), zero_order_hold.get_input_port()
        )
        builder.Connect(zero_order_hold.get_output_port(), pose_cmd)
        return pusher_pose_controller

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
    ) -> PlanarPose:
        mode = mode_traj[0]
        controller = self._get_mpc_for_mode(mode)
        system = self._get_system_for_mode(mode)

        x_traj = [
            system.get_state_from_planar_poses(slider_pose, pusher_pose)
            for slider_pose, pusher_pose in zip(slider_pose_traj, pusher_pose_traj)
        ]
        u_traj = [
            system.get_control_from_contact_force(force, slider_pose)
            for force, slider_pose in zip(contact_force_traj, slider_pose_traj)
        ]
        x_curr = system.get_state_from_planar_poses(curr_slider_pose, curr_pusher_pose)

        # modes_eq_to_curr = [m == mode for m in mode_traj]
        # if not all(modes_eq_to_curr):
        #     N = modes_eq_to_curr.index(False)
        #
        #     # repeat last element of the trajectory that is still in contact
        #     for idx in range(N, len(x_traj)):
        #         x_traj[idx] = x_traj[N - 1]
        #
        #     for idx in range(N, len(u_traj)):
        #         u_traj[idx] = u_traj[N - 1]
        # else:
        #     N = len(x_traj)

        x_dot_curr, u_input = controller.compute_control(x_curr, x_traj, u_traj)

        h = 1 / self.mpc_config.rate_Hz
        x_at_next_mpc_step = x_curr + h * x_dot_curr
        next_pusher_pose = system.get_pusher_planar_pose_from_state(x_at_next_mpc_step)
        return next_pusher_pose

    def DoCalcOutput(self, context: Context, output):
        mode_traj: List[PlanarPushingContactMode] = self.contact_mode_traj.Eval(context)  # type: ignore
        curr_mode_desired = mode_traj[0]
        pusher_planar_pose_traj: List[PlanarPose] = self.pusher_planar_pose_traj.Eval(context)  # type: ignore

        if (
            not self.closed_loop
            or curr_mode_desired == PlanarPushingContactMode.NO_CONTACT
        ):
            curr_planar_pose = pusher_planar_pose_traj[0]
            pusher_pose_desired = curr_planar_pose.to_pose(z_value=self.z_dist)
            output.set_value(pusher_pose_desired)
        else:  # do control of angle
            slider_pose: RigidTransform = self.slider_pose.Eval(context)  # type: ignore

            slider_planar_pose = PlanarPose.from_pose(slider_pose)
            slider_planar_pose_traj: List[PlanarPose] = self.slider_planar_pose_traj.Eval(context)  # type: ignore

            pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
            pusher_planar_pose = PlanarPose.from_pose(pusher_pose)

            contact_force_traj: List[npt.NDArray[np.float64]] = self.contact_force_traj.Eval(context)  # type: ignore
            pusher_planar_pose_cmd = self._call_mpc(
                slider_planar_pose,
                pusher_planar_pose,
                slider_planar_pose_traj,
                pusher_planar_pose_traj,
                contact_force_traj,
                mode_traj,
            )

            pusher_pose_command = pusher_planar_pose_cmd.to_pose(z_value=self.z_dist)
            output.set_value(pusher_pose_command)
