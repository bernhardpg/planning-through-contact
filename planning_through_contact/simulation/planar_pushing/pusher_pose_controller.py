from typing import List

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
        object_geometry: CollisionGeometry,
        mpc_config: HybridMpcConfig = HybridMpcConfig(),
        z_dist_to_table: float = 0.5,
    ):
        super().__init__()
        self.z_dist = z_dist_to_table
        self.object_geometry = object_geometry

        self.systems = {
            loc: SliderPusherSystem(object_geometry, loc)
            for loc in object_geometry.contact_locations
        }
        # one controller per face
        self.mpc_controllers = {
            loc: HybridMpc(system, mpc_config) for loc, system in self.systems.items()
        }
        self.mpc_config = mpc_config

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
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.contact_mode_desired = self.DeclareAbstractInputPort(
            "contact_mode_desired",
            AbstractValue.Make(PlanarPushingContactMode(0)),
        )
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose",
            AbstractValue.Make(RigidTransform()),
        )
        self.output = self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput
        )

    @classmethod
    def AddToBuilder(
        cls,
        builder: DiagramBuilder,
        slider: RigidBody,
        mpc_config: HybridMpcConfig,
        contact_mode_desired: OutputPort,
        slider_planar_pose_traj: OutputPort,
        pusher_planar_pose_traj: OutputPort,
        contact_force_traj: OutputPort,
        pusher_planar_pose_measured: OutputPort,
        slider_pose_measured: OutputPort,
        pose_cmd: InputPort,
    ) -> "PusherPoseController":
        pusher_pose_controller = builder.AddNamedSystem(
            "PusherPoseController",
            cls(slider.geometry, mpc_config, z_dist_to_table=0.02),
        )

        builder.Connect(
            contact_mode_desired,
            pusher_pose_controller.GetInputPort("contact_mode_desired"),
        )
        builder.Connect(
            pusher_planar_pose_measured,
            pusher_pose_controller.GetInputPort("pusher_pose_measured"),
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
        builder.Connect(
            slider_pose_measured,
            pusher_pose_controller.GetInputPort("slider_pose"),
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

    def _compute_control(
        self,
        theta: float,
        theta_desired: float,
        p_W_c: npt.NDArray[np.float64],
        p_WB: npt.NDArray[np.float64],
        loc: PolytopeContactLocation,
    ) -> npt.NDArray[np.float64]:
        theta_error = theta_desired - theta

        K_P = 0.5

        # Commanded difference in position along contact face
        delta_lam = -K_P * theta_error

        pv1, pv2 = self.object_geometry.get_proximate_vertices_from_location(loc)

        # TODO(bernhardpg): consider not normalizing this
        dir_vector = (pv2 - pv1) / np.linalg.norm(pv2 - pv1)
        delta_p_B_c = delta_lam * dir_vector

        R_WB = two_d_rotation_matrix_from_angle(theta)
        p_B_c = R_WB.T.dot(p_W_c - p_WB)

        p_B_c_cmd = p_B_c + delta_p_B_c
        cmd_between_pv1_and_pv2 = (
            np.cross(pv2.flatten(), p_B_c_cmd.flatten()) >= 0
            and np.cross(pv1.flatten(), p_B_c_cmd.flatten()) <= 0
        )
        if not cmd_between_pv1_and_pv2:  # saturate control
            print("saturate")
            delta_p_B_c = np.zeros((2, 1))

        # We only care about the diffs here, i.e. no constant translation
        R_WB_desired = two_d_rotation_matrix_from_angle(theta_desired)
        delta_p_W_c = R_WB_desired.dot(delta_p_B_c)

        return delta_p_W_c

    def _get_mpc_for_mode(self, mode: PlanarPushingContactMode) -> HybridMpc:
        loc = mode.to_contact_location()
        return self.mpc_controllers[loc]

    def _get_system_for_mode(
        self, mode: PlanarPushingContactMode
    ) -> SliderPusherSystem:  # type: ignore
        loc = mode.to_contact_location()
        return self.systems[loc]

    def _get_next_p_B_c_from_lam_dot(
        self, lam_dot: float, lam_curr: float, loc: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        h = 1 / self.mpc_config.rate_Hz
        lam_next = lam_curr + h * lam_dot

        p_B_c_next = self.object_geometry.get_p_B_c_from_lam(lam_next, loc)
        return p_B_c_next

    def _get_p_W_c_from_p_B_c(
        self, curr_slider_pose: PlanarPose, p_B_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        R_WB = two_d_rotation_matrix_from_angle(curr_slider_pose.theta)
        p_WB = curr_slider_pose.pos()
        p_W_c = p_WB + R_WB.dot(p_B_c)
        return p_W_c

    def _call_mpc(
        self,
        curr_slider_pose: PlanarPose,
        curr_pusher_pose: PlanarPose,
        slider_pose_traj: List[PlanarPose],
        pusher_pose_traj: List[PlanarPose],
        contact_force_traj: List[npt.NDArray[np.float64]],
        mode: PlanarPushingContactMode,
    ) -> PlanarPose:
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

        x_next, _ = controller.compute_control(x_curr, x_traj, u_traj)
        next_pusher_pose = system.get_pusher_planar_pose_from_state(x_next)
        return next_pusher_pose

    def DoCalcOutput(self, context: Context, output):
        mode_desired: PlanarPushingContactMode = self.contact_mode_desired.Eval(context)  # type: ignore
        pusher_planar_pose_traj: List[PlanarPose] = self.pusher_planar_pose_traj.Eval(context)  # type: ignore

        if mode_desired == PlanarPushingContactMode.NO_CONTACT:
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
                mode_desired,
            )

            pusher_pose_command = pusher_planar_pose_cmd.to_pose(z_value=self.z_dist)
            output.set_value(pusher_pose_command)
