import numpy as np
import numpy.typing as npt
from pydrake.solvers import MosekSolver

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCostType,
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)
from planning_through_contact.simulation.systems.slider_pusher_trajectory_feeder import (
    SliderPusherTrajSegment,
)


def _pose_is_close(pose1: PlanarPose, pose2: PlanarPose, tol=1e-6) -> bool:
    pos_is_close = np.allclose(pose1.pos(), pose2.pos(), atol=tol)
    th_is_close = np.isclose(pose1.theta, pose2.theta, atol=tol)
    return pos_is_close and th_is_close  # type: ignore


def test_planar_pushing_trajectory_values_no_rotation() -> None:
    # Create test objects

    box_geometry = Box2d(width=0.3, height=0.3)
    pusher_radius = 0.05
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    cfg = SliderPusherSystemConfig(
        slider=box,
        pusher_radius=pusher_radius,
        friction_coeff_slider_pusher=0.5,
        friction_coeff_table_slider=0.5,
        integration_constant=0.7,
    )
    contact_config = ContactConfig(cost_type=ContactCostType.KEYPOINT_DISPLACEMENTS)
    plan_cfg = PlanarPlanConfig(
        dynamics_config=cfg,
        num_knot_points_contact=4,
        use_approx_exponential_map=False,
        use_band_sparsity=False,
        contact_config=contact_config,
    )
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)  # left face
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(-0.3, 0, 0)

    mode = FaceContactMode.create_from_plan_spec(contact_location, plan_cfg)
    mode.set_slider_initial_pose(initial_pose)
    mode.set_slider_final_pose(final_pose)

    mode.formulate_convex_relaxation()
    solver = MosekSolver()
    result = solver.Solve(mode.relaxed_prog)  # type: ignore
    assert result.is_success()

    vars = mode.variables.eval_result(result)
    traj = PlanarPushingTrajectory(plan_cfg, [vars])

    slider_pusher_seg = traj.slider_pusher_traj_segments[0]
    assert slider_pusher_seg is not None

    sys = SliderPusherSystem(contact_location, cfg)

    # Make sure the slider boundary position is correct
    assert _pose_is_close(traj.get_slider_planar_pose(traj.start_time), initial_pose)
    assert _pose_is_close(traj.get_slider_planar_pose(traj.end_time), final_pose)

    def _assert_values_match(t: float) -> None:
        pusher_pose_from_traj = traj.get_pusher_planar_pose(t)
        slider_pose_from_traj = traj.get_slider_planar_pose(t)
        state = slider_pusher_seg.eval_state(t)

        assert np.isclose(slider_pose_from_traj.x, state[0])
        assert np.isclose(slider_pose_from_traj.y, state[1])
        assert np.isclose(slider_pose_from_traj.theta, state[2], atol=1e-6)

        # For now we don't check rotations
        assert np.isclose(slider_pose_from_traj.theta, 0)

        pusher_pose_from_state = sys.get_pusher_planar_pose_from_state(state)
        assert _pose_is_close(pusher_pose_from_traj, pusher_pose_from_state)

    # Check values at times
    times = np.linspace(traj.start_time, traj.end_time, 10)
    for t in times:
        _assert_values_match(t)

    breakpoint()
