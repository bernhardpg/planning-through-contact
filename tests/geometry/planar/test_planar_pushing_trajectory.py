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


def _get_state_from_path_knot_points(
    vars: FaceContactVariables, idx: int
) -> npt.NDArray[np.float64]:
    x = vars.p_WB_xs[idx]
    y = vars.p_WB_ys[idx]
    lam = vars.lams[idx]
    cos_th = vars.cos_ths[idx]
    th = np.arccos(cos_th)

    return np.array([x, y, lam, th])


def _pose_is_close(pose1: PlanarPose, pose2: PlanarPose) -> bool:
    pos_is_close = np.allclose(pose1.pos(), pose2.pos())
    th_is_close = np.isclose(pose1.theta, pose2.theta)
    return pos_is_close and th_is_close  # type: ignore


def test_planar_pushing_trajectory_construction() -> None:
    # Create test objects

    box_geometry = Box2d(width=0.3, height=0.3)
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    cfg = SliderPusherSystemConfig(
        slider=box,
        pusher_radius=0.05,
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
    contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)
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

    sys = SliderPusherSystem(contact_location, cfg)

    # Check starting values
    t = traj.start_time
    pusher_pose_from_traj = traj.get_pusher_planar_pose(t)
    slider_pose_from_traj = traj.get_slider_planar_pose(t)

    # Make sure the starting position is correct
    assert _pose_is_close(slider_pose_from_traj, initial_pose)

    state = _get_state_from_path_knot_points(vars, 0)
    pusher_pose_from_state = sys.get_pusher_planar_pose_from_state(state)
    assert _pose_is_close(pusher_pose_from_traj, pusher_pose_from_state)

    breakpoint()
