import pytest

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    PlanarTrajectoryBuilder,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import box_geometry, planner, rigid_body_box


@pytest.mark.parametrize(
    "planner",
    [
        (
            {
                "partial": True,
                "allow_teleportation": True,
                "penalize_mode_transition": False,
                "boundary_conds": {
                    "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                    "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                    "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                    "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
                },
            }
        ),
    ],
    indirect=["planner"],
)
def test_path_with_teleportation(planner: PlanarPushingPlanner) -> None:
    result = planner._solve(print_output=False)
    assert result.is_success()
    path = planner.get_solution_path(result)

    # We should always have one more vertex than edge in a solution
    assert len(path.pairs) == len(path.edges) + 1

    prog = path._construct_nonlinear_program_from_path()

    expected_num_vars = sum([p.mode.prog.num_vars() for p in path.pairs])
    assert prog.num_vars() == expected_num_vars

    expected_num_constraints = sum(
        [len(p.mode.prog.GetAllConstraints()) for p in path.pairs]
    ) + sum([len(edge.GetConstraints()) for edge in path.edges])
    assert len(prog.GetAllConstraints()) == expected_num_constraints


@pytest.mark.parametrize(
    "planner",
    [
        (
            {
                "partial": True,
                "allow_teleportation": True,
                "penalize_mode_transition": False,
                "boundary_conds": {
                    "finger_initial_pose": PlanarPose(x=0, y=-0.5, theta=0.0),
                    "finger_target_pose": PlanarPose(x=-0.3, y=0, theta=0.0),
                    "box_initial_pose": PlanarPose(x=0, y=0, theta=0.0),
                    "box_target_pose": PlanarPose(x=-0.2, y=-0.2, theta=0.4),
                },
            }
        ),
    ],
    indirect=["planner"],
)
def test_path_rounding(planner: PlanarPushingPlanner) -> None:
    result = planner._solve(print_output=False)
    assert result.is_success()

    path = planner.get_solution_path(result)
    rounded_result = path._do_nonlinear_rounding()
    assert rounded_result.is_success()

    vars_on_path = path.get_rounded_vars()
    traj = PlanarTrajectoryBuilder(vars_on_path).get_trajectory(interpolate=False)

    DEBUG = True
    if DEBUG:
        visualize_planar_pushing_trajectory(traj, planner.slider.geometry)
