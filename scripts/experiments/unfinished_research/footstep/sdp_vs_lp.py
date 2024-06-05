from pathlib import Path

import numpy as np

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_planner import FootstepPlanner
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanResult,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain
from planning_through_contact.tools.script_utils import default_script_setup


def plan_with_one_stone(
    use_lp: bool, output_dir: Path, debug: bool = False
) -> FootstepPlanResult:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=1.0, z_pos=0.2, name="initial")

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot, use_lp_approx=use_lp)

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    desired_displacement = 0.2
    initial_pos = (
        np.array([stone.x_pos - desired_displacement, stone.z_pos]) + desired_robot_pos
    )
    target_pos = (
        np.array([stone.x_pos + desired_displacement, stone.z_pos]) + desired_robot_pos
    )

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=stone.name,
        target_stone_name=stone.name,
    )

    planner.plan(print_flows=debug, print_solver_output=debug, print_debug=debug)
    if use_lp:
        name = "lp"
    else:
        name = "sdp"

    planner.save_analysis(str(output_dir / name))

    best_result = planner.get_best_result()
    return best_result


def main(output_dir: Path, debug: bool = False) -> None:
    result_lp = plan_with_one_stone(True, output_dir, debug)
    logger.info(f"LP relaxed solve time: {result_lp.relaxed_metrics.solve_time:.3f} s")
    logger.info(f"LP rounding time: {result_lp.rounded_metrics.solve_time:.3f} s")
    result_sdp = plan_with_one_stone(False, output_dir, debug)
    logger.info(
        f"SDP relaxed solve time: {result_sdp.relaxed_metrics.solve_time:.3f} s"
    )
    logger.info(f"SDP rounding time: {result_sdp.rounded_metrics.solve_time:.3f} s")


if __name__ == "__main__":
    debug, output_dir, logger = default_script_setup()
    main(output_dir, debug)
