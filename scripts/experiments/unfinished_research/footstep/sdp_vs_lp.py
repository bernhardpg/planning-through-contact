from pathlib import Path
from typing import List, Tuple

import numpy as np

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_planner import FootstepPlanner
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanResult,
)
from planning_through_contact.planning.footstep.in_plane_terrain import (
    InPlaneSteppingStone,
    InPlaneTerrain,
)
from planning_through_contact.tools.script_utils import default_script_setup

TerrainStoneTuple = Tuple[InPlaneTerrain, InPlaneSteppingStone, InPlaneSteppingStone]


def get_one_stone_terrain() -> TerrainStoneTuple:
    terrain = InPlaneTerrain()
    stone = terrain.add_stone(x_pos=1.0, width=1.0, z_pos=0.2, name="initial")
    return terrain, stone, stone


def get_complex_terrain() -> TerrainStoneTuple:
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.25, width=0.5, z_pos=0.2, name="initial")
    terrain.add_stone(x_pos=1.0, width=1.0, z_pos=0.5, name="stone_2")
    terrain.add_stone(x_pos=2.0, width=1.0, z_pos=0.7, name="stone_3")
    terrain.add_stone(x_pos=2.75, width=0.5, z_pos=0.9, name="stone_4")
    terrain.add_stone(x_pos=3.25, width=0.5, z_pos=0.7, name="stone_5")
    target_stone = terrain.add_stone(x_pos=3.75, width=0.5, z_pos=0.5, name="target")
    return terrain, initial_stone, target_stone


def plan(
    terrain_and_stones: TerrainStoneTuple,
    use_lp: bool,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[FootstepPlanResult, List[FootstepPlanResult]]:

    robot = PotatoRobot()
    cfg = FootstepPlanningConfig(robot=robot, use_lp_approx=use_lp)

    terrain, initial_stone, target_stone = terrain_and_stones

    desired_robot_pos = np.array([0.0, cfg.robot.desired_com_height])
    desired_displacement = 0.2
    initial_pos = (
        np.array([initial_stone.x_pos - desired_displacement, initial_stone.z_pos])
        + desired_robot_pos
    )
    target_pos = (
        np.array([target_stone.x_pos + desired_displacement, target_stone.z_pos])
        + desired_robot_pos
    )

    initial_pose = np.concatenate([initial_pos, [0]])
    target_pose = np.concatenate([target_pos, [0]])

    planner = FootstepPlanner(
        cfg,
        terrain,
        initial_pose,
        target_pose,
        initial_stone_name=initial_stone.name,
        target_stone_name=target_stone.name,
    )

    planner.plan(print_flows=debug, print_solver_output=debug, print_debug=debug)
    if use_lp:
        name = "lp"
    else:
        name = "sdp"

    planner.save_analysis(str(output_dir / name))

    best_result = planner.get_best_result()
    results = planner.get_results()
    return best_result, results


def main(output_dir: Path, debug: bool = False) -> None:
    terrains = [get_one_stone_terrain(), get_complex_terrain()]
    names = ["one_stone", "complex"]

    for terrain, name in zip(terrains, names):
        logger.info(f"## Terrain: {name} ##")
        result_lp, results_lp = plan(terrain, True, output_dir / name, debug)
        assert result_lp.gcs_metrics is not None

        result_sdp, results_sdp = plan(terrain, False, output_dir / name, debug)
        assert result_sdp.gcs_metrics is not None

        logger.info(f" - LP num found paths: {len(results_lp)}")
        logger.info(f" - SDP num found paths: {len(results_sdp)}\n")

        logger.info(f" - LP best path length: {result_lp.num_modes}")
        logger.info(f" - SDP best path length: {result_sdp.num_modes}\n")

        logger.info(f" - LP GCS time: {result_lp.gcs_metrics.solve_time:.3f} s")
        logger.info(f" - SDP GCS time: {result_sdp.gcs_metrics.solve_time :.3f} s\n")

        logger.info(
            f" - LP relaxed solve time: {result_lp.relaxed_metrics.solve_time:.3f} s"
        )
        logger.info(
            f" - SDP relaxed solve time: {result_sdp.relaxed_metrics.solve_time:.3f} s\n"
        )

        logger.info(
            f" - LP rounding time: {result_lp.rounded_metrics.solve_time:.3f} s"
        )
        logger.info(
            f" - SDP rounding time: {result_sdp.rounded_metrics.solve_time:.3f} s\n"
        )

        logger.info(f" - LP rounded cost: {result_lp.rounded_metrics.cost :.3f}")
        logger.info(f" - SDP rounded cost: {result_lp.rounded_metrics.cost :.3f}\n")

        logger.info(f" - LP opt_gap: {result_lp.ub_relaxation_gap_pct :.3f} %")
        logger.info(f" - SDP opt_gap: {result_lp.ub_relaxation_gap_pct :.3f} %\n")


if __name__ == "__main__":
    debug, output_dir, logger = default_script_setup()
    main(output_dir, debug)
