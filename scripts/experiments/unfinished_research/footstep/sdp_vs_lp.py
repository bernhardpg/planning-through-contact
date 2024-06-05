import argparse
from pathlib import Path

import numpy as np
import pytest
from pydrake.solvers import (  # CommonSolverOption,
    Binding,
    CommonSolverOption,
    MosekSolver,
    PositiveSemidefiniteConstraint,
    SolutionResult,
    Solve,
    SolverOptions,
)

from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.footstep_planner import FootstepPlanner
from planning_through_contact.planning.footstep.footstep_trajectory import (
    FootstepPlanSegmentProgram,
    get_X_from_semidefinite_relaxation,
)
from planning_through_contact.planning.footstep.in_plane_terrain import InPlaneTerrain
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.footstep_visualizer import animate_footstep_plan


# TODO(bernhardpg): Move these to a script utils folder
def make_output_folder() -> Path:
    curr_name = Path(__file__).name.split(".")[0]
    high_level_output_dir = Path("SCRIPT_OUTPUTS")
    output_dir = high_level_output_dir / curr_name

    from datetime import datetime

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H%M%S")

    unique_output_dir = output_dir / timestamp_str
    unique_output_dir.mkdir(exist_ok=True, parents=True)

    return unique_output_dir


def parse_debug_flag() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", help="Debug", action="store_true")

    args = parser.parse_args()
    debug = args.debug

    return debug


def plan_with_one_stone(use_lp: bool, output_dir: Path, debug: bool = False) -> None:
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
    results = planner.get_results()
    if use_lp:
        name = "lp"
    else:
        name = "sdp"

    planner.save_analysis(str(output_dir / name))


def main(output_dir: Path, debug: bool = False) -> None:
    plan_with_one_stone(True, output_dir, debug)
    plan_with_one_stone(False, output_dir, debug)


if __name__ == "__main__":
    debug = parse_debug_flag()
    output_dir = make_output_folder()
    main(output_dir, debug)
