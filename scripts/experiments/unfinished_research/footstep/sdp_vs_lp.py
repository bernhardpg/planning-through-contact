import argparse
import logging
from pathlib import Path
from typing import Optional

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
    FootstepPlanResult,
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


def make_default_logger(
    output_dir: Optional[Path] = None, test_logger: bool = False
) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the default log level (could be DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create handlers for both console and file logging
    console_handler = logging.StreamHandler()
    if output_dir is not None:
        name = str(output_dir / "script.log")
    else:
        name = "script.log"
    file_handler = logging.FileHandler(name)

    # Set the log level for each handler
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages
    if test_logger:
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")

    return logger


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
    debug = parse_debug_flag()
    output_dir = make_output_folder()
    logger = make_default_logger(output_dir)
    main(output_dir, debug)
