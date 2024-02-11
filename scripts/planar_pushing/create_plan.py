import argparse
import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    do_one_run_get_path,
)
from planning_through_contact.experiments.utils import (
    get_box,
    get_default_plan_config,
    get_default_solver_params,
    get_sugar_box,
    get_tee,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    ContactConfig,
    ContactCost,
    ContactCostType,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import (
    analyze_mode_result,
    analyze_plan,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar_pushing import (
    compare_trajs,
    make_traj_figure,
    plot_forces,
    visualize_planar_pushing_start_and_goal,
    visualize_planar_pushing_trajectory,
)


def get_predefined_plan(traj_number: int) -> PlanarPushingStartAndGoal:
    if traj_number == 1:
        slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=0.0)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 2:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.2, theta=0.5)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 3:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=-0.2)
        slider_target_pose = PlanarPose(x=0.70, y=0.4, theta=0.5)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
    elif traj_number == 4:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2 - 0.2)
        slider_target_pose = PlanarPose(x=0.75, y=-0.2, theta=np.pi / 2 + 0.4)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 5:
        slider_initial_pose = PlanarPose(x=0.60, y=0.1, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.70, y=-0.05, theta=np.pi / 2 + 0.4)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 6:
        slider_initial_pose = PlanarPose(x=0.60, y=0.0, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.65, y=-0.1, theta=np.pi / 2 + 0.3)
        pusher_initial_pose = PlanarPose(x=-0.2, y=0.0, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 7:
        slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=0.3, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 8:
        slider_initial_pose = PlanarPose(x=0.0, y=0.0, theta=0.0)
        slider_target_pose = PlanarPose(x=-0.3, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 9:
        slider_initial_pose = PlanarPose(x=0.5, y=0.0, theta=1.0)
        slider_target_pose = PlanarPose(x=0.6, y=-0.15, theta=-0.5)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 10:
        slider_initial_pose = PlanarPose(x=0.7, y=0.2, theta=0.3)
        slider_target_pose = PlanarPose(x=0.55, y=-0.15, theta=1.2)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 11:
        # Rotate in place is a bit loose
        slider_initial_pose = PlanarPose(x=0.65, y=0.0, theta=0)
        slider_target_pose = PlanarPose(x=0.65, y=0.0, theta=np.pi - 0.1)
        pusher_initial_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.0, y=0.2, theta=0.0)
    elif traj_number == 14:  # Rotate in place
        slider_initial_pose = PlanarPose(x=0, y=0, theta=1.0)
        slider_target_pose = PlanarPose(x=0, y=0, theta=0.0)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 15:  # Rotate in place
        slider_initial_pose = PlanarPose(x=0, y=0, theta=np.pi * 0.8)
        slider_target_pose = PlanarPose(x=0, y=0, theta=0.0)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 16:  # Rotate in place
        slider_initial_pose = PlanarPose(x=0, y=0, theta=-np.pi * 0.9)
        slider_target_pose = PlanarPose(x=0, y=0, theta=0.0)
        pusher_initial_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=-0.2, y=-0.2, theta=0.0)
    elif traj_number == 512:
        slider_initial_pose = PlanarPose(x=0.55, y=-0.1, theta=0.9)
        slider_target_pose = PlanarPose(x=0.55, y=0.1, theta=0.9)
        pusher_initial_pose = PlanarPose(x=0.45, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.45, y=-0.2, theta=0.0)
    elif traj_number == 513:
        slider_initial_pose = PlanarPose(x=0.6, y=-0.05, theta=0)
        slider_target_pose = PlanarPose(x=0.6, y=0.15, theta=0)
        pusher_initial_pose = PlanarPose(x=0.45, y=-0.2, theta=0.0)
        pusher_target_pose = PlanarPose(x=0.45, y=-0.2, theta=0.0)
    else:
        raise NotImplementedError()

    return PlanarPushingStartAndGoal(
        slider_initial_pose, slider_target_pose, pusher_initial_pose, pusher_target_pose
    )


def _slider_within_workspace(
    workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: CollisionGeometry
) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    p_Wv_s = [
        slider.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all(
        [v >= lb for v in p_Wv_s]
    )
    return vertices_within_workspace


# TODO: refactor
def _check_collision(
    pusher_pose_world: PlanarPose,
    slider_pose_world: PlanarPose,
    config: PlanarPlanConfig,
) -> bool:
    p_WP = pusher_pose_world.pos()
    R_WB = slider_pose_world.two_d_rot_matrix()
    p_WB = slider_pose_world.pos()

    # We need to compute the pusher pos in the frame of the slider
    p_BP = R_WB.T @ (p_WP - p_WB)
    pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(pusher_pose_body, loc, config)
    ]
    if len(matching_locs) == 0:
        return True
    else:
        return False


def _get_slider_pose_within_workspace(
    workspace: PlanarPushingWorkspace,
    slider: CollisionGeometry,
    pusher_pose: PlanarPose,
    config: PlanarPlanConfig,
    limit_rotations: bool = False,
    enforce_entire_slider_within_workspace: bool = True,
) -> PlanarPose:
    valid_pose = False

    slider_pose = None
    while not valid_pose:
        x_initial = np.random.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = np.random.uniform(workspace.slider.y_min, workspace.slider.y_max)
        EPS = 0.01
        if limit_rotations:
            th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = _check_collision(pusher_pose, slider_pose, config)
        within_workspace = _slider_within_workspace(workspace, slider_pose, slider)

        if enforce_entire_slider_within_workspace:
            valid_pose = within_workspace and not collides_with_pusher
        else:
            valid_pose = not collides_with_pusher

    assert slider_pose is not None  # fix LSP errors

    return slider_pose


def get_plan_start_and_goals_to_point(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    point: Tuple[float, float] = (0, 0),  # Default is origin
    init_pusher_pose: Optional[PlanarPose] = None,
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(seed)

    slider = config.slider_geometry

    # Hardcoded pusher start pose to be at top edge
    # of workspace
    ws = workspace.slider.new_workspace_with_buffer(new_buffer=0)
    if init_pusher_pose is not None:
        pusher_pose = init_pusher_pose
    else:
        pusher_pose = PlanarPose(ws.x_min, 0, 0)

    plans = []
    for _ in range(num_plans):
        slider_initial_pose = _get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )

        slider_target_pose = PlanarPose(point[0], point[1], 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans


def create_plan(
    plan_spec: PlanarPushingStartAndGoal,
    output_dir: str = "",
    traj_name: str = "Untitled_traj",
    save_video: bool = False,
    pusher_radius: float = 0.035,
    do_rounding: bool = True,
    interpolate_video: bool = False,
    animation_lims: Optional[Tuple[float, float, float, float]] = None,
    save_traj: bool = False,
    save_analysis: bool = False,
    debug: bool = False,
    hardware: bool = False,
    save_relaxed: bool = False,
):
    # Set up folders
    folder_name = f"{output_dir}/{traj_name}"
    os.makedirs(folder_name, exist_ok=True)
    trajectory_folder = f"{folder_name}/trajectory"
    os.makedirs(trajectory_folder, exist_ok=True)
    analysis_folder = f"{folder_name}/analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    if debug:
        visualize_planar_pushing_start_and_goal(
            config.dynamics_config.slider.geometry,
            pusher_radius,
            plan_spec,
            # show=True,
            save=True,
            filename=f"{folder_name}/start_and_goal",
        )

    if debug:
        solve_data, path = do_one_run_get_path(
            config, solver_params, start_and_goal=plan_spec, save_cost_vals=True
        )
    else:
        planner = PlanarPushingPlanner(config)
        planner.config.start_and_goal = plan_spec
        planner.formulate_problem()
        path = planner.plan_path(solver_params)
        solve_data = None

    if debug and solve_data is not None:
        solve_data.save(f"{analysis_folder}/solve_data.pkl")
        solve_data.save_as_text(f"{analysis_folder}/solve_data.txt")

    # We may get infeasible
    if path is not None:
        traj_relaxed = path.to_traj()

        if do_rounding:
            traj_rounded = path.to_traj(rounded=True)
        else:
            traj_rounded = None

        if save_traj:
            if save_relaxed:
                traj_relaxed.save(f"{trajectory_folder}/traj_relaxed.pkl")  # type: ignore

            if traj_rounded is not None:
                traj_rounded.save(f"{trajectory_folder}/traj_rounded.pkl")  # type: ignore

        if save_analysis:
            analyze_plan(path, filename=f"{analysis_folder}/relaxed")

            if traj_rounded is not None:
                analyze_plan(
                    path,
                    filename=f"{analysis_folder}/rounded",
                    rounded=True,
                )

        slider_color = COLORS["aquamarine4"].diffuse()

        if save_relaxed:
            make_traj_figure(
                traj_relaxed,
                filename=f"{analysis_folder}/relaxed_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=hardware,
            )

            if save_analysis:
                plot_forces(
                    traj_relaxed, filename=f"{analysis_folder}/relaxed_traj_forces"
                )

        if traj_rounded is not None:
            make_traj_figure(
                traj_rounded,
                filename=f"{analysis_folder}/rounded_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=hardware,
            )

            if save_analysis:
                plot_forces(
                    traj_rounded, filename=f"{analysis_folder}/rounded_traj_forces"
                )

                compare_trajs(
                    traj_relaxed,
                    traj_rounded,
                    traj_a_legend="relaxed",
                    traj_b_legend="rounded",
                    filename=f"{analysis_folder}/comparison",
                )

        if save_video:
            if save_relaxed:
                ani = visualize_planar_pushing_trajectory(
                    traj_relaxed,  # type: ignore
                    save=True,
                    # show=True,
                    filename=f"{analysis_folder}/relaxed_traj",
                    visualize_knot_points=not interpolate_video,
                    lims=animation_lims,
                )

            if traj_rounded is not None:
                ani = visualize_planar_pushing_trajectory(
                    traj_rounded,  # type: ignore
                    save=True,
                    # show=True,
                    filename=f"{analysis_folder}/rounded_traj",
                    visualize_knot_points=not interpolate_video,
                    lims=animation_lims,
                )
                return ani


def _get_time_as_str() -> str:
    current_time = datetime.now()
    # For example, YYYYMMDDHHMMSS format
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Random seed for generating trajectories",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--traj",
        help="(Optional) specify a specific trajectory number to generate, with the given random seed.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--body",
        help="Which slider body to use.",
        type=str,
        default="box",
    )
    parser.add_argument(
        "--num",
        help="Number of trajectories to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_relaxed",
        help="Also save the relaxed trajectory, which may not be feasible.",
        action="store_true",
    )
    parser.add_argument(
        "--hardware_demos",
        help="Generate demos for hardware experiments.",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode. Will output a number of additional data.",
        action="store_true",
    )
    parser.add_argument(
        "--interpolate",
        help="Interpolate trajectory in video (does not impact the plans themselves).",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir", help="Output directory.", type=str, default="trajectories"
    )
    parser.add_argument(
        "--data_collection",
        help="Run in data collectino mode. Overrides many other options.",
        action="store_true",
    )

    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    hardware_demos = args.hardware_demos
    debug = args.debug
    rounding = True
    interpolate = args.interpolate
    slider_type = args.body
    num_trajs = args.num
    output_dir = args.output_dir
    save_relaxed = args.save_relaxed
    data_collection = args.data_collection

    pusher_radius = 0.015

    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=pusher_radius,
        hardware=hardware_demos,
    )
    solver_params = get_default_solver_params(debug, clarabel=False)

    if data_collection:
        from tqdm import tqdm
        import logging

        output_dir = "data_collection_trajectories"
        if os.path.exists(output_dir):
            user_input = input(f"{output_dir} already exists. Overwrite? (y/n): ")
            if user_input.lower() != "y":
                print("Exiting")
                exit()
        
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        folder_name = f"{output_dir}"
        os.makedirs(folder_name, exist_ok=True)

        workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=0.35,
                height=0.5,
                center=np.array([0.5, 0.0]),
                buffer=0,
            ),
        )

        plans = get_plan_start_and_goals_to_point(
            seed,
            num_trajs,
            workspace,
            config,
            (0.5, 0.0),
            init_pusher_pose=PlanarPose(0.5, 0.25, 0.0),
            limit_rotations=False,
        )
        print("Finished finding random plans within workspace")

        logging.getLogger('drake').setLevel(logging.WARNING)
        pbar = tqdm(total=len(plans))
        for idx, plan in enumerate(plans):
            create_plan(
                plan,
                output_dir=folder_name,
                debug=False,
                traj_name=f"traj_{idx}",
                save_video=False,
                pusher_radius=pusher_radius,
                save_traj=True,
                animation_lims=None,
                interpolate_video=interpolate,
                save_analysis=False,
                do_rounding=rounding,
                hardware=False,
                save_relaxed=False,
            )
            pbar.update(1)
    elif hardware_demos:
        output_dir = "hardware_demos"
        os.makedirs(output_dir, exist_ok=True)
        folder_name = f"{output_dir}/hw_demos_{_get_time_as_str()}_{slider_type}"
        if traj_number is not None:
            folder_name += f"_traj_{traj_number}"

        os.makedirs(folder_name, exist_ok=True)

        workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=0.35,
                height=0.5,
                center=np.array([0.575, 0.0]),
                buffer=0,
            ),
        )

        num_trajs = 30
        plans = get_plan_start_and_goals_to_point(
            seed,
            num_trajs,
            workspace,
            config,
            (0.575, -0.04285714),
            limit_rotations=False,
        )
        print("Finished finding random plans within workspace")
        if traj_number is not None:
            create_plan(
                plans[traj_number],
                debug=debug,
                output_dir=folder_name,
                traj_name=f"hw_demo_{traj_number}",
                save_video=True,
                pusher_radius=pusher_radius,
                save_traj=True,
                animation_lims=None,
                interpolate_video=interpolate,
                save_analysis=debug,
                do_rounding=rounding,
                hardware=True,
                save_relaxed=save_relaxed,
            )
        else:
            for idx, plan in enumerate(plans):
                create_plan(
                    plan,
                    output_dir=folder_name,
                    debug=debug,
                    traj_name=f"hw_demo_{idx}",
                    save_video=True,
                    pusher_radius=pusher_radius,
                    save_traj=True,
                    animation_lims=None,
                    interpolate_video=interpolate,
                    save_analysis=debug,
                    do_rounding=rounding,
                    hardware=True,
                    save_relaxed=save_relaxed,
                )
    else:
        os.makedirs(output_dir, exist_ok=True)
        folder_name = f"{output_dir}/run_{_get_time_as_str()}_{slider_type}"
        if traj_number is not None:
            folder_name += f"_traj_{traj_number}"
        os.makedirs(folder_name, exist_ok=True)

        workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=0.6,
                height=0.6,
                center=np.array([0.0, 0.0]),
                buffer=0,
            ),
        )

        plans = get_plan_start_and_goals_to_point(
            seed,
            num_trajs,
            workspace,
            config,
            (0.0, 0.0),
            limit_rotations=False,
        )
        if traj_number is not None:
            create_plan(
                plans[traj_number],
                debug=debug,
                output_dir=folder_name,
                traj_name=f"run_{traj_number}",
                save_video=True,
                pusher_radius=pusher_radius,
                save_traj=True,
                animation_lims=None,
                interpolate_video=interpolate,
                save_analysis=debug,
                do_rounding=rounding,
                save_relaxed=save_relaxed,
            )
        else:
            for idx, plan in enumerate(plans):
                create_plan(
                    plan,
                    output_dir=folder_name,
                    debug=debug,
                    traj_name=f"run_{idx}",
                    save_video=True,
                    pusher_radius=pusher_radius,
                    save_traj=True,
                    animation_lims=None,
                    interpolate_video=interpolate,
                    save_analysis=debug,
                    do_rounding=rounding,
                    save_relaxed=save_relaxed,
                )