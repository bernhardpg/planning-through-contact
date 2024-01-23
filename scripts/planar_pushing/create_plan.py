import argparse
from typing import List, Literal, Optional, Tuple

import numpy as np

from planning_through_contact.experiments.utils import (
    get_box,
    get_default_contact_cost,
    get_default_non_collision_cost,
    get_default_plan_config,
    get_default_solver_params,
    get_sugar_box,
    get_tee,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCost,
    ContactCostType,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
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


def get_plans_to_point(
    num_plans: int,
    lims: Tuple[float, float, float, float],
    pusher_radius: float,
    point: Tuple[float, float] = (0, 0),  # Default is origin
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(1)

    x_min, x_max, y_min, y_max = lims
    EPS = 0.01
    pusher_pose = PlanarPose(
        x_max + pusher_radius + EPS, y_max + pusher_radius + EPS, 0
    )

    plans = []
    for _ in range(num_plans):
        x_initial = np.random.uniform(x_min, x_max)
        y_initial = np.random.uniform(y_min, y_max)
        if limit_rotations:
            th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_initial_pose = PlanarPose(x_initial, y_initial, th_initial)
        slider_target_pose = PlanarPose(point[0], point[1], 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans


def create_plan(
    plan_spec: PlanarPushingStartAndGoal,
    slider_type: Literal["box", "tee", "sugar_box"] = "sugar_box",
    traj_name: str = "Untitled_traj",
    visualize: bool = False,
    pusher_radius: float = 0.035,
    time_in_contact: float = 7,
    do_rounding: bool = True,
    time_in_non_collision: float = 7,
    animation_output_dir: str = "",
    interpolate_video: bool = False,
    animation_lims: Optional[Tuple[float, float, float, float]] = None,
    save_traj: bool = False,
    save_analysis: bool = False,
    debug: bool = False,
    use_old_params: bool = False,
):
    if animation_output_dir != "":
        traj_name = animation_output_dir + "/" + traj_name

    if use_old_params:
        if slider_type == "box":
            slider = get_box()
        elif slider_type == "tee":
            slider = get_tee()
        elif slider_type == "sugar_box":
            slider = get_sugar_box()

        dynamics_config = SliderPusherSystemConfig(
            pusher_radius=pusher_radius,
            slider=slider,
            friction_coeff_slider_pusher=0.25,
            friction_coeff_table_slider=0.5,
            integration_constant=0.02,
        )

        # Configure contact cost
        contact_cost = ContactCost(
            cost_type=ContactCostType.OPTIMAL_CONTROL,
            force_regularization=5.0,
            ang_displacements=1.0,
            lin_displacements=1.0,
            mode_transition_cost=None,
        )

        contact_config = ContactConfig(
            cost=contact_cost,
            lam_min=0.47,
            lam_max=0.53,
            delta_vel_max=0.05 * 2,
            delta_theta_max=0.4 * 2,
        )

        # Configure non-collision cost
        non_collision_cost = NonCollisionCost(
            distance_to_object_quadratic=None,
            distance_to_object_socp=1.0,
            pusher_velocity_regularization=1.0,
            pusher_arc_length=None,
        )

        config = PlanarPlanConfig(
            dynamics_config=dynamics_config,
            time_in_contact=time_in_contact,
            time_non_collision=time_in_non_collision,
            num_knot_points_contact=4,
            num_knot_points_non_collision=4,
            allow_teleportation=False,
            use_band_sparsity=True,
            use_entry_and_exit_subgraphs=True,
            contact_config=contact_config,
            continuity_on_pusher_velocity=True,
            non_collision_cost=non_collision_cost,
        )

        solver_params = PlanarSolverParams(
            measure_solve_time=True,
            gcs_max_rounded_paths=20,
            print_flows=False,
            print_solver_output=debug,
            save_solver_output=False,
            print_path=debug,
            print_cost=debug,
            assert_result=True,
        )

    else:
        config = get_default_plan_config(
            slider_type=slider_type,
            pusher_radius=pusher_radius,
            integration_constant=0.3,
            friction_coeff=0.05,
            lam_buffer=0.4,
        )
        solver_params = get_default_solver_params(debug, clarabel=False)

    if debug:
        visualize_planar_pushing_start_and_goal(
            config.dynamics_config.slider.geometry,
            pusher_radius,
            plan_spec,
            # show=True,
            save=True,
            filename=f"{traj_name}_start_and_goal_{slider_type}",
        )

    planner = PlanarPushingPlanner(config)

    planner.config.start_and_goal = plan_spec
    planner.formulate_problem()

    if debug:
        planner.create_graph_diagram("graph")

    plan = planner.plan_path(solver_params)
    traj_relaxed = plan.to_traj()

    if do_rounding:
        traj_rounded = plan.to_traj(do_rounding=True, solver_params=solver_params)
    else:
        traj_rounded = None

    if save_traj:
        filename = f"trajectories/{slider_type}_pushing_{traj_name}.pkl"
        traj_relaxed.save(filename)  # type: ignore

        if traj_rounded is not None:
            filename = f"trajectories/{slider_type}_pushing_{traj_name}_rounded.pkl"
            traj_rounded.save(filename)  # type: ignore

    if save_analysis:
        analyze_plan(planner.path, filename=f"{traj_name}_{slider_type}")

        if traj_rounded is not None:
            analyze_plan(
                planner.path,
                filename=f"{traj_name}_{slider_type}_rounded",
                rounded=True,
            )

    make_traj_figure(
        traj_relaxed,
        filename=f"{traj_name}_{slider_type}",
    )
    plot_forces(
        traj_relaxed,
        filename=f"{traj_name}_{slider_type}_forces",
    )
    if traj_rounded is not None:
        make_traj_figure(
            traj_rounded,
            filename=f"{traj_name}_{slider_type}_rounded",
        )
    plot_forces(
        traj_relaxed,
        filename=f"{traj_name}_{slider_type}_forces_rounded",
    )

    if visualize:
        if debug:
            visualize_planar_pushing_start_and_goal(
                config.slider_geometry,
                config.pusher_radius,
                planner.config.start_and_goal,
                save=True,
                # show=True,
                filename=f"{traj_name}_start_and_goal_{slider_type}",
            )
        ani = visualize_planar_pushing_trajectory(
            traj_relaxed,  # type: ignore
            save=True,
            # show=True,
            filename=f"{traj_name}_{slider_type}",
            visualize_knot_points=not interpolate_video,
            lims=animation_lims,
        )

        if traj_rounded is not None:
            ani = visualize_planar_pushing_trajectory(
                traj_rounded,  # type: ignore
                save=True,
                # show=True,
                filename=f"{traj_name}_{slider_type}_rounded",
                visualize_knot_points=not interpolate_video,
                lims=animation_lims,
            )

            compare_trajs(
                traj_relaxed,
                traj_rounded,
                traj_a_legend="relaxed",
                traj_b_legend="rounded",
                filename=f"{traj_name}_{slider_type}_comparison",
            )

        return ani


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj",
        help="Which trajectory to plan",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--body",
        help="Which body to plan for",
        type=str,
        default="box",
    )
    parser.add_argument("--round", help="Do nonlinear rounding", action="store_true")
    parser.add_argument("--demos", help="Generate demos", action="store_true")
    parser.add_argument("--hardware_demos", help="Generate demos", action="store_true")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument(
        "--interpolate", help="Interpolate trajectory in video", action="store_true"
    )
    args = parser.parse_args()
    traj_number = args.traj
    make_demos = args.demos
    hardware_demos = args.hardware_demos
    debug = args.debug
    rounding = args.round
    interpolate = args.interpolate

    pusher_radius = 0.015

    if make_demos:
        lims = (-0.4, 0.4, -0.4, 0.4)
        animation_lims = (np.array(lims) * 1.3).tolist()
        plans = get_plans_to_point(9, lims, pusher_radius)

        if traj_number is not None:
            create_plan(
                plans[traj_number],
                debug=debug,
                slider_type=args.body,
                traj_name=f"demo_{traj_number}",
                visualize=True,
                pusher_radius=pusher_radius,
                save_traj=True,
                animation_output_dir="demos",
                animation_lims=animation_lims,
                time_in_contact=2.0,
                time_in_non_collision=1.0,
                interpolate_video=interpolate,
                save_analysis=True,
                do_rounding=rounding,
            )
        else:
            for idx, plan in enumerate(plans):
                create_plan(
                    plan,
                    debug=debug,
                    slider_type=args.body,
                    traj_name=f"demo_{idx}",
                    visualize=True,
                    pusher_radius=pusher_radius,
                    save_traj=True,
                    animation_output_dir="demos",
                    animation_lims=animation_lims,
                    time_in_contact=2.0,
                    time_in_non_collision=1.0,
                    interpolate_video=interpolate,
                    save_analysis=True,
                    do_rounding=rounding,
                )
    elif hardware_demos:
        lims = (0.5, 0.6, -0.15, 0.15)
        animation_lims = (np.array(lims) * 1.3).tolist()
        plans = get_plans_to_point(10, lims, pusher_radius, (0.575, -0.04285714))
        if traj_number is not None:
            create_plan(
                plans[traj_number],
                debug=debug,
                slider_type=args.body,
                traj_name=f"hw_demo_C_{traj_number}",
                visualize=True,
                pusher_radius=pusher_radius,
                save_traj=True,
                animation_output_dir="demos",
                animation_lims=animation_lims,
                time_in_contact=6.0,
                time_in_non_collision=2.0,
                interpolate_video=interpolate,
                save_analysis=True,
                do_rounding=rounding,
            )
        else:
            for idx, plan in enumerate(plans):
                create_plan(
                    plan,
                    debug=debug,
                    slider_type=args.body,
                    traj_name=f"hw_demo_C_{idx}",
                    visualize=True,
                    pusher_radius=pusher_radius,
                    save_traj=True,
                    animation_output_dir="demos",
                    animation_lims=animation_lims,
                    time_in_contact=4.0,
                    time_in_non_collision=2.0,
                    interpolate_video=interpolate,
                    save_analysis=True,
                    do_rounding=rounding,
                )
    else:
        plan_spec = get_predefined_plan(traj_number)

        create_plan(
            plan_spec,
            debug=debug,
            pusher_radius=pusher_radius,
            slider_type=args.body,
            traj_name=str(traj_number),
            visualize=True,
            save_traj=True,
            save_analysis=True,
            do_rounding=rounding,
        )
