import argparse

from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
    get_hardware_plans,
)
from planning_through_contact.planning.planar.utils import create_plan

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
        help="Also save the relaxed trajectory (which may not be feasible).",
        action="store_true",
    )
    parser.add_argument(
        "--hardware_demos",
        help="Generate demos for hardware experiments. This flag will automatically save all outputs to the folder 'hardware_demos/'",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode. Will print additional information, including solver output.",
        action="store_true",
    )
    parser.add_argument(
        "--interpolate",
        help="Interpolate trajectory in video (does not impact the plans themselves).",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        help="High-level output directory for plans. If the folder doesn't exist, it will be created. A timestamped \
        subfolder is created within 'output_dir'.",
        type=str,
        default="trajectories",
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

    pusher_radius = 0.015

    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=pusher_radius,
        hardware=hardware_demos,
    )
    solver_params = get_default_solver_params(debug, clarabel=False)

    if hardware_demos:
        output_dir = "hardware_demos"
        folder_name = create_output_folder(output_dir, slider_type, traj_number)

        plans = get_hardware_plans(seed, config)
    else:
        folder_name = create_output_folder(output_dir, slider_type, traj_number)
        plans = get_default_experiment_plans(seed, num_trajs, config)

    if traj_number is not None:
        plans_to_plan_for = [plans[traj_number]]
    else:
        plans_to_plan_for = plans

    for idx, plan in enumerate(plans_to_plan_for):
        create_plan(
            plan,
            config,
            solver_params,
            output_folder=folder_name,
            debug=debug,
            output_name=f"hw_demo_{idx}",
            save_video=True,
            save_traj=True,
            animation_lims=None,
            interpolate_video=interpolate,
            save_analysis=debug,
            do_rounding=rounding,
            hardware=True,
            save_relaxed=save_relaxed,
        )
