import os

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar_pushing import make_traj_figure


def get_run_traj_path(traj_number, main_folder, run_folder):
    traj_path = (
        main_folder
        + "/"
        + run_folder
        + f"/run_{traj_number}/trajectory/traj_rounded.pkl"
    )
    return traj_path


def plot_traj(traj_path, output_dir, traj_name):
    traj = PlanarPushingTrajectory.load(traj_path)

    os.makedirs(output_dir, exist_ok=True)
    output_name = output_dir + f"/{traj_name}"

    make_traj_figure(
        traj,
        filename=output_name,
        split_on_mode_type=True,
        start_end_legend=False,
        plot_lims=None,
        plot_knot_points=False,
        plot_forces=False,
        num_contact_frames=4,
        num_non_collision_frames=8,
    )


main_folder = "trajectories"
# run_folder = "run_20240202094838_tee_FINAL_higher_acc"
run_folder = "run_20240202064957_sugar_box_FINAL"
output_dir = f"trajectory_figures/{run_folder}"
# plot_traj(0, main_folder, run_folder)

# for n in range(50):
#     plot_traj(get_run_traj_path(n, main_folder, run_folder), output_dir, f"run_{n}")

get_video_traj_name = lambda n: f"videos/{n}/hw_demo_{n}/trajectory/traj_rounded.pkl"
output_dir = f"trajectory_figures/videos/"
video_num = 14
video_traj_path = get_video_traj_name(video_num)
plot_traj(video_traj_path, output_dir, f"video_{video_num}")
