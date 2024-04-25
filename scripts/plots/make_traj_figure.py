import argparse
import os
from pathlib import Path

from tqdm import tqdm

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
    SimplePlanarPushingTrajectory,
)
from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    plot_simple_traj,
)


def get_run_traj_path(traj_number, main_folder, run_folder):
    traj_path = (
        main_folder
        + "/"
        + run_folder
        + f"/run_{traj_number}/trajectory/traj_rounded.pkl"
    )
    return traj_path


def plot_traj(traj_path, output_dir, traj_name):
    try:
        traj = PlanarPushingTrajectory.load(traj_path)
    except:
        try:
            traj = SimplePlanarPushingTrajectory.load(traj_path)
        except:
            raise RuntimeError("Does not recognize trajectory type.")

    os.makedirs(output_dir, exist_ok=True)
    output_name = output_dir + f"/{traj_name}"

    if type(traj) == PlanarPushingTrajectory:
        make_traj_figure(
            traj,
            filename=output_name,
            split_on_mode_type=True,
            start_end_legend=True,
            plot_lims=None,
            plot_knot_points=False,
            plot_forces=False,
            num_contact_frames=4,
            num_non_collision_frames=8,
        )
    elif type(traj) == SimplePlanarPushingTrajectory:
        plot_simple_traj(
            traj,
            filename=output_name,
            start_end_legend=False,
            plot_lims=None,
            keyframe_times=[0.0, 0.5, 1.1, 3.5, 3.8, 7.0],
            times_for_keyframes=[5, 4, 13, 3, 15],
        )
    else:
        raise RuntimeError("Invalid trajectory type")


def main() -> None:
    """
    This is a simple script that plots a figure given a trajectory saved as ".pkl". It contains some hardcoded functions
    for generating some of the plots in the paper.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj_path",
        help="Path to trajectory to generate plot for.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run_folder",
        help="Folder for several trajectories",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--traj",
        help="Specify a specific trajectory from a run to generate a figure for.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Specifies an output dir. If none is provided, the plot is saved to the same location as the trajectory.",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    traj_path = args.traj_path
    run_folder = args.run_folder
    output_dir = args.output_dir
    traj = args.traj

    if traj_path is not None:
        trajs_to_make_figs_for = [traj_path]
    elif run_folder is not None:
        run_folder = Path(run_folder)
        if traj is not None:
            children_names = [c.name for c in run_folder.iterdir()]
            if not traj in children_names:
                raise RuntimeError(f"Could not find folder {traj} in {run_folder}")

            traj_folders = [run_folder / traj]
        else:
            traj_folders = list(run_folder.iterdir())

        trajs_to_make_figs_for = []
        for folder in traj_folders:
            all_traj_files = list(folder.glob("**/*traj*.pkl"))
            trajs_to_make_figs_for.extend(all_traj_files)

        trajs_to_make_figs_for = sorted(trajs_to_make_figs_for)
    else:
        raise RuntimeError("No path provided.")

    for traj_path in tqdm(trajs_to_make_figs_for):
        traj_path = Path(traj_path)
        output_dir = str(traj_path.parent)
        name = traj_path.name.split(".")[0]
        plot_traj(traj_path, output_dir, name)

    return


if __name__ == "__main__":
    main()
