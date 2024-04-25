import argparse

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    run_ablation_with_default_config,
)
from planning_through_contact.visualize.ablation_study import (
    visualize_ablation_optimality_gaps,
)

parser = argparse.ArgumentParser()
parser.add_argument("--slider", help="Slider type", type=str, default="sugar_box")
parser.add_argument("--const", help="Integration constant", type=float, default=0.5)
parser.add_argument("--num", help="Number of runs in study", type=int, default=10)
parser.add_argument("--vis", help="Visualize a study", action="store_true")
parser.add_argument("--arc", help="Arc length weight", type=float, default=None)
parser.add_argument("--run", help="Run a study", action="store_true")
parser.add_argument(
    "--sweep",
    help="Run a sweep over multiple integration constants",
    action="store_true",
)

args = parser.parse_args()
slider = args.slider
integration_constant = args.const
num_runs = args.num
visualize = args.vis
run_study = args.run
run_int_const_sweep = args.sweep
arc_length_weight = args.arc

pusher_radius = 0.035


def _float_to_str(fl) -> str:
    if fl is None:
        return str(fl)
    else:
        parts = str(fl).split(".")
        return parts[1].zfill(2)


if run_int_const_sweep:
    integration_constants = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
else:
    integration_constants = [integration_constant]

for integration_constant in integration_constants:
    filename = f"results/ablation_results_{slider}_{num_runs}_{_float_to_str(integration_constant)}_{_float_to_str(arc_length_weight)}.pkl"

    if run_study:
        run_ablation_with_default_config(
            slider,
            pusher_radius,
            integration_constant,
            num_runs,
            arc_length_weight=arc_length_weight,
            filename=filename,
        )

    if visualize:
        study = AblationStudy.load(filename)
        visualize_ablation_optimality_gaps(study)
