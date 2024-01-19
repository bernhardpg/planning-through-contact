from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    run_ablation_with_default_config,
)
from planning_through_contact.visualize.ablation_study import (
    visualize_ablation_optimality_gap_3d,
    visualize_ablation_optimality_gap_thetas,
    visualize_ablation_optimality_gaps,
    visualize_ablation_sdp_optimality_gap_3d,
    visualize_ablation_sdp_optimality_gap_thetas,
)

slider = "box"
num_runs = 100
filename = f"results/ablation_results_{slider}_{num_runs}.pkl"

run_ablation_with_default_config(slider, num_runs, filename)

# study = AblationStudy.load(filename)
# visualize_ablation_optimality_gaps(study)
