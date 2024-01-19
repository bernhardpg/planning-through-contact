from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    run_ablation_with_default_config,
)
from planning_through_contact.visualize.ablation_study import visualize_ablation_study

filename = "results/ablation_results.pkl"
num_runs = 20

run_ablation_with_default_config(num_runs, filename)

study = AblationStudy.load(filename)
visualize_ablation_study(study)
