import fnmatch
import os

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.ablation_study import (
    visualize_ablation_optimality_gaps,
    visualize_multiple_ablation_studies,
)


def find_files(directory, pattern):
    matches = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
    return matches


main_folder = "demos/"
tee_folder = "hw_demos_20240125152221_tee_socp"

data_files = find_files(main_folder + tee_folder, pattern="solve_data.pkl")

results = [SingleRunResult.load(filename) for filename in data_files]
ablation_study = AblationStudy(results)

visualize_multiple_ablation_studies([ablation_study])
