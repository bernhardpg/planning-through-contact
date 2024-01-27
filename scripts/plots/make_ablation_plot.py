import fnmatch
import os

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.ablation_study import (
    visualize_ablation_as_histogram,
    visualize_ablation_optimality_percentages,
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
# tee_folder = "hw_demos_20240125152221_tee_socp"
tee_folder = "hw_demos_20240125085656_tee_full_rot_25"
tee_folder = "hw_demos_20240124150815_tee_lam_buff_04_full_rot"
# tee_folder = "hw_demos_20240126183148_tee"

data_files = find_files(main_folder + tee_folder, pattern="solve_data.pkl")

results = [SingleRunResult.load(filename) for filename in data_files]
ablation_study = AblationStudy(results)

# visualize_multiple_ablation_studies([ablation_study])
visualize_ablation_optimality_percentages(ablation_study)
# visualize_ablation_as_histogram(ablation_study)
