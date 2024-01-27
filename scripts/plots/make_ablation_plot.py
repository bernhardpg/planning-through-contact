from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.ablation_study import (
    visualize_ablation_as_histogram,
    visualize_ablation_optimality_percentages,
    visualize_multiple_ablation_studies,
)
from planning_through_contact.visualize.colors import AQUAMARINE4, BROWN2, DODGERBLUE2

main_folder = "demos/"
# tee_folder = "hw_demos_20240125001442_tee_wrong_distance"
# tee_folder = "hw_demos_20240124130732_tee_lam_buff_04"
# tee_folder = "hw_demos_20240124150815_tee_lam_buff_04_full_rot"
study_folder_1 = "hw_demos_20240125085656_tee_full_rot_25"
# tee_folder = "hw_demos_20240126183148_tee"
# tee_folder = "hw_demos_20240125152221_tee_socp"
study_folder_2 = "hw_demos_20240126183148_tee_socp_higher"

folders = [main_folder + study_folder_1, main_folder + study_folder_2]
studies = [AblationStudy.load_from_folder(folder) for folder in folders]


for study in studies:
    print(f"Num trajs: {len(study)}")
    print(f"Success rate: {study.percentage_success}%")
    print(f"Success rate rounding: {study.percentage_rounded_success}%")

colors = [
    BROWN2.diffuse(),
    AQUAMARINE4.diffuse(),
    DODGERBLUE2.diffuse(),
]


visualize_multiple_ablation_studies(studies, colors=colors)
# visualize_ablation_optimality_percentages(ablation_study)
# visualize_ablation_as_histogram(ablation_study)
