import numpy as np

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
study_folder_3 = "hw_demos_20240125001442_tee_wrong_distance"
# tee_folder = "hw_demos_20240124130732_tee_lam_buff_04"
# tee_folder = "hw_demos_20240124150815_tee_lam_buff_04_full_rot"
study_folder_1 = "hw_demos_20240125085656_tee_full_rot_25"
# tee_folder = "hw_demos_20240126183148_tee"
study_folder_4 = "hw_demos_20240125152221_tee_socp"
study_folder_2 = "hw_demos_20240126183148_tee_socp_higher"
study_folder_5 = "hw_demos_20240127182809_box"
study_folder_6 = "hw_demos_20240127224039_tee_socp_too_big"
study_folder_7 = "hw_demos_20240128101451_box_quadratic"
study_folder_8 = "hw_demos_20240128165501_sugar_box"


study_folder_9 = "hw_demos_20240128184254_sugar_box"
study_folder_10 = "hw_demos_20240128204456_sugar_box"
study_folder_11 = "hw_demos_20240128223337_tee"
study_folder_12 = "hw_demos_20240129095609_box"

# folders = [study_folder_7, study_folder_1]
folders = [study_folder_12]
study_folders = [main_folder + folder for folder in folders]
studies = [AblationStudy.load_from_folder(folder) for folder in study_folders]


for study in studies:
    print(f"Num trajs: {len(study)}")
    print(f"Success rate: {study.percentage_success}%")
    print(f"Success rate rounding: {study.percentage_rounded_success}%")
    print(
        f"Mean optimality gap: {np.mean([gap for gap in study.optimality_gaps if not np.isinf(gap)])}"
    )
    print(
        f"Std optimality gap: {np.std([gap for gap in study.optimality_gaps if not np.isinf(gap)])}"
    )
    print("#####")
    print(f"Mean solve time SDP: {np.mean(study.solve_times_sdp)}")
    print(f"Std solve time SDP: {np.std(study.solve_times_sdp)}")
    print(f"Median solve time SDP: {np.median(study.solve_times_sdp)}")
    print(f"Max solve time SDP: {np.max(study.solve_times_sdp)}")
    print(f"Min solve time SDP: {np.min(study.solve_times_sdp)}")
    print("#####")
    print(f"Mean solve time rounding: {np.mean(study.solve_times_rounding)}")
    print(f"Std solve time rounding: {np.std(study.solve_times_rounding)}")
    print(f"Median solve time rounding: {np.median(study.solve_times_rounding)}")
    print(f"Max solve time rounding: {np.max(study.solve_times_rounding)}")
    print(f"Min solve time rounding: {np.min(study.solve_times_rounding)}")

colors = [
    "blue",
    "green",
    # BROWN2.diffuse(),
    # AQUAMARINE4.diffuse(),
    # DODGERBLUE2.diffuse(),
]

visualize_multiple_ablation_studies(
    studies,
    colors=colors,
    # legends=["Box", "Triangle", "Tee"],
    legends=["Box", "Tee"],
    filename="box_triangle_tee",
    show_sdp_and_rounded=True,
)
# visualize_ablation_optimality_percentages(ablation_study)
# visualize_ablation_as_histogram(ablation_study)
