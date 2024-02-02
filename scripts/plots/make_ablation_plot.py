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

# main_folder = "demos/"
# # tee_folder = "hw_demos_20240125001442_tee_wrong_distance"
# study_folder_3 = "hw_demos_20240125001442_tee_wrong_distance"
# # tee_folder = "hw_demos_20240124130732_tee_lam_buff_04"
# # tee_folder = "hw_demos_20240124150815_tee_lam_buff_04_full_rot"
# study_folder_1 = "hw_demos_20240125085656_tee_full_rot_25"
# # tee_folder = "hw_demos_20240126183148_tee"
# study_folder_4 = "hw_demos_20240125152221_tee_socp"
# study_folder_2 = "hw_demos_20240126183148_tee_socp_higher"
# study_folder_5 = "hw_demos_20240127182809_box"
# study_folder_6 = "hw_demos_20240127224039_tee_socp_too_big"
# study_folder_7 = "hw_demos_20240128101451_box_quadratic"
# study_folder_8 = "hw_demos_20240128165501_sugar_box"
#
#
# study_folder_9 = "hw_demos_20240128184254_sugar_box"
# study_folder_10 = "hw_demos_20240128204456_sugar_box"
# study_folder_11 = "hw_demos_20240128223337_tee"
# study_folder_12 = "hw_demos_20240129095609_box"
# study_folder_13 = "hw_demos_20240129102356_box"
# study_folder_14 = "hw_demos_20240129103445_box"
# study_folder_15 = "hw_demos_20240129104800_box"
# study_folder_16 = "hw_demos_20240129115732_tee"
# study_folder_17 = "hw_demos_20240129135032_sugar_box"
#
# folders = [study_folder_17, study_folder_16]
# folders = [study_folder_16]

main_folder = "trajectories/"
# study = "hw_demos_20240130115816_box"
# study = "run_20240131211332_box"
# study = "run_20240130160321_box"
# study = "run_20240130161023_box"
# study = "run_20240130173234_tee"
# study = "run_20240130174605_tee"
# study = "run_20240130222224_tee"
# study = "run_20240131091107_box"
# study = "run_20240131110644_tee"
# study = "run_20240131140048_tee"
# study = "run_20240131153837_sugar_box"
# study = "run_20240131164250_box"
# study = "run_20240201102145_box"
# study = "run_20240201153823_box"  # tuned magnitudes
# study = "run_20240201225301_tee"
# study = "run_20240201184004_tee_keypoint_reg"
# study = "run_20240201190626_tee"
# study = "run_20240201192206_box"
# study = "run_20240201194934_box"
# study = "run_20240201200842_tee"
study = "run_20240201215042_tee"  # last one
# main_folder = "demos/"
# study = "hw_demos_20240131170037_box"
# study = "hw_demos_20240131172730_box"
# study = "hw_demos_20240131194747_box"
# study = "hw_demos_20240131200249_box"
# study = "hw_demos_20240131212709_tee"  # This one is the best one so far
study_names = [study]

study_folders = [main_folder + folder for folder in study_names]
studies = [AblationStudy.load_from_folder(folder) for folder in study_folders]


for study in studies:
    print(f"Num trajs: {len(study)}")
    print(f"Binary flows success rate: {study.percentage_binary_flows_success}%")
    print(f"Feasible success rate: {study.percentage_feasible_success}%")
    print(
        f"Mean optimality gap: {np.mean([gap for gap in study.optimality_gaps if gap is not None])}"
    )
    print(
        f"Median optimality gap: {np.median([gap for gap in study.optimality_gaps if gap is not None])}"
    )
    print(
        f"Std optimality gap: {np.std([gap for gap in study.optimality_gaps if gap is not None])}"
    )
    print("#####")
    print(f"Mean solve time GCS relaxation: {np.mean(study.solve_times_gcs_relaxed)}")
    print(f"Std solve time GCS relaxation: {np.std(study.solve_times_gcs_relaxed)}")
    print(
        f"Median solve time GCS relaxation: {np.median(study.solve_times_gcs_relaxed)}"
    )
    print(f"Max solve time GCS relaxation: {np.max(study.solve_times_gcs_relaxed)}")
    print(f"Min solve time GCS relaxation: {np.min(study.solve_times_gcs_relaxed)}")
    print("#####")
    print(f"Mean solve time binary flows: {np.mean(study.solve_times_binary_flows)}")
    print(f"Std solve time binary flows: {np.std(study.solve_times_binary_flows)}")
    print(
        f"Median solve time binary flows: {np.median(study.solve_times_binary_flows)}"
    )
    print(f"Max solve time binary flows: {np.max(study.solve_times_binary_flows)}")
    print(f"Min solve time binary flows: {np.min(study.solve_times_binary_flows)}")
    print(f"Infeasible:")
    for name in study.get_infeasible_idxs():
        print(name)
    # print(f"Mean solve time feasible: {np.mean(study.solve_times_feasible)}")
    # print(f"Std solve time feasible: {np.std(study.solve_times_feasible)}")
    # print(f"Median solve time feasible: {np.median(study.solve_times_feasible)}")
    # print(f"Max solve time feasible: {np.max(study.solve_times_feasible)}")
    # print(f"Min solve time feasible: {np.min(study.solve_times_feasible)}")


colors = [
    "blue",
    "green",
    # BROWN2.diffuse(),
    # AQUAMARINE4.diffuse(),
    # DODGERBLUE2.diffuse(),
]

# save the ablation study in each folder
for study_name, study_folder in zip(study_names, study_folders):
    visualize_multiple_ablation_studies(
        studies,
        colors=colors,
        # legends=["Box", "Triangle", "Tee"],
        legends=["Box", "Tee"],
        filename=study_folder + "/0_" + study_name,
        show_sdp_and_rounded=False,
        next_to_each_other=False,
    )
# visualize_ablation_optimality_percentages(ablation_study)
# visualize_ablation_as_histogram(ablation_study)
