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

main_folder = "trajectories/"
# study_tee = "run_20240513144942_sugar_box"
# study_tee = "run_20240522174456_tee"
# study_tee = "run_20240522222534_sugar_box"
study_tee = "run_20240523080453_sugar_box"
# study_sugar_box = "run_20240202064957_sugar_box_FINAL"
# study_sugar_box = "run_20240202080034_sugar_box"
# study_tee = "run_20240202094838_tee"

# study_names = [study_sugar_box]
study_names = [study_tee]

num_trajs = 100
study_folders = [main_folder + folder for folder in study_names]
studies = [AblationStudy.load_from_folder(folder, 100) for folder in study_folders]


for folder, study in zip(study_folders, studies):
    with open(folder + "/data.txt", "w") as f:

        print(f"Num trajs: {len(study)}", file=f)
        print(
            f"Binary flows success rate: {study.percentage_binary_flows_success:.2f}%",
            file=f,
        )
        print(
            f"Feasible success rate: {study.percentage_feasible_success:.2f}%", file=f
        )
        print(
            f"Mean optimality gap: {np.mean([gap for gap in study.optimality_gaps if gap is not None]):.2f} %",
            file=f,
        )
        print(
            f"Median optimality gap: {np.median([gap for gap in study.optimality_gaps if gap is not None]):.2f} %",
            file=f,
        )
        print(
            f"Std optimality gap: {np.std([gap for gap in study.optimality_gaps if gap is not None]):.2f} %",
            file=f,
        )
        print("#####", file=f)
        print(
            f"Mean solve time GCS relaxation: {np.mean(study.solve_times_gcs_relaxed)}",
            file=f,
        )
        print(
            f"Std solve time GCS relaxation: {np.std(study.solve_times_gcs_relaxed)}",
            file=f,
        )
        print(
            f"Median solve time GCS relaxation: {np.median(study.solve_times_gcs_relaxed)}",
            file=f,
        )
        print(
            f"Max solve time GCS relaxation: {np.max(study.solve_times_gcs_relaxed)}",
            file=f,
        )
        print(
            f"Min solve time GCS relaxation: {np.min(study.solve_times_gcs_relaxed)}",
            file=f,
        )
        print("#####", file=f)
        print(
            f"Mean solve time binary flows: {np.mean(study.solve_times_binary_flows)}",
            file=f,
        )
        print(
            f"Std solve time binary flows: {np.std(study.solve_times_binary_flows)}",
            file=f,
        )
        print(
            f"Median solve time binary flows: {np.median(study.solve_times_binary_flows)}",
            file=f,
        )
        print(
            f"Max solve time binary flows: {np.max(study.solve_times_binary_flows)}",
            file=f,
        )
        print(
            f"Min solve time binary flows: {np.min(study.solve_times_binary_flows)}",
            file=f,
        )
        print("#####", file=f)
        # print(f"Mean rounding time: {np.mean(study.total_rounding_times)}")
        # print(f"Std rounding time: {np.std(study.total_rounding_times)}")
        # print(f"Median rounding time: {np.median(study.total_rounding_times)}")
        print(f"Infeasible runs::", file=f)
        for name in study.get_infeasible_idxs():
            print(name, file=f)

        print(f"Numerical difficulties runs::", file=f)
        for name in study.get_numerical_difficulties_idxs():
            print(name, file=f)

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
