import matplotlib.pyplot as plt

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.colors import GRAY, GRAY1


def visualize_ablation_study(study: AblationStudy) -> None:
    plt.scatter(study.thetas, study.optimality_gaps, alpha=0.7)
    plt.xlabel("Rotation [rad]")
    plt.ylabel("Optimality gap [%]")
    plt.hlines(
        [100],
        xmin=min(study.thetas),
        xmax=max(study.thetas),
        linestyles="--",
        color=GRAY.diffuse(),
    )
    plt.show()
