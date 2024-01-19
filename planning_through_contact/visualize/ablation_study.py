import matplotlib.pyplot as plt

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.colors import GRAY, GRAY1


def visualize_ablation_optimality_gaps(study: AblationStudy) -> None:
    # Creating a 2x2 subplot
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(221)
    ax.scatter(study.thetas, study.optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Optimality gap [%]")
    ax.hlines(
        [100],
        xmin=min(study.thetas),
        xmax=max(study.thetas),
        linestyles="--",
        color=GRAY.diffuse(),
    )
    ax.set_title("Rounding")

    ax = fig.add_subplot(222)
    ax.scatter(study.thetas, study.sdp_optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Optimality gap [%]")
    ax.hlines(
        [100],
        xmin=min(study.thetas),
        xmax=max(study.thetas),
        linestyles="--",
        color=GRAY.diffuse(),
    )
    ax.set_title("SDP relaxation")

    ax = fig.add_subplot(223, projection="3d")
    ax.scatter(study.thetas, study.distances, study.optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Distance [m]")
    ax.set_zlabel("Optimality gap [%]")  # type: ignore

    ax = fig.add_subplot(224, projection="3d")
    ax.scatter(study.thetas, study.distances, study.sdp_optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Distance [m]")
    ax.set_zlabel("Optimality gap [%]")  # type: ignore

    fig.tight_layout()

    plt.show()


def visualize_ablation_optimality_gap_thetas(study: AblationStudy) -> None:
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


def visualize_ablation_sdp_optimality_gap_thetas(study: AblationStudy) -> None:
    plt.scatter(study.thetas, study.sdp_optimality_gaps, alpha=0.7)
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


def visualize_ablation_optimality_gap_3d(study: AblationStudy) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(study.thetas, study.distances, study.optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Distance [m]")
    ax.set_zlabel("Optimality gap [%]")  # type: ignore
    plt.show()


def visualize_ablation_sdp_optimality_gap_3d(study: AblationStudy) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(study.thetas, study.distances, study.sdp_optimality_gaps, alpha=0.7)
    ax.set_xlabel("Rotation [rad]")
    ax.set_ylabel("Distance [m]")
    ax.set_zlabel("Optimality gap [%]")  # type: ignore
    plt.show()
