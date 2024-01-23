import matplotlib.pyplot as plt
import numpy as np

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.colors import GRAY, GRAY1


def visualize_ablation_optimality_gaps(study: AblationStudy) -> None:
    # Creating a 1x2 subplot
    fig = plt.figure(figsize=(10, 5))

    det_min = min(
        min(study.rounded_mean_determinants), min(study.relaxed_mean_determinants)
    )

    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(
        study.thetas,
        study.optimality_gaps,
        alpha=0.7,
        c=study.rounded_mean_determinants,
        vmin=det_min,
        vmax=1.0,
    )
    ax1.set_xlabel("Rotation [rad]")
    ax1.set_ylabel("Optimality gap [%]")
    ax1.set_ylim((0, 110))
    ax1.set_xlim((-np.pi, np.pi))
    ax1.hlines(
        [100],
        xmin=-np.pi,
        xmax=np.pi,
        linestyles="--",
        color=GRAY.diffuse(),
    )
    ax1.set_title("Rounding")

    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(
        study.thetas,
        study.sdp_optimality_gaps,
        alpha=0.7,
        c=study.relaxed_mean_determinants,
        vmin=det_min,
        vmax=1.0,
    )
    ax2.set_xlabel("Rotation [rad]")
    ax2.set_ylabel("Optimality gap [%]")
    ax2.set_ylim((0, 110))
    ax2.set_xlim((-np.pi, np.pi))
    ax2.hlines(
        [100],
        xmin=-np.pi,
        xmax=np.pi,
        linestyles="--",
        color=GRAY.diffuse(),
    )
    ax2.set_title("SDP relaxation")
    # Add a colorbar
    cbar = fig.colorbar(scatter2)
    cbar.set_label("Determinants")

    # ax = fig.add_subplot(223, projection="3d")
    # ax.scatter(study.thetas, study.distances, study.optimality_gaps, alpha=0.7)
    # ax.set_xlabel("Rotation [rad]")
    # ax.set_ylabel("Distance [m]")
    # ax.set_zlabel("Optimality gap [%]")  # type: ignore
    #
    # ax = fig.add_subplot(224, projection="3d")
    # ax.scatter(study.thetas, study.distances, study.sdp_optimality_gaps, alpha=0.7)
    # ax.set_xlabel("Rotation [rad]")
    # ax.set_ylabel("Distance [m]")
    # ax.set_zlabel("Optimality gap [%]")  # type: ignore
    #
    # ax = fig.add_subplot(225, projection="3d")
    # ax.scatter(
    #     study.thetas, study.relaxed_mean_determinants, study.optimality_gaps, alpha=0.7
    # )
    # ax.set_xlabel("Rotation [rad]")
    # ax.set_ylabel("Mean determinant")
    # ax.set_zlabel("Optimality gap [%]")  # type: ignore
    #
    # ax = fig.add_subplot(226, projection="3d")
    # ax.scatter(
    #     study.thetas,
    #     study.relaxed_mean_determinants,
    #     study.sdp_optimality_gaps,
    #     alpha=0.7,
    # )
    # ax.set_xlabel("Rotation [rad]")
    # ax.set_ylabel("Mean determinant")
    # ax.set_zlabel("Optimality gap [%]")  # type: ignore

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
