from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    AblationStudy,
    SingleRunResult,
)
from planning_through_contact.visualize.colors import GRAY, GRAY1


def visualize_ablation_as_histogram(study: AblationStudy) -> None:
    color = "blue"

    # Creating a 1x2 subplot
    fig = plt.figure(figsize=(10, 5))

    print(f"Mean optimality gap: {np.mean(study.optimality_gaps)}%")
    print(f"Mean SDP optimality gap: {np.nanmean(study.sdp_optimality_gaps)}%")

    ax1 = fig.add_subplot(121)
    ax1.hist(study.optimality_gaps, bins=40, color="blue", edgecolor="black")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 30)
    ax1.set_ylabel("Problem instances [%]")
    ax1.set_xlabel("Optimality gap [%]")
    ax1.set_title("Rounding")

    ax2 = fig.add_subplot(122)
    ax2.hist(study.sdp_optimality_gaps, bins=40, color="blue", edgecolor="black")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 30)
    ax2.set_ylabel("Problem instances [%]")
    ax2.set_xlabel("Optimality gap [%]")
    ax2.set_title("SDP Relaxation")

    fig.tight_layout()

    plt.show()


def visualize_multiple_ablation_studies(
    studies: List[AblationStudy],
    colors: Optional[List] = None,
    legends: Optional[List[str]] = None,
    show_sdp_and_rounded: bool = False,
    filename: Optional[str] = None,
    ALPHA: float = 0.5,
) -> None:
    # Colors for each subplot
    if colors is None:
        colors = ["red", "blue", "green", "purple", "orange"]

    if show_sdp_and_rounded:
        # Creating a 1x2 subplot
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121)
        ax1.set_xlabel("Total trajectory rotation [rad]")
        ax1.set_ylabel("Optimality [%]")
        ax1.set_ylim((-15, 100))
        ax1.set_title("Rounded")
        ax1.set_xlim((0, np.pi))
        ax1.hlines(
            [0],
            xmin=0,
            xmax=np.pi,
            linestyles="--",
            color=GRAY.diffuse(),
        )

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel("Total trajectory rotation [rad]")
        ax2.set_ylabel("Optimality [%]")
        ax2.set_title("SDP")
        ax2.set_ylim((-15, 100))
        ax2.set_xlim((0, np.pi))
        ax2.hlines(
            [0],
            xmin=0,
            xmax=np.pi,
            linestyles="--",
            color=GRAY.diffuse(),
        )

        # Rounded
        for idx, study in enumerate(studies):
            theta_success = [
                th
                for th, is_success in zip(study.thetas, study.rounded_is_success)
                if is_success
            ]
            optimality_gaps_success = [
                gap
                for gap, is_success in zip(
                    study.optimality_gaps, study.rounded_is_success
                )
                if is_success
            ]
            color = colors[idx]

            ax1.scatter(
                np.abs(theta_success),
                optimality_gaps_success,
                alpha=ALPHA,
                c=color,
            )
            theta_not_success = [
                th
                for th, is_success in zip(study.thetas, study.rounded_is_success)
                if not is_success
            ]
            ax1.scatter(
                theta_not_success,
                -10 * np.ones(len(theta_not_success)),
                alpha=ALPHA,
                c=color,
                marker="x",
            )

        # SDP
        for idx, study in enumerate(studies):
            theta_success = [
                th
                for th, is_success in zip(study.thetas, study.sdp_is_success)
                if is_success
            ]
            optimality_gaps_success = [
                gap
                for gap, is_success in zip(study.optimality_gaps, study.sdp_is_success)
                if is_success
            ]
            color = colors[idx]

            ax2.scatter(
                np.abs(theta_success),
                optimality_gaps_success,
                alpha=ALPHA,
                c=color,
            )
            theta_not_success = [
                th
                for th, is_success in zip(study.thetas, study.sdp_is_success)
                if not is_success
            ]
            ax2.scatter(
                theta_not_success,
                -10 * np.ones(len(theta_not_success)),
                alpha=ALPHA,
                c=color,
                marker="x",
            )

    else:  # only show rounded
        fig = plt.figure(figsize=(10, 3))

        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("Total trajectory rotation [rad]")
        ax1.set_ylabel("Optimality [%]")
        ax1.set_ylim((-15, 100))
        ax1.set_xlim((0, np.pi))
        ax1.hlines(
            [0],
            xmin=0,
            xmax=np.pi,
            linestyles="--",
            color=GRAY.diffuse(),
        )

        for idx, study in enumerate(studies):
            theta_success = [
                th
                for th, is_success in zip(study.thetas, study.rounded_is_success)
                if is_success
            ]
            optimality_gaps_success = [
                gap
                for gap, is_success in zip(
                    study.optimality_gaps, study.rounded_is_success
                )
                if is_success
            ]
            color = colors[idx]

            ax1.scatter(
                np.abs(theta_success),
                optimality_gaps_success,
                alpha=0.7,
                c=color,
            )
            theta_not_success = [
                th
                for th, is_success in zip(study.thetas, study.rounded_is_success)
                if not is_success
            ]
            ax1.scatter(
                theta_not_success,
                -10 * np.ones(len(theta_not_success)),
                alpha=0.7,
                c=color,
                marker="x",
            )

    # Create a list of patches to use as legend handles
    if legends is not None:
        custom_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(legends, colors)
        ]
        # Creating the custom legend
        plt.legend(handles=custom_patches)

    fig.tight_layout()

    if filename:
        fig.savefig(filename + f"_ablation.pdf")  # type: ignore
    else:
        plt.show()


def visualize_ablation_optimality_percentages(study: AblationStudy) -> None:
    # Creating a 1x2 subplot
    fig = plt.figure(figsize=(10, 5))

    det_min = min(
        min(study.rounded_mean_determinants), min(study.relaxed_mean_determinants)
    )

    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(
        study.thetas,
        study.optimality_percentages,
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
        study.sdp_optimality_percentages,
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
