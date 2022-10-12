from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation


def plot_positions_and_forces(
    positions: Dict[str, npt.NDArray[np.float64]],
    normal_forces: Dict[str, npt.NDArray[np.float64]],
    friction_forces: Dict[str, npt.NDArray[np.float64]],
) -> None:
    fig, (ax_pos_x, ax_pos_y, ax_norm, ax_fric) = plt.subplots(4, 1)
    fig.set_size_inches(8, 7)
    fig.tight_layout()

    GRAPH_BUFFER = 1.2
    max_force = (
        max(
            max(np.concatenate([c for c in normal_forces.values()])),
            max(np.concatenate([c for c in friction_forces.values()])),
        )
        * GRAPH_BUFFER
    )
    min_force = -1
    max_pos = (
        np.max(np.concatenate([c for c in positions.values()]), axis=0) * GRAPH_BUFFER
    )
    min_pos = (
        np.min(np.concatenate([c for c in positions.values()]), axis=0) * GRAPH_BUFFER
    )

    for name, curve in positions.items():
        (line_x,) = ax_pos_x.plot(curve[:, 0])
        line_x.set_label(f"{name}_x")
    ax_pos_x.set_title("x-positions")
    ax_pos_x.set_ylim((min_pos[0], max_pos[0]))
    ax_pos_x.legend()

    for name, curve in positions.items():
        (line_y,) = ax_pos_y.plot(curve[:, 1])
        line_y.set_label(f"{name}_y")
    ax_pos_y.set_title("y-positions")
    ax_pos_y.set_ylim((min_pos[1], max_pos[1]))
    ax_pos_y.legend()

    for name, curve in normal_forces.items():
        (line,) = ax_norm.plot(curve)
        line.set_label(f"{name}_normal")
    ax_norm.set_title("Normal forces")
    ax_norm.set_ylim((min_force, max_force))
    ax_norm.legend()

    for name, curve in friction_forces.items():
        (line,) = ax_fric.plot(curve)
        line.set_label(f"{name}_friction")
    ax_fric.set_title("Friction forces")
    ax_fric.set_ylim((min_force, max_force))
    ax_fric.legend()


def animate_positions(
    positions: Dict[str, npt.NDArray[np.float64]],
    box_width: float,
    box_height: float,
):
    fig = plt.figure()
    # TODO make scaleable
    ax = plt.axes(xlim=(-10, 20), ylim=(-10, 12))

    boxes = {}
    fingers = {}
    for body in positions.keys():
        if "box" in body:
            (box,) = ax.plot([], [], "r", lw=5)
            boxes[body] = box
        elif "finger" in body:
            (finger,) = ax.plot([], [], "bo", lw=10)
            fingers[body] = finger
        elif "ground" in body:
            (ground,) = ax.plot([], [])

    # TODO clean up
    n_frames = positions[list(positions.keys())[0]].shape[0]

    box_shape_x = np.array(
        [
            -box_width,
            +box_width,
            +box_width,
            -box_width,
            -box_width,
        ]
    )
    box_shape_y = np.array(
        [
            -box_height,
            -box_height,
            +box_height,
            +box_height,
            -box_height,
        ]
    )

    def init():
        breakpoint()
        for box in boxes.values():
            box.set_data([], [])
        for finger in fingers.values():
            finger.set_data([], [])
        ground.set_data([], [])
        return (*(boxes.values()), *(fingers.values()), ground)

    def animate(i):
        for body in positions.keys():
            if "box" in body:
                boxes[body].set_data(
                    positions[body][i, 0] + box_shape_x,
                    positions[body][i, 1] + box_shape_y,
                )
            elif "finger" in body:
                fingers[body].set_data(positions[body][i, 0], positions[body][i, 1])
            elif "ground" in body:
                ground.set_data(
                    [-999, 999], [positions[body][i, 0], positions[body][i, 1]]
                )

        return (*(boxes.values()), *(fingers.values()), ground)

    animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()


def animate_2d_box(
    finger_pos_x,
    finger_pos_y,
    box_pos_x,
    box_pos_y,
    ground_pos_x,
    ground_pos_y,
    finger_box_normal_force,
    finger_box_friction_force,
    ground_box_normal_force,
    ground_box_friction_force,
    box_width,
    box_height,
):
    n_frames = finger_pos_x.shape[0]
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, 20), ylim=(-10, 12))
    (box,) = ax.plot([], [], "r", lw=5)
    (finger,) = ax.plot([], [], "bo", lw=10)
    (finger_box_normal_force_plot,) = ax.plot([], [], "g>-", lw=2)
    (finger_box_friction_force_plot,) = ax.plot([], [], "g^-", lw=2)
    (ground_box_normal_force_plot,) = ax.plot([], [], "c^-", lw=2)
    (ground_box_friction_force_plot,) = ax.plot([], [], "c<-", lw=2)
    (ground,) = ax.plot([-100, 100], [0, 0])

    # initialization function: plot the background of each frame
    def init():
        finger.set_data([], [])
        box.set_data([], [])
        finger_box_normal_force_plot.set_data([], [])
        finger_box_friction_force_plot.set_data([], [])
        ground_box_normal_force_plot.set_data([], [])
        ground_box_friction_force_plot.set_data([], [])
        return (
            finger,
            box,
            finger_box_normal_force_plot,
            finger_box_friction_force_plot,
            ground_box_normal_force_plot,
            ground_box_friction_force_plot,
        )

    # animation function.  This is called sequentially
    def animate(i):
        finger.set_data(finger_pos_x[i], finger_pos_y[i])
        box_shape_x = np.array(
            [
                box_pos_x[i] - box_width,
                box_pos_x[i] + box_width,
                box_pos_x[i] + box_width,
                box_pos_x[i] - box_width,
                box_pos_x[i] - box_width,
            ]
        )
        box_shape_y = np.array(
            [
                box_pos_y[i] - box_height,
                box_pos_y[i] - box_height,
                box_pos_y[i] + box_height,
                box_pos_y[i] + box_height,
                box_pos_y[i] - box_height,
            ]
        )
        box.set_data(box_shape_x, box_shape_y)

        finger_box_normal_force_plot.set_data(
            [
                box_pos_x[i] - box_width,
                box_pos_x[i] - box_width + finger_box_normal_force[i],
            ],
            finger_pos_y[i],
        )
        ground_box_normal_force_plot.set_data(
            box_pos_x[i],
            [
                box_pos_y[i] - box_height,
                box_pos_y[i] - box_height + ground_box_normal_force[i],
            ],
        )
        finger_box_friction_force_plot.set_data(
            box_pos_x[i] - box_width,
            [finger_pos_y[i], finger_pos_y[i] + finger_box_friction_force[i]],
        )
        ground_box_friction_force_plot.set_data(
            [box_pos_x[i], box_pos_x[i] + ground_box_friction_force[i]],
            box_pos_y[i] - box_height,
        )

        return (
            finger,
            box,
            finger_box_normal_force_plot,
            finger_box_friction_force_plot,
            ground_box_normal_force_plot,
            ground_box_friction_force_plot,
        )

    # call the animator.  blit=True means only re-draw the parts that have changed.
    animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()
