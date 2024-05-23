from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deprecated.geometry.rigid_body import RigidBody
from matplotlib import animation

# WARNING: This will be deprecated soon in favor of Visualizer2d


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
    min_force = (
        min(
            min(np.concatenate([c for c in normal_forces.values()])),
            min(np.concatenate([c for c in friction_forces.values()])),
        )
        * GRAPH_BUFFER
    )
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


def create_box_shape(width: float, height: float) -> npt.NDArray[np.float64]:
    shape_x = np.array(
        [
            -width,
            +width,
            +width,
            -width,
            -width,
        ]
    )
    shape_y = np.array(
        [
            -height,
            -height,
            +height,
            +height,
            -height,
        ]
    )

    return np.vstack([shape_x, shape_y])


def animate_positions(
    positions: Dict[str, npt.NDArray[np.float64]],
    bodies: List[RigidBody],
):
    fig = plt.figure()
    # TODO make scaleable
    ax = plt.axes(xlim=(-10, 20), ylim=(-10, 12))

    boxes = {}
    points = {}
    for b in bodies:
        if b.geometry == "box":
            if b.name in "ground":
                (ground,) = ax.plot([], [], "k")
                boxes[b.name] = ground
                continue
            (box,) = ax.plot([], [], "r", lw=5)
            boxes[b.name] = box
        elif b.geometry == "point":
            (finger,) = ax.plot([], [], "bo", lw=10)
            points[b.name] = finger

    # TODO clean up
    n_frames = positions[list(positions.keys())[0]].shape[0]
    box_shapes = {
        b.name: create_box_shape(b.width, b.height)
        for b in bodies
        if b.geometry == "box"
    }

    def init():
        for box in boxes.values():
            box.set_data([], [])
        for finger in points.values():
            finger.set_data([], [])
        return (*(boxes.values()), *(points.values()))

    def animate(i):
        for b in bodies:
            if b.geometry == "box":
                boxes[b.name].set_data(
                    positions[b.name][i, 0] + box_shapes[b.name][0, :],
                    positions[b.name][i, 1] + box_shapes[b.name][1, :],
                )
            if b.geometry == "point":
                points[b.name].set_data(
                    positions[b.name][i, 0], positions[b.name][i, 1]
                )

        return (*(boxes.values()), *(points.values()))

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()
