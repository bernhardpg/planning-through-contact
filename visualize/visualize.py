import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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
