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
    box_ground_normal_force,
    box_ground_friction_force,
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
    (box_ground_normal_force_plot,) = ax.plot([], [], "c^-", lw=2)
    (box_ground_friction_force_plot,) = ax.plot([], [], "c<-", lw=2)
    (ground,) = ax.plot([-100, 100], [0, 0])

    # initialization function: plot the background of each frame
    def init():
        finger.set_data([], [])
        box.set_data([], [])
        finger_box_normal_force_plot.set_data([], [])
        finger_box_friction_force_plot.set_data([], [])
        box_ground_normal_force_plot.set_data([], [])
        box_ground_friction_force_plot.set_data([], [])
        return (
            finger,
            box,
            finger_box_normal_force_plot,
            finger_box_friction_force_plot,
            box_ground_normal_force_plot,
            box_ground_friction_force_plot,
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
        box_ground_normal_force_plot.set_data(
            box_pos_x[i],
            [
                box_pos_y[i] - box_height,
                box_pos_y[i] - box_height + box_ground_normal_force[i],
            ],
        )
        finger_box_friction_force_plot.set_data(
            box_pos_x[i] - box_width,
            [finger_pos_y[i], finger_pos_y[i] + finger_box_friction_force[i]],
        )
        box_ground_friction_force_plot.set_data(
            [box_pos_x[i], box_pos_x[i] + box_ground_friction_force[i]],
            box_pos_y[i] - box_height,
        )

        return (
            finger,
            box,
            finger_box_normal_force_plot,
            finger_box_friction_force_plot,
            box_ground_normal_force_plot,
            box_ground_friction_force_plot,
        )

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()


def animate_1d_box(x_f, x_b, lam_n, lam_f):
    n_frames = x_f.shape[0]
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 10), ylim=(-2, 4))
    (box,) = ax.plot([], [], "r", lw=5)
    (finger,) = ax.plot([], [], "bo", lw=10)
    (force_normal,) = ax.plot([], [], "g>-", lw=2)
    (force_friction,) = ax.plot([], [], "g<-", lw=2)
    (ground,) = ax.plot([-100, 100], [0, 0])

    # initialization function: plot the background of each frame
    def init():
        finger.set_data([], [])
        box.set_data([], [])
        force_normal.set_data([], [])
        force_friction.set_data([], [])
        return (finger, box, force_normal, force_friction)

    # animation function.  This is called sequentially
    def animate(i):
        l = 2.0  # TODO
        height = 1.0  # TODO
        y = 0.0
        finger_height = y + 0.5

        box_com = x_b[i]
        box_shape_x = np.array(
            [box_com - l, box_com + l, box_com + l, box_com - l, box_com - l]
        )
        box_shape_y = np.array([y, y, y + height, y + height, y])
        box.set_data(box_shape_x, box_shape_y)

        force_normal.set_data([box_com - l, box_com - l + lam_n[i]], finger_height)
        force_friction.set_data([box_com, box_com + lam_f[i]], y)
        finger.set_data(x_f[i], finger_height)

        return (finger, box, force_normal, force_friction)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()
