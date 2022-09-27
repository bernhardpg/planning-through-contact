import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def animate_1d_box(x_a, x_u, lam_f, lam_n):
    n_frames = x_a.shape[0]
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 10), ylim=(0, 4))
    (finger,) = ax.plot([], [], "bo", lw=5)
    (box,) = ax.plot([], [], "r", lw=5)
    (force_normal,) = ax.plot([], [], "g>-", lw=2)
    (force_friction,) = ax.plot([], [], "g<-", lw=2)

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
        y = 1.0
        finger_height = y + 0.5
        finger.set_data(x_a[i], finger_height)

        box_com = x_u[i]
        box_shape_x = np.array(
            [box_com - l, box_com + l, box_com + l, box_com - l, box_com - l]
        )
        box_shape_y = np.array([y, y, y + height, y + height, y])
        box.set_data(box_shape_x, box_shape_y)

        force_normal.set_data([box_com - l, box_com - l + lam_n[i]], finger_height)
        force_friction.set_data([box_com - lam_f[i], box_com], y)

        return (finger, box, force_normal, force_friction)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    plt.show()


def plot_1d_box_positions(x_a_curves, x_u_curves):
    i = 0
    for x_a_curve, x_u_curve in zip(x_a_curves, x_u_curves):
        s_range = np.arange(0.0, 1.01, 0.01)
        # plt.scatter(curve.ctrl_points[0, :], curve.ctrl_points[1, :])
        x_a_curve_values = np.concatenate(
            [x_a_curve.eval(s) for s in s_range], axis=1
        ).T

        x_u_curve_values = np.concatenate(
            [x_u_curve.eval(s) for s in s_range], axis=1
        ).T

        plt.plot(s_range + i, x_a_curve_values, "r")
        plt.plot(s_range + i, x_u_curve_values, "b")
        i += 1

    plt.legend(["x_a", "x_u"])
    plt.show()
