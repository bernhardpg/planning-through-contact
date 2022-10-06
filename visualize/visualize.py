import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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


def plot_1d_box_positions(x_f_curves, x_b_curves):
    i = 0
    for x_f_curve, x_b_curve in zip(x_f_curves, x_b_curves):
        s_range = np.arange(0.0, 1.01, 0.01)
        # plt.scatter(curve.ctrl_points[0, :], curve.ctrl_points[1, :])
        x_f_curve_values = np.concatenate(
            [x_f_curve.eval(s) for s in s_range], axis=1
        ).T

        x_b_curve_values = np.concatenate(
            [x_b_curve.eval(s) for s in s_range], axis=1
        ).T

        plt.plot(s_range + i, x_f_curve_values, "r")
        plt.plot(s_range + i, x_b_curve_values, "b")
        i += 1

    plt.legend(["x_f", "x_b"])
    plt.show()
