import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from geometry.bezier import BezierCurve

PLOT_WIDTH_INCH = 7
PLOT_HEIGHT_INCH = 4.5


def create_static_equilibrium_analysis(
    fb_violation_ctrl_points: npt.NDArray[np.float64],
    mb_violation_ctrl_points: npt.NDArray[np.float64],
):
    MIN_Y_AXIS = -1
    MAX_Y_AXIS = 4

    num_ctrl_points = fb_violation_ctrl_points.shape[1]

    fb_violation = BezierCurve.create_from_ctrl_points(
        fb_violation_ctrl_points
    ).eval_entire_interval()
    mb_violation = np.abs(
        BezierCurve.create_from_ctrl_points(
            mb_violation_ctrl_points
        ).eval_entire_interval()
    )

    fb_norm_violation_ctrl_points = np.apply_along_axis(
        np.linalg.norm, 0, fb_violation_ctrl_points
    ).reshape(
        (1, -1)
    )  # (1, num_ctrl_points)
    fb_norm_violation = BezierCurve.create_from_ctrl_points(
        fb_norm_violation_ctrl_points
    ).eval_entire_interval()

    N = fb_violation.shape[0]
    x_axis = np.linspace(0, num_ctrl_points, N)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Force and torque balance violation")

    axs[0].plot(x_axis, fb_norm_violation)
    axs[0].set_title("Norm of force balance violation")
    axs[0].set(ylabel="[N]")

    axs[1].plot(x_axis, mb_violation)
    axs[1].set_title("Torque balance violation")
    axs[1].set(xlabel="Control point", ylabel="[Nm]")
    axs[1].xaxis.set_ticks(np.arange(0, num_ctrl_points + 1))

    for ax in axs:
        ax.grid()
        ax.set_ylim(MIN_Y_AXIS, MAX_Y_AXIS)

    fig.set_size_inches(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH)  # type: ignore
    fig.tight_layout()  # type: ignore


def create_newtons_third_law_analysis():
    ...
