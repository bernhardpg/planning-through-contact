import matplotlib.pyplot as plt
import numpy as np
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def simple_rotations_test():

    N_DIMS = 2
    NUM_CTRL_POINTS = 3

    prog = MathematicalProgram()

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    FINGER_POS = np.array([[-BOX_WIDTH / 2], [BOX_HEIGHT / 2]])
    GROUND_CONTACT_POS = np.array([[BOX_WIDTH / 2], [-BOX_HEIGHT / 2]])

    f_gravity = np.array([[0], [-BOX_MASS * GRAV_ACC]])
    f_finger = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_finger")
    f_contact = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_contact")
    cos_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "cos_th")
    sin_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "sin_th")

    R_f_gravity = np.concatenate(
        (
            cos_th * f_gravity[0] - sin_th * f_gravity[1],
            sin_th * f_gravity[0] + cos_th * f_gravity[1],
        )
    )
    force_balance = eq(f_finger + f_contact + R_f_gravity, 0)
    moment_balance = eq(
        cross(FINGER_POS, f_finger) + cross(GROUND_CONTACT_POS, f_contact), 0
    )

    prog.AddLinearConstraint(force_balance)
    prog.AddLinearConstraint(moment_balance)

    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_finger.flatten(),
    )
    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_contact.flatten(),
    )

    cos_th_sq = (cos_th * cos_th)[0]
    sin_th_sq = (sin_th * sin_th)[0]
    prog.AddLorentzConeConstraint(1, cos_th_sq[0] + sin_th_sq[0])
    prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])
    prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])

    th_initial = 0
    prog.AddLinearConstraint(cos_th[0, 0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_th[0, 0] == np.sin(th_initial))

    th_final = np.pi / 8
    prog.AddLinearConstraint(cos_th[0, -1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_th[0, -1] == np.cos(th_final))

    result = Solve(prog)
    assert result.is_success()

    results_finger = result.GetSolution(f_finger)
    results_contact = result.GetSolution(f_contact)
    results_cos_th = result.GetSolution(cos_th)
    results_sin_th = result.GetSolution(sin_th)

    fig, axs = plt.subplots(6, 1)
    axs[0].set_title("Finger force x")
    axs[1].set_title("Finger force y")
    axs[0].plot(results_finger[0, :])
    axs[1].plot(results_finger[1, :])

    axs[2].set_title("Contact force x")
    axs[3].set_title("Contact force y")
    axs[2].plot(results_contact[0, :])
    axs[3].plot(results_contact[1, :])

    axs[4].set_title("cos(th)")
    axs[5].set_title("sin(th)")
    axs[4].plot(results_cos_th[0])
    axs[5].plot(results_sin_th[0])

    plt.tight_layout()
    plt.show()

    breakpoint()
