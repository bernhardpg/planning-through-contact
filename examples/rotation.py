import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def mccormick():
    x = sym.Variable("x")
    y = sym.Variable("y")
    variables = np.array([x, y])

    n = variables.shape[0] + 1
    DEGREE = 2
    basis = np.flip(sym.MonomialBasis(variables, DEGREE))  # TODO generalize degree

    def get_monomial_coeffs(poly: sym.Polynomial, basis: npt.NDArray[sym.Monomial]):
        coeff_map = poly.monomial_to_coefficient_map()
        coeffs = np.array(
            [coeff_map.get(m, sym.Expression(0)).Evaluate() for m in basis]
        )
        return coeffs

    def construct_quadratic_constraint(
        poly: sym.Polynomial, basis: npt.NDArray[sym.Monomial], n: int
    ):
        coeffs = get_monomial_coeffs(poly, basis)
        Q = np.zeros((n, n))
        Q[np.triu_indices(n)] = coeffs

        return Q

    test_polynomial = sym.Polynomial(x**2 + y**2 - 1)
    Q = construct_quadratic_constraint(test_polynomial, basis, n)

    prog = MathematicalProgram()
    X = prog.NewSymmetricContinuousVariables(n, "X")  # TODO generalize 3
    prog.AddConstraint(X[0, 0] == 1)
    prog.AddConstraint(X[0, 1] >= 0.5)  # TODO just for testing
    prog.AddConstraint(X[0, 2] >= 0.5)
    prog.AddPositiveSemidefiniteConstraint(X)
    prog.AddConstraint(np.trace(Q @ X) == 0)
    result = Solve(prog)
    assert result.is_success()

    X_result = result.GetSolution(X)
    breakpoint()

    # Plan:
    # 1. Implement above relaxation scheme as a general class
    # 2. Add all of my constraints in to this
    # 3. Check if it is tight!

    return


def test():
    c_th = sym.Variable("c_th")
    s_th = sym.Variable("s_th")
    p_WB = sym.Variable("p_WB")

    R_WB = np.array([[c_th, -s_th], [s_th, c_th]])

    x = np.array([c_th, s_th, p_WB]).reshape((-1, 1))
    basis = sym.MonomialBasis(x, 2)

    def decompose_polynomial_in_monomials(
        p: sym.Polynomial, basis: npt.NDArray[sym.Monomial]
    ):

        coeff_map = p.monomial_to_coefficient_map()
        coeffs = np.array(
            [coeff_map.get(m, sym.Expression(0)).Evaluate() for m in basis]
        ).reshape((-1, 1))
        breakpoint()
        return coeffs

    so_2_constraint = sym.Polynomial(
        c_th**2 + s_th**2 - 1
    )  # == 0 TODO must use Unapply
    a, b = decompose_polynomial_in_monomials(so_2_constraint, basis)
    breakpoint()

    # Plan:
    # 1. Make a symbolic expression
    # 2. Decompose symbolic expression into monomials
    # 3. Add symmetric decision variables to prog
    # 4. Add McCormick constraints to prog

    return


def simple_rotations_test(use_sdp_relaxation: bool = True):
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

    # Force and moment balance
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

    # Force minimization cost
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

    # Path length minimization cost
    cos_cost = np.sum(np.diff(cos_th) ** 2)
    sin_cost = np.sum(np.diff(sin_th) ** 2)
    prog.AddQuadraticCost(cos_cost + sin_cost)

    # SO(2) constraint
    if use_sdp_relaxation:
        aux_vars = prog.NewContinuousVariables(3, NUM_CTRL_POINTS, "X")
        Xs = [np.array([[z[0], z[1]], [z[1], z[2]]]) for z in aux_vars.T]
        xs = [np.vstack([c, s]) for c, s in zip(cos_th.T, sin_th.T)]
        Ms = [np.block([[1, x.T], [x, X]]) for X, x in zip(Xs, xs)]
        for X, M in zip(Xs, Ms):
            prog.AddLinearConstraint(X[0, 0] + X[1, 1] - 1 == 0)
            prog.AddPositiveSemidefiniteConstraint(M)
    else:
        cos_th_sq = (cos_th * cos_th)[0]
        sin_th_sq = (sin_th * sin_th)[0]
        prog.AddLorentzConeConstraint(1, cos_th_sq[0] + sin_th_sq[0])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])

    # Initial and final condition
    th_initial = 0
    prog.AddLinearConstraint(cos_th[0, 0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_th[0, 0] == np.sin(th_initial))

    th_final = 0.5
    prog.AddLinearConstraint(cos_th[0, -1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_th[0, -1] == np.sin(th_final))

    # Solve
    result = Solve(prog)
    assert result.is_success()

    # Rounding and projection onto SO(2)
    to_float = np.vectorize(lambda x: x.Evaluate())
    Ms_result = [to_float(result.GetSolution(M)) for M in Ms]
    ws, vs = zip(*[np.linalg.eig(M) for M in Ms_result])
    idx_highest_eigval = [np.argmax(w) for w in ws]
    vks = [v[:, idx] / v[0, idx] for idx, v in zip(idx_highest_eigval, vs)]

    vk_ps = [vk[[1, 2]] for vk in vks]
    xvs = [np.array([[vk_p[0], -vk_p[1]], [vk_p[1], vk_p[0]]]) for vk_p in vk_ps]

    Us, Ss, Vs = zip(*[np.linalg.svd(xv) for xv in xvs])
    R_hats = [
        U.dot(np.diag([1, np.linalg.det(U) * np.linalg.det(V)])).dot(V.T)
        for U, V in zip(Us, Vs)
    ]

    results_finger = result.GetSolution(f_finger)
    results_contact = result.GetSolution(f_contact)
    results_cos_th = np.array([R[0, 0] for R in R_hats])
    results_sin_th = np.array([R[1, 0] for R in R_hats])
    results_th = np.array(
        [
            [np.arccos(cos_th) for cos_th in results_cos_th],
            [np.arcsin(sin_th) for sin_th in results_sin_th],
        ]
    )

    # Plot
    fig, axs = plt.subplots(7, 1)
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
    axs[4].plot(results_cos_th)
    axs[5].plot(results_sin_th)

    axs[6].plot(results_th.T)
    axs[6].set_title("theta")

    plt.tight_layout()
    plt.show()
