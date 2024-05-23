import numpy as np
from pydrake.math import eq
from pydrake.solvers import MakeSemidefiniteRelaxation, MathematicalProgram, Solve

from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs

np.set_printoptions(precision=2, suppress=True)


def test_trace_constraint():
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x").item()
    y = prog.NewContinuousVariables(1, "y").item()

    prog.AddConstraint(x**2 + y**2 == 1)
    prog.AddBoundingBoxConstraint(-1, 1, x)
    prog.AddBoundingBoxConstraint(-1, 1, y)
    prog.AddLinearEqualityConstraint(x == 0)

    prog.AddBoundingBoxConstraint(-1, 1, y)

    prog.AddCost(x * y + x + y)

    relaxed = MakeSemidefiniteRelaxation(prog)
    result = Solve(relaxed)

    assert result.is_success()

    X = relaxed.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    X = X.reshape((int(N), int(N)))

    print("X:")
    print(result.GetSolution(X))

    xy = np.array([x, y])
    print(f"x, y: {result.GetSolution(xy)}")


def test_dist_from_origin():
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x").item()
    y = prog.NewContinuousVariables(1, "y").item()

    prog.AddConstraint(x**2 + y**2 == 1)
    prog.AddBoundingBoxConstraint(-1, 1, x)
    prog.AddBoundingBoxConstraint(-1, 1, y)

    prog.AddCost((x - 1) ** 2 + (y - 0) ** 2)

    prog.AddConstraint(y >= np.sqrt(2) / 2)
    prog.AddConstraint(x <= np.sqrt(2) / 2)

    relaxed = MakeSemidefiniteRelaxation(prog)

    X = relaxed.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    X = X.reshape((int(N), int(N)))

    relaxed.AddCost(np.trace(X))
    result = Solve(relaxed)
    assert result.is_success()

    X_val = result.GetSolution(X)
    print("X:")
    print(X_val)

    xy = np.array([x, y])
    xy_sol = result.GetSolution(xy)
    print(f"x, y: {xy_sol}")

    r_val = xy_sol.reshape((1, 2))
    plot_cos_sine_trajs(r_val)


def test_more_knot_points():
    NUM_CTRL_POINTS = 10
    NUM_DIMS = 2

    prog = MathematicalProgram()
    r = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS, "r")

    # Constrain the points to lie on the unit circle
    for i in range(NUM_CTRL_POINTS):
        r_s = r[:, i]
        so_2_constraint = r_s.T.dot(r_s) == 1
        prog.AddConstraint(so_2_constraint)

    # Initial conditions
    th_initial = np.pi + 1e-4
    th_final = 0
    create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

    r_s = create_r_vec_from_angle(th_initial)
    r_t = create_r_vec_from_angle(th_final)

    # Minimize distance to goal
    for i in range(NUM_CTRL_POINTS):
        r_i = r[:, i]
        prog.AddCost((r_i - r_t).T @ (r_i - r_t))
        # prog.AddCost((r_i).T @ (r_i))

    def two_d_rot_matrix(r):
        return np.array([[r[0], -r[1]], [r[1], r[0]]])

    delta_th_max = 0.2
    two_d_rotation_matrix_from_angle(delta_th_max)

    # Constraint on max movement
    for i in range(NUM_CTRL_POINTS - 1):
        r_i = r[:, i]
        r_i_next = r[:, i + 1]
        delta_r_i = r_i_next - r_i
        two_d_rot_matrix(delta_r_i)
        R_i = two_d_rot_matrix(r_i)
        # This seems to work well
        prog.AddConstraint(delta_r_i.T @ delta_r_i <= delta_th_max**2)
        # prog.AddConstraint(le(delta_r_i, delta_th_max))
        # prog.AddConstraint(ge(delta_r_i, -delta_th_max))

        # delta_theta = delta_R_i.dot(R_i.T)[1, 0]
        # prog.AddConstraint(delta_theta <= -delta_th_max)

        R_i_next = two_d_rot_matrix(r_i_next)
        R_i_next @ R_i.T
        # prog.AddQuadraticConstraint(delta_R_i[0, 0], -np.inf, 0)
        # prog.AddQuadraticConstraint(delta_R_i[1, 0], -np.inf, 0)
        # This seems to be doing something:
        # prog.AddQuadraticConstraint(delta_R_i[0, 0], 0.3, np.inf)
        # prog.AddQuadraticConstraint(delta_R_i[1, 0], 0.3, np.inf)
    #
    # Initial condition
    for c in eq(r[:, 0], r_s):
        prog.AddConstraint(c)

    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    X = relaxed_prog.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    X = X.reshape((int(N), int(N)))

    # relaxed_prog.AddCost(1e-6 * np.trace(X))

    result = Solve(relaxed_prog)
    assert result.is_success()
    print(f"Cost: {result.get_optimal_cost()}")

    X_val = result.GetSolution(X)

    tol = 1e-3
    num_nonzero_eigvals = len(
        [val for val in np.linalg.eigvals(X_val) if np.abs(val) >= tol]
    )
    print(f"Eigvals: {np.linalg.eigvals(X_val)}")
    print(f"Rank of X: {num_nonzero_eigvals}")
    print(f"cost: {result.get_optimal_cost()}")

    r_val = result.GetSolution(r)

    cos_vals = r_val[0, :]
    th_vals = np.array([np.arccos(cos) for cos in cos_vals])
    delta_ths = th_vals[1:] - th_vals[:-1]

    print(f"delta_ths: {delta_ths}")

    plot_cos_sine_trajs(r_val.T)


test_more_knot_points()
