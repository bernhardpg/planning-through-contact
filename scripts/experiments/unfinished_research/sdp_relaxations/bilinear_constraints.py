import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge, le
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    SemidefiniteRelaxationOptions,
    Solve,
    SolverOptions,
)

from planning_through_contact.convex_relaxation.sdp import (
    get_X_from_semidefinite_relaxation,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.utils import evaluate_np_expressions_array


def example_1():
    """
    Example 1.
    Cost tight.
    With bilinear constraint: x y = 1
    Here the cost is 2
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y - 1, 0, 0)

    prog.AddQuadraticCost(x**2)
    prog.AddQuadraticCost(y**2)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()
    relaxed_prog = MakeSemidefiniteRelaxation(prog, options)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    # relaxed_prog.AddCost(1e-5 * np.trace(X))

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")
    print(f"y: {relaxed_result.GetSolution(y)}")

    x_val = relaxed_result.GetSolution(x)
    y_val = relaxed_result.GetSolution(y)

    print(f"x * y - x - 1 = {x_val * y_val - x_val - 1}")


def example_1_homogenous():
    """ """
    prog = MathematicalProgram()
    S = prog.NewSymmetricContinuousVariables(2)
    prog.AddPositiveSemidefiniteConstraint(S)
    prog.AddLinearCost(S[0, 0] + S[1, 1])
    prog.AddLinearConstraint(S[0, 1] == 1)
    relaxed_result = Solve(prog)
    S_val = relaxed_result.GetSolution(S)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(S_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print("S:")
    print(S_val)

    eigvals, eigvecs = np.linalg.eig(S_val)
    idx = np.flatnonzero(eigvals)[0]

    x_1 = np.sqrt(eigvals[idx]) * eigvecs[:, idx]
    print(f"x_1 = {x_1}")

    x_2 = -np.sqrt(eigvals[idx]) * eigvecs[:, idx]
    print(f"x_2 = {x_2}")


def example_4():
    """
    Example 1 with xy = 0.
    Is this cost tight?
    WIP: Cost tight ?
    Here the cost is 1
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y, 0, 0)
    prog.AddLinearConstraint(x >= 0)
    prog.AddLinearConstraint(y >= 0)

    prog.AddQuadraticCost((x - 0.2) ** 2)
    prog.AddQuadraticCost((y - 1) ** 2)

    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")
    print(f"")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")
    print(f"y: {relaxed_result.GetSolution(y)}")


def example_2():
    """
    Example 2.
    Cost tight and rank tight.
    With bilinear constraint: x y = x + 1
    Here the cost is 0.67
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y - x - 1, 0, 0)

    prog.AddQuadraticCost(x**2)
    prog.AddQuadraticCost(y**2)

    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    # relaxed_prog.AddCost(1e-5 * np.trace(X))

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")
    print(f"y: {relaxed_result.GetSolution(y)}")

    x_val = relaxed_result.GetSolution(x)
    y_val = relaxed_result.GetSolution(y)

    print(f"x * y - x - 1 = {x_val * y_val - x_val - 1}")


def example_2_no_implied_constraints():
    """
    Example 2 without implied constraints and trace cost. Still tight
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y - x - 1, 0, 0)

    prog.AddQuadraticCost(x**2)
    prog.AddQuadraticCost(y**2)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    relaxed_prog = MakeSemidefiniteRelaxation(prog, options)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")
    print(f"y: {relaxed_result.GetSolution(y)}")

    x_val = relaxed_result.GetSolution(x)
    y_val = relaxed_result.GetSolution(y)

    print(f"x * y - x - 1 = {x_val * y_val - x_val - 1}")


def example_2_smaller_psd_cones():
    """
    Example 2.
    3x3 PSD replaced by 3 2x2 PSD constraints.
    Cost tight but not rank tight.
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y - x - 1, 0, 0)

    prog.AddQuadraticCost(x**2)
    prog.AddQuadraticCost(y**2)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    relaxed_prog = MakeSemidefiniteRelaxation(prog, options)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    # remove 3x3 PSD constraint and replace by 3 2x2 PSD constraints:
    psd_constraint = relaxed_prog.positive_semidefinite_constraints()[0]
    relaxed_prog.RemoveConstraint(psd_constraint)  # type: ignore
    X1 = X[:2, :2]
    X2 = X[1:, 1:]
    X3 = X[np.ix_([0, 2], [0, 2])]
    relaxed_prog.AddPositiveSemidefiniteConstraint(X1)
    relaxed_prog.AddPositiveSemidefiniteConstraint(X2)
    relaxed_prog.AddPositiveSemidefiniteConstraint(X3)

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"Primal cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")
    print(f"y: {relaxed_result.GetSolution(y)}")

    x_val = relaxed_result.GetSolution(x)
    y_val = relaxed_result.GetSolution(y)

    print(f"x * y - x - 1 = {x_val * y_val - x_val - 1}")


def solve_lcp_dual(
    M: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    dual = MathematicalProgram()
    n = M.shape[0]
    assert M.shape == (n, n)  # M must be symmetric

    assert len(q) == n

    lam = dual.NewContinuousVariables(n, "lambda")  # Mx + q >= 0
    mu = dual.NewContinuousVariables(n, "mu")  # x >= 0
    nu = dual.NewContinuousVariables(1, "nu").item()  # xᵀ(Mx + q) = 0
    gamma = dual.NewContinuousVariables(1, "gamma").item()  # slack var

    dual.AddLinearCost(-gamma)  # max gamma

    dual.AddLinearConstraint(ge(lam, 0))
    dual.AddLinearConstraint(ge(mu, 0))

    P_tilde = np.eye(n) + nu * M
    q_tilde = (nu * q - M.T @ lam - mu).reshape((n, 1))
    r_tilde = -q.T @ lam - gamma
    H = np.block([[P_tilde, 0.5 * q_tilde], [0.5 * q_tilde.T, r_tilde]])

    assert H.shape == (n + 1, n + 1)
    dual.AddPositiveSemidefiniteConstraint(H)

    solver_options = SolverOptions()
    # solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    dual_result = Solve(dual, solver_options=solver_options)
    print(f"Dual result: {dual_result.get_solution_result()}")
    assert dual_result.is_success()
    print(f"Dual cost: {-dual_result.get_optimal_cost():.2f}")

    # This follows the procedure in:
    # X. J. Zheng, X. L. Sun, D. Li, and Y. F. Xu,
    # “On zero duality gap in nonconvex quadratic programming problems,”
    A_star = evaluate_np_expressions_array(P_tilde, dual_result)
    b_star = 0.5 * evaluate_np_expressions_array(q_tilde, dual_result)

    print(f"A* =\n{A_star}")
    eigvals = np.linalg.eigvals(A_star)
    print(f"eigs(A*): {eigvals}")

    ZERO_TOL = 1e-5
    if np.all(eigvals >= ZERO_TOL):
        print("A* ≻ 0 (strictly PSD), so the relaxation is tight.")
    else:
        print("A* singular, so no guarantee that the relaxation is tight")

    x_sol = -np.linalg.solve(A_star, b_star).flatten()
    print(f"x_sol = {x_sol}")
    return x_sol


def solve_lcp_dual_w_redundant_constraints(
    M: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    dual = MathematicalProgram()
    n = M.shape[0]
    assert M.shape == (n, n)  # M must be symmetric

    assert len(q) == n

    lam = dual.NewContinuousVariables(n, "lambda")  # Mx + q >= 0
    mu = dual.NewContinuousVariables(n, "mu")  # x >= 0
    nu = dual.NewContinuousVariables(n, "nu")  # xᵢᵀ(Mx + q)ᵢ = 0
    gamma = dual.NewContinuousVariables(1, "gamma").item()  # slack var

    dual.AddLinearCost(-gamma)  # max gamma

    dual.AddLinearConstraint(ge(lam, 0))
    dual.AddLinearConstraint(ge(mu, 0))

    def E_ii(i: int) -> npt.NDArray[np.float64]:
        E = np.zeros((n, n))
        E[i, i] = 1
        return E

    P_tilde = np.eye(n) + sum([nu[i] * E_ii(i) @ M for i in range(n)])
    q_tilde = (sum([nu[i] * E_ii(i) @ q for i in range(n)]) - M.T @ lam - mu).reshape(
        (n, 1)
    )
    r_tilde = -q.T @ lam - gamma
    H = np.block([[P_tilde, 0.5 * q_tilde], [0.5 * q_tilde.T, r_tilde]])

    assert H.shape == (n + 1, n + 1)
    dual.AddPositiveSemidefiniteConstraint(H)

    # TODO: Why is this dual only feasible when the dual variables are equal
    # (which makes the two dual problems equal?)
    for nu_i in nu:
        dual.AddLinearEqualityConstraint(nu_i == nu[0])

    solver_options = SolverOptions()
    # solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    dual_result = Solve(dual, solver_options=solver_options)
    print(f"Dual result: {dual_result.get_solution_result()}")
    assert dual_result.is_success()
    print(f"Dual cost: {-dual_result.get_optimal_cost():.2f}")

    # This follows the procedure in:
    # X. J. Zheng, X. L. Sun, D. Li, and Y. F. Xu,
    # “On zero duality gap in nonconvex quadratic programming problems,”
    A_star = evaluate_np_expressions_array(P_tilde, dual_result)
    b_star = evaluate_np_expressions_array(q_tilde, dual_result)

    print(f"A*: {A_star}")
    eigvals = np.linalg.eigvals(A_star)
    print(f"eigvals(A*): {eigvals}")

    if np.all(eigvals >= 0):
        print("A* ≻ 0 (strictly PSD), so the relaxation is tight.")
        x_sol = np.linalg.solve(A_star, b_star)
        return x_sol

    lam_res = dual_result.GetSolution(lam)
    nu_res = dual_result.GetSolution(nu)
    print(f"Dual lambda: {lam_res}")
    print(f"Dual nu: {nu_res}")

    return None


def example_3_lcp_loose():
    """
    Example 3, a "standard" LCP formulation.

    x >= 0, Mx + q >= 0,
    x_i (Mx + q)_i = 0
    i = 1, .., n
    """
    prog = MathematicalProgram()
    n = 2
    x = prog.NewContinuousVariables(n, "x")

    M = np.array([[0, 1], [1, 0]])
    q = np.array([-1, 1])

    prog.AddQuadraticCost(x.T @ x)  # type: ignore

    z = M @ x + q

    prog.AddLinearConstraint(ge(x, 0))
    prog.AddLinearConstraint(ge(z, 0))

    complimentarity_constraint = z * x
    for c in complimentarity_constraint:
        prog.AddQuadraticConstraint(c, 0, 0)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    relaxed_prog = MakeSemidefiniteRelaxation(prog, options)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    relaxed_prog.AddCost(1e-5 * np.trace(X))

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")
    print(f"X =\n{X_val[1:,1:]}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigs(X): {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"Primal cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")

    x_sol = solve_lcp_dual(M, q)

    all_satisfied = True
    for const in prog.GetAllConstraints():
        all_satisfied = all_satisfied and const.evaluator().CheckSatisfied(  # type: ignore
            x_sol, tol=1e-4  # type: ignore
        )
    print(f"x_sol feasible: {all_satisfied}")


def example_3_lcp_tight():
    """
    Example 3, a "standard" LCP formulation.

    x >= 0, Mx + q >= 0,
    x_i (Mx + q)_i = 0
    i = 1, .., n
    """
    prog = MathematicalProgram()
    n = 2
    x = prog.NewContinuousVariables(n, "x")

    M = np.array([[0, 1], [1, 0]])
    # Here both entries of q are negative, which for some reason makes this tight!
    q = np.array([-1, -1])

    prog.AddQuadraticCost(x.T @ x)  # type: ignore

    prog.AddLinearConstraint(ge(x, 0))
    prog.AddLinearConstraint(ge(M @ x + q, 0))

    complimentarity_constraint = x * (M @ x + q)
    for c in complimentarity_constraint:
        prog.AddQuadraticConstraint(c, 0, 0)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    relaxed_prog = MakeSemidefiniteRelaxation(prog, options)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    solver_options = SolverOptions()
    # solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = Solve(relaxed_prog, solver_options=solver_options)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")
    print(f"x: {relaxed_result.GetSolution(x)}")

    x_sol = solve_lcp_dual(M, q)

    all_satisfied = True
    for const in prog.GetAllConstraints():
        all_satisfied = all_satisfied and const.evaluator().CheckSatisfied(  # type: ignore
            x_sol, tol=1e-4  # type: ignore
        )
    print(f"x_sol feasible: {all_satisfied}")


def example_forces_torques():
    """
    This example is more like the footstep planning case with forces.
    It is tight.
    """
    prog = MathematicalProgram()
    theta_ddot_des = 1.0

    f = prog.NewContinuousVariables(2, 1, "f")
    p = prog.NewContinuousVariables(2, 1, "p")

    a = prog.NewContinuousVariables(1, 1, "a").item()

    # prog.AddLinearConstraint(eq(f, np.array([[0.2], [0.2]])))
    prog.AddLinearConstraint(f[1, 0] + a == 1)
    prog.AddLinearConstraint(p[1, 0] == 0)

    tau = cross_2d(p, f)
    prog.AddQuadraticConstraint(tau - theta_ddot_des, 0, 0)

    prog.AddCost((f.T @ f).item())  # type: ignore
    prog.AddCost(a**2)  # type: ignore

    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    relaxed_prog.AddCost(1e-5 * np.trace(X))

    relaxed_result = Solve(relaxed_prog)
    X_val = relaxed_result.GetSolution(X)

    print(f"Solved with: {relaxed_result.get_solver_id().name()}")

    print(f"cost: {relaxed_result.get_optimal_cost():.2f}")

    eigvals = [e.real for e in np.linalg.eigvals(X_val)]
    print(f"eigvals: {', '.join([f"{e:.2f}" for e in eigvals])}")

    print(f"f: {relaxed_result.GetSolution(f)}")
    print(f"p: {relaxed_result.GetSolution(p)}")
    print(f"a: {relaxed_result.GetSolution(a)}")


# Set the print options to display 2 decimal places and no scientific
np.set_printoptions(precision=2, suppress=True)

# example_1()
# example_1_homogenous()
# example_2()
# example_2_no_implied_constraints()
# example_2_smaller_psd_cones()
example_3_lcp_loose()
# example_3_lcp_tight()
# example_4()
