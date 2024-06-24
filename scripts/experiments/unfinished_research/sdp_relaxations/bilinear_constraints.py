import numpy as np
from pydrake.math import eq
from pydrake.solvers import (
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    SemidefiniteRelaxationOptions,
    Solve,
)

from planning_through_contact.convex_relaxation.sdp import (
    get_X_from_semidefinite_relaxation,
)
from planning_through_contact.geometry.utilities import cross_2d


def example_1():
    """
    Example 1 (not tight)
    With bilinear constraint: x y = 1
    Here the cost is 2
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, 1, "x").item()
    y = prog.NewContinuousVariables(1, 1, "y").item()
    prog.AddQuadraticConstraint(x * y - 1, 0, 0)

    prog.AddQuadraticCost(x**2)
    prog.AddQuadraticCost(y**2)

    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    X = get_X_from_semidefinite_relaxation(relaxed_prog)

    relaxed_prog.AddCost(1e-5 * np.trace(X))

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


def example_2():
    """
    Example 2 (tight)
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

    relaxed_prog.AddCost(1e-5 * np.trace(X))

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
    Example 2 without implied constraints. Still tight
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

    relaxed_prog.AddCost(1e-5 * np.trace(X))

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


# Set the print options to display 2 decimal places
np.set_printoptions(precision=2)

# example_1()
# example_2()
example_2_no_implied_constraints()
