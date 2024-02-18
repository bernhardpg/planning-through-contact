from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MosekSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.tools.utils import evaluate_np_expressions_array


def simple_test():
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddCost(x**2)
    prog.AddQuadraticConstraint(x * y, 0, 0)
    prog.AddLinearConstraint(x >= 0)
    prog.AddLinearConstraint(y >= 0)
    prog.AddLinearConstraint(x + y == 1)

    result = Solve(prog)

    print(f"is success: {result.is_success()}")
    print(f"cost: {result.get_optimal_cost()}")
    print(f"x: {result.GetSolution(x)}")
    print(f"y: {result.GetSolution(y)}")

    relaxed_prog = MakeSemidefiniteRelaxation(prog)

    solver = MosekSolver()
    # solver = ClarabelSolver()
    solver_options = SolverOptions()
    # solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    result = solver.Solve(relaxed_prog, solver_options=solver_options)
    print(f"is success: {result.is_success()}")
    print(f"cost: {result.get_optimal_cost()}")
    print(f"x: {result.GetSolution(x)}")
    print(f"y: {result.GetSolution(y)}")

    Y = relaxed_prog.positive_semidefinite_constraints()[0].variables().reshape((3, 3))
    print("Y:")
    print(result.GetSolution(Y))


def _check_feasible(x, y) -> bool:
    if x < 0 or y < 0:
        return False

    prog = MathematicalProgram()
    xy = prog.NewContinuousVariables(1, "xy")[0]

    x_sq = prog.NewContinuousVariables(1, "x_sq")[0]
    y_sq = prog.NewContinuousVariables(1, "y_sq")[0]
    prog.AddLinearConstraint(xy == 0)

    X = np.array(
        [
            [1, x, y],
            [x, x_sq, xy],
            [y, xy, y_sq],
        ]
    )
    prog.AddPositiveSemidefiniteConstraint(X)
    result = Solve(prog)

    return result.is_success()


def plot_feasible_region():
    # Step 2: Create a grid of x and y values
    x_values = np.linspace(-1.5, 1.5, 10)  # Adjust the range and density as needed
    y_values = np.linspace(-1.5, 1.5, 10)  # Adjust the range and density as needed
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Step 3: Evaluate the function on each point of the grid
    z = np.zeros(
        (len(y_values), len(x_values))
    )  # Initialize the result grid with zeros
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            z[j, i] = _check_feasible(x, y)  # Store the result (True/False as 1/0)

    # Step 4: Plot the region using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.contourf(
        x_values, y_values, z, levels=[0, 0.5, 1], cmap="Greens"
    )  # Use a suitable colormap
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Region where my_function returns True")
    plt.grid(True)
    plt.show()


def _generate_symmetric_matrices(
    n, size, to_zero: Optional[List[Tuple[int, int]]] = None
):
    """
    Generate n symmetric matrices of a given size.

    Parameters:
    - n: The number of symmetric matrices to generate.
    - size: The size of the square matrices.

    Returns:
    - A list of n symmetric matrices.
    """
    symmetric_matrices = []
    for _ in range(n):
        # Step 1: Generate the upper triangle of the matrix
        upper_triangle = np.random.rand(size, size) - 1 / 2
        upper_triangle = np.triu(upper_triangle)

        # Step 2: Mirror the upper triangle to the lower triangle
        symmetric_matrix = (
            upper_triangle + upper_triangle.T - np.diag(upper_triangle.diagonal())
        )

        symmetric_matrices.append(symmetric_matrix)

    if to_zero is not None:
        for i, j in to_zero:
            for matrix in symmetric_matrices:
                matrix[i, j] = 0
                matrix[j, i] = 0

    return symmetric_matrices


def _solve(C: npt.NDArray) -> Optional[npt.NDArray]:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]
    xy = prog.NewContinuousVariables(1, "xy")[0]

    x_sq = prog.NewContinuousVariables(1, "x_sq")[0]
    y_sq = prog.NewContinuousVariables(1, "y_sq")[0]
    prog.AddLinearConstraint(x >= 0)
    prog.AddLinearConstraint(y >= 0)
    # prog.AddLinearConstraint(x <= 1)
    # prog.AddLinearConstraint(y <= 1)
    prog.AddLinearConstraint(xy == 0)

    X = np.array(
        [
            [1, x, y],
            [x, x_sq, xy],
            [y, xy, y_sq],
        ]
    )

    prog.AddLinearCost(np.sum(C * X))
    prog.AddPositiveSemidefiniteConstraint(X)
    prog.AddLinearCost(1e-6 * np.trace(X))

    result = Solve(prog)

    X_sol = evaluate_np_expressions_array(X, result)
    # print(f"rank(X): {np.linalg.matrix_rank(X_sol, tol=1e-2)}")
    print(f"eigvals(X): {np.linalg.eigvals(X_sol)}")

    if result.is_success():
        vec = np.array([x, y])
        return result.GetSolution(vec)
    else:
        return None


def plot_solutions():
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    to_zero = [(0, 0)]
    # to_zero = [(0, 0), (1, 1), (2, 2), (2, 1)]
    # to_zero = [(0, 0), (1, 1), (2, 2)]
    # to_zero = [(0, 1), (0, 2)]
    Cs = _generate_symmetric_matrices(500, 3, to_zero=to_zero)

    sols = [_solve(C) for C in Cs]
    sols = [sol for sol in sols if sol is not None]
    xs = np.array([sol[0] for sol in sols])
    ys = np.array([sol[1] for sol in sols])

    # Step 4: Plot the region using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.grid(True)
    plt.show()


np.random.seed(0)

# plot_feasible_region()
plot_solutions()
