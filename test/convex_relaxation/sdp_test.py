import numpy as np
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation, eliminate_equality_constraints


def test_equality_elimination_with_initial_guess():
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
    prog.AddLinearConstraint(x == y)

    prog.SetInitialGuess(x, 0.7)
    prog.SetInitialGuess(y, 0.7)

    result = Solve(prog)
    assert result.is_success()

    sol = result.GetSolution(prog.decision_variables())

    smaller_prog, _ = eliminate_equality_constraints(prog)
    z = smaller_prog.decision_variables()[0]
    smaller_prog.SetInitialGuess(z, 0.7)
    smaller_result = Solve(smaller_prog)
    assert smaller_result.is_success()


def test_equality_elimination_with_sdp_relaxation():
    # TODO: Fix problem where sdp relaxation doesn't pick up quadratic constraints
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
    prog.AddLinearConstraint(x == y)

    prog.AddQuadraticCost(
        x * y + x + y
    )  # add a cost with a linear term to make the relaxation tight

    # Solve with SDP relaxation
    relaxed_prog, X, _ = create_sdp_relaxation(prog)
    x = X[1:, 0]

    result = Solve(relaxed_prog)
    assert result.is_success()

    sol_relaxation = result.GetSolution(x)

    # Eliminate linear equality constraints, then solve relaxation
    smaller_prog, retrieve_x = eliminate_equality_constraints(prog)
    relaxed_prog, Z, _ = create_sdp_relaxation(smaller_prog)
    relaxed_result = Solve(relaxed_prog)
    assert relaxed_result.is_success()

    z_sol = relaxed_result.GetSolution(Z[1:, 0])
    eliminated_sol = retrieve_x(z_sol)
    assert np.allclose(sol_relaxation, eliminated_sol, atol=1e-3)

    TRUE_SOL = np.array([-0.70710678, -0.70710678])

    assert np.allclose(eliminated_sol, TRUE_SOL, atol=1e-3)


if __name__ == "__main__":
    test_equality_elimination_with_initial_guess()
    test_equality_elimination_with_sdp_relaxation()
