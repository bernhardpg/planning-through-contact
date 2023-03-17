import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation
from tools.types import NpVariableArray


def _get_sol_from_svd(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    eigenvals, eigenvecs = np.linalg.eig(X)
    idx_highest_eigval = np.argmax(eigenvals)
    solution_nonnormalized = eigenvecs[:, idx_highest_eigval]
    solution = solution_nonnormalized / solution_nonnormalized[0]
    return solution


def main():
    prog = MathematicalProgram()
    c = sym.Variable("c")
    s = sym.Variable("s")
    prog.AddDecisionVariables(np.array([c, s]))
    prog.AddLinearConstraint(c + s >= 1)
    prog.AddConstraint(c**2 + s**2 == 1)

    vars = prog.decision_variables()

    relaxed_prog, X = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    assert result.is_success()
    X_val = result.GetSolution(X)
    rounded_sols = _get_sol_from_svd(X_val)

    # TODO: Also implement cost for SDP!

    breakpoint()


def test():
    prog = MathematicalProgram()
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.Variable("z")
    prog.AddDecisionVariables([x, y, z])  # type: ignore
    prog.AddLinearConstraint(2 * x == 2)
    prog.AddLinearConstraint(3 * y == x + 1)
    prog.AddLinearConstraint(-2 <= x + 2 * y)
    prog.AddLinearConstraint(x + 2 * y <= 2)
    prog.AddConstraint(x * y == 2)
    prog.AddConstraint(x**2 + y**2 <= 1)
    prog.AddConstraint(-2 <= x * z)
    prog.AddConstraint(x * z <= 2)
    vars = prog.decision_variables()

    relaxed_prog, X = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    assert result.is_success()
    X_val = result.GetSolution(X)

    # TODO: Also implement cost for SDP!

    breakpoint()


if __name__ == "__main__":
    main()
