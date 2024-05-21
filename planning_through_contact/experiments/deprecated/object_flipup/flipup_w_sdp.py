import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from convex_relaxation.sdp import create_sdp_relaxation
from geometry.utilities import cross_2d
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve
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

    R = np.array([[c, -s], [s, c]])

    f_1 = np.array([sym.Variable("f_1_x"), sym.Variable("f_1_y")]).reshape((-1, 1))
    f_2 = np.array([sym.Variable("f_2_x"), sym.Variable("f_2_y")]).reshape((-1, 1))
    f = np.concatenate((f_1, f_2))
    p_1 = np.array([sym.Variable("p_x"), sym.Variable("p_y")]).reshape((-1, 1))
    p_2 = np.array([-2, -2])  # corner

    mg = 1 * 9.81 * np.array([0, -1]).reshape((-1, 1))
    sum_of_forces = R.dot(mg) + f_1 + f_2
    sum_of_torques = cross_2d(p_1, f_1) + cross_2d(p_2, f_2)

    prog.AddDecisionVariables(np.array([c, s]))
    prog.AddDecisionVariables(f_1)
    prog.AddDecisionVariables(f_2)
    prog.AddDecisionVariables(p_1)

    prog.AddLinearConstraint(c + s >= 1)  # non penetration
    prog.AddConstraint(c**2 + s**2 == 1)  # so 2
    prog.AddConstraint(eq(sum_of_forces, 0))
    for c in eq(sum_of_torques, 0).flatten():
        prog.AddConstraint(c)

    # constraint = le(np.array([f[0, 0] + c, -f[1, 0] + c]), np.array([10, -10]))
    # test constraint with two sided bound
    # prog.AddLinearConstraint(constraint)

    prog.AddCost(f.T.dot(f).item())

    vars = prog.decision_variables()

    relaxed_prog, X = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    assert result.is_success()
    X_val = result.GetSolution(X)
    x_val = X_val[:, 0]
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
