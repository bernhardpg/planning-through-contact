import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import SdpRelaxation, create_sdp_relaxation
from tools.types import NpVariableArray


def main():
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

    breakpoint()


if __name__ == "__main__":
    main()
