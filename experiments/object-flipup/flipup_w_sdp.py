import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgram

from convex_relaxation.sdp import SdpRelaxation, create_sdp_relaxation
from tools.types import NpVariableArray


def main():
    prog = MathematicalProgram()
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.Variable("z")
    prog.AddDecisionVariables([x, y, z])  # type: ignore
    prog.AddLinearConstraint(2 * x == 2)
    prog.AddLinearConstraint(3 * y == x)
    vars = prog.decision_variables()

    relaxed_prog = create_sdp_relaxation(prog)

    #    lin_eqs = [
    #        LinearConstraint(
    #            b.evaluator().GetDenseA(), b.evaluator().lower_bound(), b.variables()
    #        )
    #        for b in prog.linear_equality_constraints()
    #    ]
    #    lin_eqs_sym = np.array([c.A.dot(c.x) - c.b for c in lin_eqs])
    #    A_eq, b_eq = sym.DecomposeAffineExpressions(lin_eqs_sym)

    relaxation = SdpRelaxation(vars)
    relaxation.add_constraint(2 * x == 2)

    breakpoint()


if __name__ == "__main__":
    main()
