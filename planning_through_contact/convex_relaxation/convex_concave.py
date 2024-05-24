from typing import List, Optional

import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression


def inner_product_as_convex_concave(
    prog: MathematicalProgram,
    x: np.ndarray,
    y: np.ndarray,
    nonconvex_constraints: Optional[List] = None,
) -> Expression:
    """
    Encodes the inner product xᵀy with a convex-concave relaxation.
    xᵀy = Q⁺ - Q⁻ is exact when Q⁺ = ‖x + y‖² and Q⁻ = ‖x - y‖² (which are quadratic
    equality constraints and hence nonconvex).
    We instead enforce the relaxed (convex) condition:
    xᵀy = Q⁺ - Q⁻, Q⁺ ≥ ‖x + y‖² and Q⁻ ≥ ‖x - y‖²

    @param nonconvex_constraints: If this list is passed, then it will be populated with
                                  the original (not relaxed) nonconvex constraints.
    @return Q⁺ - Q⁻ (i.e. the relaxed inner product)

    """
    Q_pos = prog.NewContinuousVariables(1, "Q+")[0]
    Q_neg = prog.NewContinuousVariables(1, "Q-")[0]

    N = len(x)
    if not len(y) == N:
        raise RuntimeError("x and y must be same length")

    x = x.reshape((N,))
    y = y.reshape((N,))

    expr_pos = Q_pos - (x + y).T @ (x + y)  # ≥ 0
    prog.AddQuadraticConstraint(expr_pos, 0, np.inf)

    expr_neg = Q_neg - (x - y).T @ (x - y)  # ≥ 0
    prog.AddQuadraticConstraint(expr_neg, 0, np.inf)

    if nonconvex_constraints is not None:
        # Add the constraints to the prog to parse them into bindings, but then remove them.
        const_pos = prog.AddQuadraticConstraint(expr_pos, 0, 0)
        const_neg = prog.AddQuadraticConstraint(expr_neg, 0, 0)
        prog.RemoveConstraint(const_pos)  # type: ignore
        prog.RemoveConstraint(const_neg)  # type: ignore

        nonconvex_constraints.append(const_pos)
        nonconvex_constraints.append(const_neg)

    # According to the paper
    # B. Ponton, A. Herzog, S. Schaal, and L. Righetti,
    # “A convex model of humanoid momentum dynamics for multi-contact motion generation,” in 2016 IEEE-RAS
    # adding this cost is sufficient to make the relaxation tight.
    # TODO: Move?
    prog.AddQuadraticCost(Q_pos**2, is_convex=True)
    prog.AddQuadraticCost(Q_neg**2, is_convex=True)

    return Q_pos - Q_neg


def cross_product_2d_as_convex_concave(
    prog: MathematicalProgram,
    x: np.ndarray,
    y: np.ndarray,
    nonconvex_constraints: Optional[List] = None,
):
    """
    Adds the cross product x ⊗ y = x_1 * y_2 - x_2 * y_1 with a convex-concave
    relaxation. See the documentation for _inner_product_as_convex_concave for more details.
    """
    if not (len(x) == 2 and len(y) == 2):
        raise RuntimeError("Can only add cross products for two dimensional vectors")

    x = x.reshape((2,))
    y = y.reshape((2,))

    # Encodes x ⊗ y as x̃ᵀy
    x_tilde = np.array([-x[1], x[0]])
    relaxed_cross_product = inner_product_as_convex_concave(
        prog, x_tilde, y, nonconvex_constraints
    )

    return relaxed_cross_product
