from typing import List, Optional

import numpy as np
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Polynomial, Variable


def inner_product_as_convex_concave(
    prog: MathematicalProgram,
    x: np.ndarray,
    y: np.ndarray,
    nonconvex_constraints: Optional[List] = None,
    slack_vars: Optional[List[Variable]] = None,
    conic_form: bool = True,
) -> Expression:
    """
    Encodes the inner product xᵀy with a convex-concave relaxation.
    xᵀy = Q⁺ - Q⁻ is exact when Q⁺ = (1/4) ‖x + y‖² and Q⁻ = (1/4) ‖x - y‖² (which are quadratic
    equality constraints and hence nonconvex).
    We instead enforce the relaxed (convex) condition:
    xᵀy = Q⁺ - Q⁻, Q⁺ ≥ (1/4) ‖x + y‖² and Q⁻ ≥ (1/4) ‖x - y‖²

    @param nonconvex_constraints: If this list is passed, then it will be populated with
                                  the original (not relaxed) nonconvex constraints.
    @return Q⁺ - Q⁻ (i.e. the relaxed inner product)

    """
    Q_pos = prog.NewContinuousVariables(1, "Q+")[0]
    Q_neg = prog.NewContinuousVariables(1, "Q-")[0]

    if slack_vars is not None:
        slack_vars.append(Q_pos)
        slack_vars.append(Q_neg)

    N = len(x)
    if not len(y) == N:
        raise RuntimeError("x and y must be same length")

    x = x.reshape((N,))
    y = y.reshape((N,))

    expr_pos = 0.25 * (x + y).T @ (x + y)
    expr_neg = 0.25 * (x - y).T @ (x - y)

    # -∞ ≤ ‖x + y‖² - Q ≤ 0
    c_pos = prog.AddQuadraticConstraint(expr_pos - Q_pos, -np.inf, 0)
    c_neg = prog.AddQuadraticConstraint(expr_neg - Q_neg, -np.inf, 0)

    def _add_as_conic(binding) -> None:
        # Quadratic constraints are saved as: lb ≤ .5 xᵀQx + bᵀx ≤ ub
        # which we parse as: 0.5xᵀQx + bᵀx + c <= 0
        # Note: this function only allows constraints with an upper bound
        # (this is only for fast implementation)
        # e.g. we only handle 0.5 xᵀQx + bᵀx ≤ ub
        assert not np.isinf(binding.evaluator().upper_bound())
        assert np.isinf(binding.evaluator().lower_bound())

        Q = binding.evaluator().Q()
        b = binding.evaluator().b()
        c = -binding.evaluator().upper_bound()

        # This part should really be a unit test
        vars = binding.variables()
        expr = Polynomial(0.5 * vars.T @ Q @ vars + b.T @ vars + c)
        expr_target = Polynomial(
            binding.evaluator().Eval(vars).item() - binding.evaluator().upper_bound()
        )
        for mon, coeff in expr.monomial_to_coefficient_map().items():
            coeff = coeff.Evaluate()
            if np.isclose(coeff, 0):
                continue  # skip monomials with zero coeff

            assert mon in expr_target.monomial_to_coefficient_map().keys()

            coeff_target = expr_target.monomial_to_coefficient_map()[mon].Evaluate()
            assert np.isclose(coeff, coeff_target)

        prog.AddQuadraticAsRotatedLorentzConeConstraint(
            Q, b, c, binding.variables(), psd_tol=1e-14
        )

    if conic_form:
        # Remove the constraints and parse them as conic constraints
        prog.RemoveConstraint(c_pos)  # type: ignore
        prog.RemoveConstraint(c_neg)  # type: ignore

        _add_as_conic(c_pos)
        _add_as_conic(c_neg)

    if nonconvex_constraints is not None:
        # Add the constraints to the prog to parse them into bindings, but then remove them.
        const_pos = prog.AddQuadraticConstraint(expr_pos - Q_pos, 0, 0)  # type: ignore
        const_neg = prog.AddQuadraticConstraint(expr_neg - Q_neg, 0, 0)  # type: ignore
        prog.RemoveConstraint(const_pos)  # type: ignore
        prog.RemoveConstraint(const_neg)  # type: ignore

        nonconvex_constraints.append(const_pos)
        nonconvex_constraints.append(const_neg)

    # According to the paper
    # B. Ponton, A. Herzog, S. Schaal, and L. Righetti,
    # “A convex model of humanoid momentum dynamics for multi-contact motion generation,” in 2016 IEEE-RAS
    # adding this cost is sufficient to make the relaxation tight.
    # TODO: Move?
    prog.AddQuadraticCost(1e-8 * Q_pos**2, is_convex=True)
    prog.AddQuadraticCost(1e-8 * Q_neg**2, is_convex=True)

    return Q_pos - Q_neg


def cross_product_2d_as_convex_concave(
    prog: MathematicalProgram,
    x: np.ndarray,
    y: np.ndarray,
    nonconvex_constraints: Optional[List] = None,
    slack_vars: Optional[List[Variable]] = None,
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
        prog, x_tilde, y, nonconvex_constraints, slack_vars
    )

    return relaxed_cross_product
