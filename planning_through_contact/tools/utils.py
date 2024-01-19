from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.tools.types import NpExpressionArray, NpFormulaArray


def convert_formula_to_lhs_expression(form: sym.Formula) -> sym.Expression:
    lhs, rhs = form.Unapply()[1]  # type: ignore
    expr = lhs - rhs
    return expr


convert_np_formulas_array_to_lhs_expressions: Callable[
    [NpFormulaArray], NpExpressionArray
] = np.vectorize(convert_formula_to_lhs_expression)

convert_np_exprs_array_to_floats: Callable[
    [NpExpressionArray], npt.NDArray[np.float64]
] = np.vectorize(lambda expr: expr.Evaluate())


def evaluate_np_formulas_array(
    formulas: NpFormulaArray, result: MathematicalProgramResult
) -> npt.NDArray[np.float64]:
    expressions = convert_np_formulas_array_to_lhs_expressions(formulas)
    evaluated_expressions = convert_np_exprs_array_to_floats(
        result.GetSolution(expressions)
    )
    return evaluated_expressions


def evaluate_np_expressions_array(
    expr: NpExpressionArray, result: MathematicalProgramResult
) -> npt.NDArray[np.float64]:
    from_expr_to_float = np.vectorize(lambda expr: expr.Evaluate())
    solutions = from_expr_to_float(result.GetSolution(expr))
    return solutions


def calc_displacements(
    vars, dt: Optional[float] = None
) -> List[npt.NDArray[np.float64]]:
    if dt is not None:
        scale = 1 / dt
    else:
        scale = 1

    displacements = [
        (var_next - var_curr) * scale for var_curr, var_next in zip(vars[:-1], vars[1:])  # type: ignore
    ]
    return displacements


def skew_symmetric_so2(a):
    return np.array([[0, -a], [a, 0]])


def approx_exponential_map(omega_hat, num_dims: int = 2):
    # Approximates the exponential map (matrix exponential) by truncating terms of higher degree than 2
    return np.eye(num_dims) + omega_hat + 0.5 * omega_hat @ omega_hat
