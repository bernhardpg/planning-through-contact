from typing import Callable, Iterable, List

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgramResult

from tools.types import NpExpressionArray, NpFormulaArray


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
