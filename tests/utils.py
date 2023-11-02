from pydrake.symbolic import Formula, Polynomial, Variable

from planning_through_contact.tools.types import NpExpressionArray, NpFormulaArray
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression


def assert_np_expression_array_eq(
    np1: NpExpressionArray, np2: NpExpressionArray
) -> None:
    for e1, e2 in zip(np1.flatten(), np2.flatten()):
        assert e1.EqualTo(e2)


def assert_num_vars_in_expression_array(e_np, num_vars):
    """
    Asserts that each row of f_np has exactly num_vars variables
    """
    for e in e_np.flatten():
        if isinstance(e, Variable):
            assert num_vars == 1  # only one var, assert this is what we want
        else:
            assert len(e.GetVariables()) == num_vars


def assert_num_vars_in_formula_array(f_np, num_vars):
    """
    Asserts that each row of f_np has exactly num_vars variables
    """
    for f in f_np.flatten():
        expr = convert_formula_to_lhs_expression(f)
        assert len(expr.GetVariables()) == num_vars


def assert_formula_degree(f: Formula, deg: int) -> None:
    expr = convert_formula_to_lhs_expression(f)
    poly = Polynomial(expr)
    assert poly.TotalDegree() == deg


def assert_formula_array_degree(f_np: NpFormulaArray, deg: int) -> None:
    for f in f_np.flatten():
        assert_formula_degree(f, deg)
