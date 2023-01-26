import warnings
from typing import Dict, List, Tuple

import pydrake.symbolic as sym  # type: ignore
from pydrake.solvers import MathematicalProgram

from geometry.orientation.contact_pair_2d import ContactPointConstraints
from tools.utils import convert_formula_to_lhs_expression


def _create_aux_variable(m: sym.Monomial) -> sym.Variable:
    name = "".join(str(m).split(" "))
    return sym.Variable(name)


def _create_mccormick_envelopes(u, v, w, variable_bounds) -> List[sym.Formula]:
    u_name = u.get_name().split("(")[0]
    v_name = v.get_name().split("(")[0]

    if u_name not in variable_bounds.keys():
        warnings.warn(f"Name not in variable bounds: {u_name}")

    if v_name not in variable_bounds.keys():
        warnings.warn(f"Name not in variable bounds: {v_name}")

    # TODO remove this after bounds are strictly enforced
    BIG_NUM = 999
    u_L, u_U = variable_bounds.get(u_name, (-BIG_NUM, BIG_NUM))
    v_L, v_U = variable_bounds.get(v_name, (-BIG_NUM, BIG_NUM))

    bound_corners = [u_L * v_L, u_L * v_U, u_U * v_L, u_U * v_U]
    w_U = max(bound_corners)
    w_L = min(bound_corners)

    var_bounds = [
        w_L <= w,
        w <= w_U,
        u_L <= u,
        u <= u_U,
        v_L <= v,
        v <= v_U,
    ]
    mccormick_envelopes = [
        w >= u_L * v + u * v_L - u_L * v_L,
        w >= u_U * v + u * v_U - u_U * v_U,
        w <= u_U * v + u * v_L - u_U * v_L,
        w <= u_L * v + u * v_U - u_L * v_U,
    ]
    return sum([var_bounds, mccormick_envelopes], [])


def relax_bilinear_expression(
    expr: sym.Expression, variable_bounds: Dict[str, Tuple[float, float]]
) -> Tuple[sym.Expression, List[sym.Variable], List[sym.Formula]]:
    poly = sym.Polynomial(expr)

    coeff_map = poly.monomial_to_coefficient_map()
    monomials, coeffs = zip(*list(coeff_map.items()))

    max_deg_is_2 = all([m.total_degree() <= 2 for m in monomials])
    num_vars_is_2 = all(
        [m.GetVariables().size() == 2 for m in monomials if m.total_degree() == 2]
    )
    if not max_deg_is_2 or not num_vars_is_2:
        raise ValueError(
            "Monomials of degree higher than 2 was found in formula. All terms must be at most bilinear!"
        )

    bilinear_terms_replaced_with_aux_vars = [
        _create_aux_variable(m) if m.total_degree() == 2 else m.ToExpression()
        for m in monomials
    ]
    expr_with_bilinear_terms_replaced = sum(
        [c * m for c, m in zip(coeffs, bilinear_terms_replaced_with_aux_vars)]
    )

    indices_of_new_vars = [i for i, m in enumerate(monomials) if m.total_degree() == 2]
    new_aux_variables = [
        bilinear_terms_replaced_with_aux_vars[i] for i in indices_of_new_vars
    ]
    replaced_monomials = [monomials[i] for i in indices_of_new_vars]

    mccormick_envelope_constraints = []
    for w, monomial in zip(new_aux_variables, replaced_monomials):
        u, v = list(monomial.GetVariables())
        mccormick_envelope_constraints.extend(
            _create_mccormick_envelopes(u, v, w, variable_bounds)
        )

    return (
        expr_with_bilinear_terms_replaced,
        new_aux_variables,
        mccormick_envelope_constraints,
    )


# TODO: This is moved here to avoid duplicated code. However, the code in itself should be cleaned up.
def add_bilinear_expressions_to_prog(
    constraints: ContactPointConstraints,
    prog: MathematicalProgram,
    variable_bounds: Dict[str, Tuple[float, float]],
    only_A: bool = False,
    only_B: bool = False,
) -> None:  # type: ignore
    # TODO: Remove this code which is only for analysis
    if only_A:
        constraints_to_use = [constraints.in_frame_A]
    elif only_B:
        constraints_to_use = [constraints.in_frame_B]
    else: # NOTE: There is a type error here, but it is just for analysis, so leave this for now
        constraints_to_use = constraints
        
    for constraint in constraints_to_use:
        for formula in constraint.flatten():
            expr = convert_formula_to_lhs_expression(formula)
            expression_is_linear = sym.Polynomial(expr).TotalDegree() == 1
            if expression_is_linear:
                prog.AddLinearConstraint(expr == 0)
            else:
                (
                    relaxed_expr,
                    new_vars,
                    mccormick_envelope_constraints,
                ) = relax_bilinear_expression(expr, variable_bounds)
                prog.AddDecisionVariables(new_vars)  # type: ignore
                prog.AddLinearConstraint(relaxed_expr == 0)
                for formula in mccormick_envelope_constraints:
                    prog.AddLinearConstraint(formula)
