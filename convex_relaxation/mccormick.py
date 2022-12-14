from typing import Dict, List, Tuple

import pydrake.symbolic as sym  # type: ignore


def _create_aux_variable(m: sym.Monomial) -> sym.Variable:
    name = "".join(str(m).split(" "))
    return sym.Variable(name)


def _create_mccormick_envelopes(u, v, w, variable_bounds) -> List[sym.Formula]:
    BIG_NUM = 999
    u_L, u_U = variable_bounds.get(u.get_name().split("(")[0], (-BIG_NUM, BIG_NUM))
    v_L, v_U = variable_bounds.get(v.get_name().split("(")[0], (-BIG_NUM, BIG_NUM))
    w_L = u_L * v_L
    w_U = u_U * v_U

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
