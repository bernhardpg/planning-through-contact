from enum import Enum
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq, ge
from pydrake.solvers import (
    Binding,
    LinearConstraint,
    LinearEqualityConstraint,
    MathematicalProgram,
    Solve,
)

from geometry.utilities import unit_vector
from tools.types import (
    NpFormulaArray,
    NpMonomialArray,
    NpPolynomialArray,
    NpVariableArray,
)


class BoundType(Enum):
    UPPER = 0
    LOWER = 1


# TODO there is definitely a much more efficient way of doing this
def _linear_binding_to_formulas(binding: Binding) -> NpFormulaArray:
    """
    Takes in a binding and returns a polynomial p that should satisfy\
    p(x) = 0 for equality constraints, p(x) >= for inequality constraints
    
    """
    # NOTE: I cannot use binding.evaluator().Eval(binding.variables())
    # here, because it ignores the constant term for linear constraints! Is this a bug?
    A = binding.evaluator().GetDenseA()
    x = binding.variables()
    A_x = A.dot(x)
    b_upper = binding.evaluator().upper_bound()
    b_lower = binding.evaluator().lower_bound()

    formulas = []
    for a_i_x, b_i_upper, b_i_lower in zip(A_x, b_upper, b_lower):
        if b_i_upper == b_i_lower:  # eq constraint
            formulas.append(b_i_upper - a_i_x)
        elif not np.isinf(b_i_upper):
            formulas.append(b_i_upper - a_i_x)
        elif not np.isinf(b_i_lower):
            formulas.append(a_i_x - b_i_lower)

    return np.array(formulas)


def _linear_bindings_to_homogenuous_form(
    linear_bindings: List[Binding], vars: NpVariableArray
) -> npt.NDArray[np.float64]:
    binding_type = type(linear_bindings[0].evaluator())
    if not all([isinstance(b.evaluator(), binding_type) for b in linear_bindings]):
        raise ValueError("All bindings must be either ineqs or eqs.")

    linear_formulas = np.concatenate(
        [_linear_binding_to_formulas(b) for b in linear_bindings]  # type: ignore
    )

    A, b = sym.DecomposeAffineExpressions(linear_formulas.flatten(), vars)
    A_homogenous = np.hstack((b.reshape(-1, 1), A))
    return A_homogenous


def _generic_constraint_binding_to_polynomial(
    binding: Binding, bound: BoundType
) -> sym.Polynomial:
    poly = sym.Polynomial(binding.evaluator().Eval(binding.variables())[0])
    if bound == BoundType.UPPER:
        b = binding.evaluator().upper_bound().item()
        return b - poly
    elif bound == BoundType.LOWER:
        b = binding.evaluator().lower_bound().item()
        return poly - b
    else:
        raise ValueError("Unsupported bound type")


def _quadratic_cost_binding_to_homogenuous_form(
    binding: Binding, basis: NpMonomialArray, num_vars: int
) -> npt.NDArray[np.float64]:
    Q = binding.evaluator().Q()
    b = binding.evaluator().b()
    c = binding.evaluator().c()
    x = binding.variables()
    poly = sym.Polynomial(x.T.dot(Q.dot(x)) + b.T.dot(x) + c)
    Q_hom = _quadratic_polynomial_to_homoenuous_form(poly, basis, num_vars)
    return Q_hom


def _get_monomial_coeffs(
    poly: sym.Polynomial, basis: NpMonomialArray
) -> npt.NDArray[np.float64]:
    coeff_map = poly.monomial_to_coefficient_map()
    coeffs = np.array([coeff_map.get(m, sym.Expression(0)).Evaluate() for m in basis])
    return coeffs


def _construct_symmetric_matrix_from_triang(
    triang_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return triang_matrix + triang_matrix.T


def _quadratic_polynomial_to_homoenuous_form(
    poly: sym.Polynomial, basis: NpMonomialArray, num_vars: int
) -> npt.NDArray[np.float64]:
    coeffs = _get_monomial_coeffs(poly, basis)
    upper_triangular = np.zeros((num_vars, num_vars))
    upper_triangular[np.triu_indices(num_vars)] = coeffs
    Q = _construct_symmetric_matrix_from_triang(upper_triangular)
    return Q * 0.5


def _generic_constraint_bindings_to_polynomials(
    generic_bindings: List[Binding],
) -> Tuple[NpPolynomialArray, NpPolynomialArray]:
    generic_eq_bindings = [
        b for b in generic_bindings if _equal_lower_and_upper_bounds(b)
    ]
    generic_eq_constraints_as_polynomials = np.array(
        [
            _generic_constraint_binding_to_polynomial(
                b, BoundType.UPPER
            )  # Bounds are equal for eq constraints
            for b in generic_eq_bindings
        ]
    )

    generic_ineq_bindings = [
        b for b in generic_bindings if not _equal_lower_and_upper_bounds(b)
    ]
    lower_bounded_bindings = [
        b for b in generic_ineq_bindings if not np.isinf(b.evaluator().lower_bound())
    ]
    polys_lower = np.array(
        [
            _generic_constraint_binding_to_polynomial(b, BoundType.UPPER)
            for b in lower_bounded_bindings
        ]
    ).flatten()
    upper_bounded_bindings = [
        b for b in generic_ineq_bindings if not np.isinf(b.evaluator().upper_bound())
    ]
    polys_upper = np.array(
        [
            _generic_constraint_binding_to_polynomial(b, BoundType.UPPER)
            for b in upper_bounded_bindings
        ]
    ).flatten()
    generic_ineq_constraints_as_polynomials = np.concatenate([polys_lower, polys_upper])

    return (
        generic_eq_constraints_as_polynomials,
        generic_ineq_constraints_as_polynomials,
    )


def _equal_lower_and_upper_bounds(b: Binding) -> bool:
    return b.evaluator().lower_bound() == b.evaluator().upper_bound()


def _assert_max_degree(polys: NpPolynomialArray, degree: int) -> None:
    max_degree = max([p.TotalDegree() for p in polys])
    min_degree = min([p.TotalDegree() for p in polys])
    if max_degree > degree or min_degree < degree:
        raise ValueError(
            "Can only create SDP relaxation for (possibly non-convex) Quadratically Constrainted Quadratic Programs (QCQP)"
        )  # TODO for now we don't allow lower degree or higher degree


def create_sdp_relaxation(
    prog: MathematicalProgram,
) -> Tuple[MathematicalProgram, NpVariableArray]:
    DEGREE_QUADRATIC = 2  # We are only relaxing (non-convex) quadratic programs

    decision_vars = prog.decision_variables()
    num_vars = (
        len(decision_vars) + 1
    )  # 1 will also be a decision variable in the relaxation

    basis = np.flip(sym.MonomialBasis(decision_vars, DEGREE_QUADRATIC))
    relaxed_prog = MathematicalProgram()
    X = relaxed_prog.NewSymmetricContinuousVariables(num_vars, "X")
    relaxed_prog.AddPositiveSemidefiniteConstraint(X)

    relaxed_prog.AddLinearConstraint(X[0, 0] == 1)  # First variable is 1

    if len(prog.bounding_box_constraints()) > 0:
        raise NotImplementedError("Bounding box constraints are not implemented!")

    has_linear_costs = len(prog.linear_costs()) > 0
    if has_linear_costs:
        raise NotImplementedError("Linear costs not yet implemented!")

    has_quadratic_costs = len(prog.quadratic_costs()) > 0
    if has_quadratic_costs:
        quadratic_costs = prog.quadratic_costs()
        Q_cost = [
            _quadratic_cost_binding_to_homogenuous_form(c, basis, num_vars)
            for c in quadratic_costs
        ]
        for Q in Q_cost:
            relaxed_prog.AddCost(np.trace(Q.dot(X)))

    has_linear_eq_constraints = len(prog.linear_equality_constraints()) > 0
    if has_linear_eq_constraints:
        A_eq = _linear_bindings_to_homogenuous_form(
            prog.linear_equality_constraints(), decision_vars
        )
        multiplied_constraints = eq(A_eq.dot(X).dot(A_eq.T), 0)
        relaxed_prog.AddLinearConstraint(multiplied_constraints)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = eq(A_eq.dot(X).dot(e_1), 0)
        relaxed_prog.AddLinearConstraint(linear_constraints)

    has_linear_ineq_constraints = len(prog.linear_constraints()) > 0
    if has_linear_ineq_constraints:
        A_ineq = _linear_bindings_to_homogenuous_form(
            prog.linear_constraints(), decision_vars
        )
        multiplied_constraints = ge(A_ineq.dot(X).dot(A_ineq.T), 0)
        relaxed_prog.AddLinearConstraint(multiplied_constraints)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = ge(A_ineq.dot(X).dot(e_1), 0)
        relaxed_prog.AddLinearConstraint(linear_constraints)

    has_generic_constaints = len(prog.generic_constraints()) > 0
    # TODO: I can use Hongkai's PR once that is merged
    if has_generic_constaints:
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_constraint_bindings_to_polynomials(prog.generic_constraints())

        generic_constraints_as_polynomials = np.concatenate(
            (
                generic_eq_constraints_as_polynomials.flatten(),
                generic_ineq_constraints_as_polynomials.flatten(),
            )
        )
        _assert_max_degree(generic_constraints_as_polynomials, DEGREE_QUADRATIC)

        Q_eqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_eq_constraints_as_polynomials
        ]
        for Q in Q_eqs:
            constraints = eq(np.trace(X.dot(Q)), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)

        Q_ineqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_ineq_constraints_as_polynomials
        ]
        for Q in Q_eqs:
            constraints = ge(np.trace(X.dot(Q)), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)

    return relaxed_prog, X
