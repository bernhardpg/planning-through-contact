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
    NpExpressionArray,
    NpFormulaArray,
    NpMonomialArray,
    NpPolynomialArray,
    NpVariableArray,
)


class BoundType(Enum):
    UPPER = 0
    LOWER = 1


# TODO there is definitely a much more efficient way of doing this
def _linear_binding_to_expressions(binding: Binding) -> NpExpressionArray:
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
    linear_bindings: List[Binding],
    bounding_box_expressions: NpExpressionArray,
    vars: NpVariableArray,
) -> npt.NDArray[np.float64]:
    if len(linear_bindings) > 0:
        binding_type = type(linear_bindings[0].evaluator())
        if not all([isinstance(b.evaluator(), binding_type) for b in linear_bindings]):
            raise ValueError(
                "When converting to homogenous form, all bindings must be either eq or ineqs."
            )

        linear_exprs = np.concatenate(
            [
                _linear_binding_to_expressions(b)
                for b in linear_bindings
                if b.variables().size
                > 0  # some bindings are empty? This fixes it. I will have to rewrite this whole thing either way
            ]
        )
    else:
        linear_exprs = []
    all_linear_exprs = np.concatenate([linear_exprs, bounding_box_expressions])

    A, b = sym.DecomposeAffineExpressions(all_linear_exprs.flatten(), vars)
    A_homogenous = np.hstack((b.reshape(-1, 1), A))
    return A_homogenous


# TODO temporary
class ConstraintType(Enum):
    EQ = 0
    INEQ = 1


def _generic_constraint_binding_to_polynomials(
    binding: Binding,
) -> List[Tuple[sym.Polynomial, ConstraintType]]:
    # TODO replace with QuadraticConstraint
    poly = sym.Polynomial(binding.evaluator().Eval(binding.variables())[0])
    b_upper = binding.evaluator().upper_bound()
    b_lower = binding.evaluator().lower_bound()

    polys = []
    for b_u, b_l in zip(b_upper, b_lower):
        if b_u == b_l:  # eq constraint
            polys.append((b_u - poly, ConstraintType.EQ))
        else:
            if not np.isinf(b_l):
                polys.append((poly - b_l, ConstraintType.INEQ))
            if not np.isinf(b_u):
                polys.append((b_u - poly, ConstraintType.INEQ))
    return polys


def _quadratic_cost_binding_to_homogenuous_form(
    binding: Binding, basis: NpMonomialArray, num_vars: int
) -> npt.NDArray[np.float64]:
    Q = binding.evaluator().Q()
    b = binding.evaluator().b()
    c = binding.evaluator().c()
    x = binding.variables()
    poly = sym.Polynomial(0.5 * x.T.dot(Q.dot(x)) + b.T.dot(x) + c)
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
    generic_constraints_as_polynomials = sum(
        [_generic_constraint_binding_to_polynomials(b) for b in generic_bindings], []
    )
    eq_polynomials = np.array(
        [p for p, t in generic_constraints_as_polynomials if t == ConstraintType.EQ]
    )
    ineq_polynomials = np.array(
        [p for p, t in generic_constraints_as_polynomials if t == ConstraintType.INEQ]
    )

    return (eq_polynomials, ineq_polynomials)


def _assert_max_degree(polys: NpPolynomialArray, degree: int) -> None:
    max_degree = max([p.TotalDegree() for p in polys])
    min_degree = min([p.TotalDegree() for p in polys])
    if max_degree > degree or min_degree < degree:
        raise ValueError(
            "Can only create SDP relaxation for (possibly non-convex) Quadratically Constrainted Quadratic Programs (QCQP)"
        )  # TODO for now we don't allow lower degree or higher degree


def _collect_bounding_box_constraints(
    bounding_box_bindings: List[Binding],
) -> Tuple[NpExpressionArray, NpExpressionArray]:
    bounding_box_constraints = []
    for b in bounding_box_bindings:
        x = b.variables()
        b_upper = b.evaluator().upper_bound()
        b_lower = b.evaluator().lower_bound()

        for x_i, b_u, b_l in zip(x, b_upper, b_lower):
            if b_u == b_l:  # eq constraint
                bounding_box_constraints.append((x_i - b_u, ConstraintType.EQ))
            else:
                if not np.isinf(b_u):
                    bounding_box_constraints.append((b_u - x_i, ConstraintType.INEQ))
                if not np.isinf(b_l):
                    bounding_box_constraints.append((x_i - b_l, ConstraintType.INEQ))

    bounding_box_eqs = np.array(
        [c for c, t in bounding_box_constraints if t == ConstraintType.EQ]
    )
    bounding_box_ineqs = np.array(
        [c for c, t in bounding_box_constraints if t == ConstraintType.INEQ]
    )

    return bounding_box_eqs, bounding_box_ineqs


def create_sdp_relaxation(
    prog: MathematicalProgram,
    use_linear_relaxation: bool = False,
    MULTIPLY_EQS_AND_INEQS: bool = False,
) -> Tuple[MathematicalProgram, NpVariableArray, NpMonomialArray]:
    DEGREE_QUADRATIC = 2  # We are only relaxing (non-convex) quadratic programs

    decision_vars = np.array(
        sorted(prog.decision_variables(), key=lambda x: x.get_id())
    )
    num_vars = (
        len(decision_vars) + 1
    )  # 1 will also be a decision variable in the relaxation

    basis = np.flip(sym.MonomialBasis(decision_vars, DEGREE_QUADRATIC))

    relaxed_prog = MathematicalProgram()
    X = relaxed_prog.NewSymmetricContinuousVariables(num_vars, "X")
    if use_linear_relaxation == False:
        relaxed_prog.AddPositiveSemidefiniteConstraint(X)

    relaxed_prog.AddLinearConstraint(X[0, 0] == 1)  # First variable is 1

    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

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
            c = np.sum(X * Q)
            relaxed_prog.AddCost(c)

    has_linear_eq_constraints = (
        len(prog.linear_equality_constraints()) > 0 or len(bounding_box_eqs) > 0
    )
    A_eq = None
    if has_linear_eq_constraints:
        A_eq = _linear_bindings_to_homogenuous_form(
            prog.linear_equality_constraints(), bounding_box_eqs, decision_vars
        )
        multiplied_constraints = eq(A_eq.dot(X).flatten(), 0)
        for c in multiplied_constraints:
            relaxed_prog.AddLinearConstraint(c)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = eq(A_eq.dot(X).dot(e_1), 0)
        for c in linear_constraints:
            relaxed_prog.AddLinearConstraint(c)

    has_linear_ineq_constraints = (
        len(prog.linear_constraints()) > 0 or len(bounding_box_ineqs) > 0
    )
    A_ineq = None
    if has_linear_ineq_constraints:
        A_ineq = _linear_bindings_to_homogenuous_form(
            prog.linear_constraints(), bounding_box_ineqs, decision_vars
        )
        multiplied_constraints = ge(A_ineq.dot(X).dot(A_ineq.T), 0)
        for c in multiplied_constraints.flatten():
            relaxed_prog.AddLinearConstraint(c)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = ge(A_ineq.dot(X).dot(e_1), 0)
        for c in linear_constraints:
            relaxed_prog.AddLinearConstraint(c)

    # Multiply equality and inequality constraints together.
    # In theory, this should help, but it doesn't seem to make a
    # difference. Commented out for now as it is very slow, and
    # I will not need it once I implement null-space projections
    if MULTIPLY_EQS_AND_INEQS:
        if A_ineq is not None and A_eq is not None:
            for a_i in A_ineq:
                for a_j in A_eq:
                    outer_product = np.outer(a_j, a_i)
                    c = np.sum(outer_product * X)
                    relaxed_prog.AddLinearConstraint(c == 0)

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
            constraints = eq(np.sum(X * Q), 0).flatten()

            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)
        Q_ineqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_ineq_constraints_as_polynomials
        ]
        for Q in Q_ineqs:
            constraints = ge(np.sum(X * Q), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)

    return relaxed_prog, X, basis
