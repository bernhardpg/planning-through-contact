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

from tools.types import NpMonomialArray, NpPolynomialArray, NpVariableArray


class BoundType(Enum):
    UPPER = 0
    LOWER = 1


# TODO there is definitely a much more efficient way of doing this
def _linear_binding_to_formula(
    binding: Binding, bound: BoundType = BoundType.UPPER
) -> sym.Formula:
    """
    Takes in a binding and returns a polynomial p that should satisfy\
    p(x) = 0 for equality constraints, p(x) >= for inequality constraints
    
    """
    # NOTE: I cannot use binding.evaluator().Eval(binding.variables())
    # here, because it ignores the constant term for linear constraints! Is this a bug?
    A = binding.evaluator().GetDenseA()
    x = binding.variables()
    if bound == BoundType.UPPER:
        b = binding.evaluator().upper_bound()
        return b - A.dot(x)
    elif bound == BoundType.LOWER:
        b = binding.evaluator().lower_bound()
        return A.dot(x) - b
    else:
        raise ValueError("Boundtype must be either lower or upper")


def _linear_bindings_to_homogenuous_form(
    linear_bindings: List[Binding], vars: NpVariableArray
) -> npt.NDArray[np.float64]:
    binding_type = type(linear_bindings[0].evaluator())
    if not all([isinstance(b.evaluator(), binding_type) for b in linear_bindings]):
        raise ValueError("All bindings must be either ineqs or eqs.")

    if binding_type == LinearEqualityConstraint:
        # For eq constraints, the bounds are the same, so WLOG we use UPPER
        linear_formulas = np.array(
            [_linear_binding_to_formula(b, BoundType.UPPER) for b in linear_bindings]
        )
    elif binding_type == LinearConstraint:  # ineqs
        upper_bounded_bindings = [
            b for b in linear_bindings if not np.isinf(b.evaluator().upper_bound())
        ]
        linear_formulas_upper_bounded = np.array(
            [
                _linear_binding_to_formula(b, BoundType.UPPER)
                for b in upper_bounded_bindings
            ]
        ).flatten()
        lower_bounded_bindings = [
            b for b in linear_bindings if not np.isinf(b.evaluator().lower_bound())
        ]
        linear_formulas_lower_bounded = np.array(
            [
                _linear_binding_to_formula(b, BoundType.LOWER)
                for b in lower_bounded_bindings
            ]
        ).flatten()
        linear_formulas = np.concatenate(
            [linear_formulas_lower_bounded, linear_formulas_upper_bounded]
        )
    else:
        raise ValueError(f"Invalid linear binding type: {binding_type}")

    A, b = sym.DecomposeAffineExpressions(linear_formulas, vars)
    A_homogenous = np.hstack((b.reshape(-1, 1), A))
    return A_homogenous


def _generic_binding_to_polynomial(
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


def _generic_bindings_to_polynomials(
    generic_bindings: List[Binding],
) -> Tuple[NpPolynomialArray, NpPolynomialArray]:
    generic_eq_bindings = [
        b for b in generic_bindings if _equal_lower_and_upper_bounds(b)
    ]
    generic_eq_constraints_as_polynomials = np.array(
        [
            _generic_binding_to_polynomial(
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
            _generic_binding_to_polynomial(b, BoundType.UPPER)
            for b in lower_bounded_bindings
        ]
    ).flatten()
    upper_bounded_bindings = [
        b for b in generic_ineq_bindings if not np.isinf(b.evaluator().upper_bound())
    ]
    polys_upper = np.array(
        [
            _generic_binding_to_polynomial(b, BoundType.UPPER)
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

    has_linear_eq_constraints = len(prog.linear_equality_constraints()) > 0
    if has_linear_eq_constraints:
        A_eq = _linear_bindings_to_homogenuous_form(
            prog.linear_equality_constraints(), decision_vars
        )
        relaxed_prog.AddLinearConstraint(eq(A_eq.dot(X).dot(A_eq.T), 0))

    has_linear_ineq_constraints = len(prog.linear_constraints()) > 0
    if has_linear_ineq_constraints:
        A_ineq = _linear_bindings_to_homogenuous_form(
            prog.linear_constraints(), decision_vars
        )
        relaxed_prog.AddLinearConstraint(ge(A_ineq.dot(X).dot(A_ineq.T), 0))

    has_generic_constaints = len(prog.generic_constraints()) > 0
    # TODO: I can use Hongkai's PR once that is merged
    if has_generic_constaints:
        # TODO differentiate between eq and ineq
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_bindings_to_polynomials(prog.generic_constraints())

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


def add_constraint(self, formula: sym.Formula, bp: bool = False) -> None:
    kind = formula.get_kind()
    lhs, rhs = formula.Unapply()[1]  # type: ignore
    poly = sym.Polynomial(lhs - rhs)

    if poly.TotalDegree() > self.degree:
        raise ValueError(
            f"Constraint degree is {poly.TotalDegree()},"
            "program degree is {self.degree}"
        )

    Q = self._construct_quadratic_constraint(poly, self.mon_basis, self.n)
    constraint_lhs = np.trace(self.X @ Q)
    if bp:
        breakpoint()
    if kind == sym.FormulaKind.Eq:
        self.prog.AddConstraint(constraint_lhs == 0)
    elif kind == sym.FormulaKind.Geq:
        self.prog.AddConstraint(constraint_lhs >= 0)
    elif kind == sym.FormulaKind.Leq:
        self.prog.AddConstraint(constraint_lhs <= 0)
    else:
        raise NotImplementedError(f"Support for formula type {kind} not implemented")


class SdpRelaxation:
    def __init__(self, vars: npt.NDArray[sym.Variable]):
        self.n = vars.shape[0] + 1  # 1 is also a monomial
        self.order = 1  # For now, we just do the first order of the hierarchy

        # [1, x, x ** 2, ... ]
        self.mon_basis = np.flip(sym.MonomialBasis(vars, self.degree))

        self.prog = MathematicalProgram()
        self.X = self.prog.NewSymmetricContinuousVariables(self.n, "X")
        self.prog.AddConstraint(
            self.X[0, 0] == 1
        )  # First variable is not really a variable
        self.prog.AddPositiveSemidefiniteConstraint(self.X)

    @property
    def degree(self) -> int:
        return self.order + 1

    def add_constraint(self, formula: sym.Formula, bp: bool = False) -> None:
        kind = formula.get_kind()
        lhs, rhs = formula.Unapply()[1]  # type: ignore
        poly = sym.Polynomial(lhs - rhs)

        if poly.TotalDegree() > self.degree:
            raise ValueError(
                f"Constraint degree is {poly.TotalDegree()},"
                "program degree is {self.degree}"
            )

        Q = self._construct_quadratic_constraint(poly, self.mon_basis, self.n)
        constraint_lhs = np.trace(self.X @ Q)
        if bp:
            breakpoint()
        if kind == sym.FormulaKind.Eq:
            self.prog.AddConstraint(constraint_lhs == 0)
        elif kind == sym.FormulaKind.Geq:
            self.prog.AddConstraint(constraint_lhs >= 0)
        elif kind == sym.FormulaKind.Leq:
            self.prog.AddConstraint(constraint_lhs <= 0)
        else:
            raise NotImplementedError(
                f"Support for formula type {kind} not implemented"
            )

    def get_solution(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        result = Solve(self.prog)
        assert result.is_success()
        X_result = result.GetSolution(self.X)
        svd_solution = self._get_sol_from_svd(X_result)
        variable_values = svd_solution[1:]  # first value is 1
        return variable_values, X_result

    def _get_sol_from_svd(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        eigenvals, eigenvecs = np.linalg.eig(X)
        idx_highest_eigval = np.argmax(eigenvals)
        solution_nonnormalized = eigenvecs[:, idx_highest_eigval]
        solution = solution_nonnormalized / solution_nonnormalized[0]
        return solution

    def _get_monomial_coeffs(
        self, poly: sym.Polynomial, basis: npt.NDArray[sym.Monomial]
    ):
        coeff_map = poly.monomial_to_coefficient_map()
        breakpoint()
        coeffs = np.array(
            [coeff_map.get(m, sym.Expression(0)).Evaluate() for m in basis]
        )
        return coeffs

    def _construct_symmetric_matrix_from_triang(
        self,
        triang_matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return triang_matrix + triang_matrix.T

    def _construct_quadratic_constraint(
        self, poly: sym.Polynomial, basis: npt.NDArray[sym.Monomial], n: int
    ) -> npt.NDArray[np.float64]:
        coeffs = self._get_monomial_coeffs(poly, basis)
        upper_triangular = np.zeros((n, n))
        upper_triangular[np.triu_indices(n)] = coeffs
        Q = self._construct_symmetric_matrix_from_triang(upper_triangular)
        return Q * 0.5
