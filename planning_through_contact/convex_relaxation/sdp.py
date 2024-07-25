from enum import Enum
from itertools import permutations
from logging import Logger
from pathlib import Path
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq, ge
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    LinearConstraint,
    LinearEqualityConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    PositiveSemidefiniteConstraint,
    QuadraticConstraint,
    SemidefiniteRelaxationOptions,
    SnoptSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.geometry.utilities import unit_vector
from planning_through_contact.tools.script_utils import make_default_logger
from planning_through_contact.tools.types import (
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


def _linear_bindings_to_affine_terms(
    linear_bindings: List[Binding],
    bounding_box_expressions: NpExpressionArray,
    vars: NpVariableArray,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns A and b satisfying
    Ax + b = vector of expressions (bindings)
    """
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
    return A, b.reshape((-1, 1))


def _affine_terms_to_homogenous_form(
    A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Assumes Ax + b = 0 should hold (note the sign difference from Ax = b)
    """
    A_homogenous = np.hstack((b.reshape(-1, 1), A))
    return A_homogenous


def linear_bindings_to_homogenuous_form(
    linear_bindings: List[Binding],
    bounding_box_expressions: NpExpressionArray,
    vars: NpVariableArray,
) -> npt.NDArray[np.float64]:
    """
    Returns the matrix that satisfies [b A][1 x]' = 0
    """
    A, b = _linear_bindings_to_affine_terms(
        linear_bindings, bounding_box_expressions, vars
    )
    A_homogenous = _affine_terms_to_homogenous_form(A, b)
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


def _quadratic_binding_to_homogenuous_form(
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
    # if max_degree > degree or min_degree < degree:
    #     breakpoint()
    #     raise ValueError(
    #         "Can only create SDP relaxation for (possibly non-convex) Quadratically Constrainted Quadratic Programs (QCQP)"
    #     )  # TODO for now we don't allow lower degree or higher degree


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
                # TODO: Remove this part
                raise ValueError("Bounding box equalities are not supported!")
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


# TODO: Move to some utils file
def get_nullspace_matrix(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    U, s, V_hermitian_transpose = np.linalg.svd(A)
    eps = 1e-6
    zero_idxs = np.where(np.abs(s) <= eps)[0].tolist()
    V = V_hermitian_transpose.T  # Real matrix, so conjugate transpose = transpose

    remaining_idxs = list(range(len(s), len(V)))

    V_zero = V[:, zero_idxs + remaining_idxs]

    nullspace_matrix = V_zero
    return nullspace_matrix


def find_solution(
    A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


def eliminate_equality_constraints(
    prog: MathematicalProgram, print_num_vars_eliminated: bool = False
) -> Tuple[
    MathematicalProgram, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
]:
    decision_vars = np.array(
        sorted(prog.decision_variables(), key=lambda x: x.get_id())
    )  # Not really necessary, they are sorted in this order in the prog
    old_dim = len(decision_vars)
    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    has_linear_eq_constraints = (
        len(prog.linear_equality_constraints()) > 0 or len(bounding_box_eqs) > 0
    )
    if not has_linear_eq_constraints:
        raise ValueError("There are no linear equality constraints to eliminate.")

    A_eq, b_eq = _linear_bindings_to_affine_terms(
        prog.linear_equality_constraints(), bounding_box_eqs, decision_vars
    )
    F = get_nullspace_matrix(A_eq)
    x_hat = find_solution(
        A_eq, -b_eq
    )  # TODO: Sign must be flipped because of the way _linear_bindings_to_affine_terms returns A and b

    new_dim = F.shape[1]
    new_prog = MathematicalProgram()
    new_decision_vars = new_prog.NewContinuousVariables(new_dim, "x")

    if print_num_vars_eliminated:
        # In SDP relaxation we will have:
        # (N^2 - N) / 2 + N variables
        # (all entries - diagonal entries)/2 (because X symmetric) + add back diagonal)
        calc_num_vars = lambda N: ((N + 1) ** 2 - (N + 1)) / 2 + (N + 1)
        num_vars_without_elimination = calc_num_vars(old_dim)
        num_vars_with_elimination = calc_num_vars(new_dim)
        diff = num_vars_without_elimination - num_vars_with_elimination
        print(
            f"Total number of vars in SDP relaxation of original problem: {num_vars_without_elimination}"
        )
        print(
            f"Total number of vars after elimination in SDP relaxation: {num_vars_with_elimination}"
        )
        print(f"Total number of variables eliminated: {diff}")

    has_linear_ineq_constraints = (
        len(prog.linear_constraints()) > 0 or len(bounding_box_ineqs) > 0
    )
    if has_linear_ineq_constraints:
        # Notice sign change on d
        B, d = _linear_bindings_to_affine_terms(
            prog.linear_constraints(), bounding_box_ineqs, decision_vars
        )  # B x >= -d becomes B F z >= -d - B x_hat
        new_prog.AddLinearConstraint(
            B.dot(F), -d - B.dot(x_hat), np.ones_like(d) * np.inf, new_decision_vars
        )

    has_generic_constaints = len(prog.generic_constraints()) > 0
    if has_generic_constaints:
        raise ValueError(
            "Cannot eliminate equality constraints for program with generic constraints."
        )

    if len(prog.quadratic_constraints()) > 0:
        for binding in prog.quadratic_constraints():
            e = binding.evaluator()
            Q = np.zeros((old_dim, old_dim))
            binding_Q = e.Q()
            var_idxs = prog.FindDecisionVariableIndices(binding.variables())

            for binding_i, prog_i in enumerate(var_idxs):
                for binding_j, prog_j in enumerate(var_idxs):
                    Q[prog_i, prog_j] = binding_Q[binding_i, binding_j]

            b = np.zeros(old_dim)
            b[var_idxs] = e.b()

            lb = e.lower_bound().item()
            ub = e.upper_bound().item()

            new_Q = F.T.dot(Q).dot(F)
            new_b = (x_hat.T.dot(Q).dot(F) + b.T.dot(F)).T

            new_lb = lb - (0.5 * x_hat.T.dot(Q).dot(x_hat) + b.T.dot(x_hat)).item()
            new_ub = ub - (0.5 * x_hat.T.dot(Q).dot(x_hat) + b.T.dot(x_hat)).item()

            constraint_empty = (
                np.allclose(new_Q, 0)
                and np.allclose(new_b, 0)
                and np.isclose(new_lb, 0)
                and np.isclose(new_ub, 0)
            )
            if constraint_empty:
                continue

            new_prog.AddQuadraticConstraint(
                new_Q, new_b, new_lb, new_ub, new_decision_vars
            )
            # Better way of doing this:
            # Q = binding.evaluator().Q()
            # b = binding.evaluator().b()
            # var_idxs = prog.FindDecisionVariableIndices(binding.variables())
            #
            # F_rows = F[var_idxs, :]
            # new_Q = F_rows.T.dot(Q).dot(F_rows)

    has_linear_costs = len(prog.linear_costs()) > 0
    if has_linear_costs:
        raise NotImplementedError("Linear costs not yet implemented!")

    if len(prog.quadratic_costs()) > 0:
        for binding in prog.quadratic_costs():
            e = binding.evaluator()
            Q = np.zeros((old_dim, old_dim))
            binding_Q = e.Q()
            var_idxs = prog.FindDecisionVariableIndices(binding.variables())

            for binding_i, prog_i in enumerate(var_idxs):
                for binding_j, prog_j in enumerate(var_idxs):
                    Q[prog_i, prog_j] = binding_Q[binding_i, binding_j]

            b = np.zeros(old_dim)
            b[var_idxs] = e.b()

            c = e.c()

            new_Q = F.T.dot(Q).dot(F)
            new_b = (x_hat.T.dot(Q).dot(F) + b.T.dot(F)).T

            new_c = c - (0.5 * x_hat.T.dot(Q).dot(x_hat) + b.T.dot(x_hat))

            new_prog.AddQuadraticCost(
                new_Q, new_b, new_c, new_decision_vars, e.is_convex()
            )

    def get_x_from_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = z.reshape((-1, 1))  # make sure z is (N, 1)
        x = F.dot(z) + x_hat

        remove_small_coeffs = lambda expr: (
            sym.Polynomial(expr).RemoveTermsWithSmallCoefficients(1e-5).ToExpression()
        )

        x = np.array([remove_small_coeffs(e) for e in x.flatten()])
        return x  # (n_vars, )

    return new_prog, get_x_from_z


# WARNING: This is no longer used. Instead, we use the builtin Drake function
# `MakeSemidefiniteRelaxation`.
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
            _quadratic_binding_to_homogenuous_form(c, basis, num_vars)
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
        A_eq = linear_bindings_to_homogenuous_form(
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
        A_ineq = linear_bindings_to_homogenuous_form(
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

    # TODO: clean up
    if len(prog.quadratic_constraints()) > 0:
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_constraint_bindings_to_polynomials(prog.quadratic_constraints())

        generic_constraints_as_polynomials = np.concatenate(
            (
                generic_eq_constraints_as_polynomials.flatten(),
                generic_ineq_constraints_as_polynomials.flatten(),
            )
        )
        # Don't add degree 0 polynomials (these should just be equal to 0)
        generic_constraints_as_polynomials = [
            c for c in generic_constraints_as_polynomials if c.TotalDegree() > 0
        ]
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


def get_X_from_semidefinite_relaxation(relaxation: MathematicalProgram):
    assert len(relaxation.positive_semidefinite_constraints()) == 1
    X = relaxation.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


def get_X_from_psd_constraint(binding) -> npt.NDArray:
    assert type(binding.evaluator()) == PositiveSemidefiniteConstraint
    X = binding.variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


def approximate_sdp_cones_with_linear_cones(prog: MathematicalProgram) -> None:
    """
    Iterates through all the PSD constraints on `prog` and replaces the PSD cone constraints
    X ≽ 0 with with the constraints X_ii ≥ 0 for i = 1, …, N (which must be true for all PSD matrices)
    """

    for psd_constraint in prog.positive_semidefinite_constraints():
        X = get_X_from_psd_constraint(psd_constraint)
        prog.RemoveConstraint(psd_constraint)  # type: ignore
        N = X.shape[0]
        for i in range(N):
            X_i = X[i, i]
            # if i == N - 1:
            #     continue  # skip 'one' == 1
            prog.AddLinearConstraint(X_i >= 0)


def add_trace_cost_on_psd_cones(prog: MathematicalProgram, eps: float = 1e-6) -> List:
    added_costs = []
    for psd_constraint in prog.positive_semidefinite_constraints():
        X = get_X_from_psd_constraint(psd_constraint)
        c = prog.AddLinearCost(eps * np.trace(X))
        added_costs.append(c)

    return added_costs


def plot_eigenvalues(
    X: npt.NDArray[np.float64], output_dir: Path | None = None
) -> None:
    N = X.shape[0]
    assert X.shape == (N, N)

    # Compute eigenvalues
    # (note: eigvalsh is more stable for real symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(X)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Plot the sorted eigenvalues
    indices = np.arange(len(sorted_eigenvalues))
    plt.bar(indices, sorted_eigenvalues)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of the Symmetric PSD Matrix (sorted)")

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / "eigenvalues.pdf")


def print_eigenvalues(
    X: npt.NDArray[np.float64], threshold: float = 1e-4, logger: Logger | None = None
) -> None:
    if logger is None:
        logger = make_default_logger()

    N = X.shape[0]
    assert X.shape == (N, N)

    # Compute eigenvalues
    # (note: eigvalsh is more stable for real symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(X)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Filter eigenvalues above the threshold
    filtered_eigenvalues = sorted_eigenvalues[sorted_eigenvalues > threshold]

    # Print the filtered eigenvalues
    logger.info("Eigenvalues above the threshold of {}: ".format(threshold))
    for i, eigenvalue in enumerate(filtered_eigenvalues, start=1):
        logger.info("Eigenvalue {}: {:.4f}".format(i, eigenvalue))


def solve_sdp_relaxation(
    qcqp: MathematicalProgram,
    print_solver_output: bool = False,
    plot_eigvals: bool = False,
    print_eigvals: bool = False,
    trace_cost: bool = False,
    print_time: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    if logger is None:
        logger = make_default_logger()

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    sdp_relaxation = MakeSemidefiniteRelaxation(qcqp, options)

    solver_options = SolverOptions()
    if print_solver_output:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    if trace_cost:
        add_trace_cost_on_psd_cones(sdp_relaxation)

    relaxed_result = Solve(sdp_relaxation, solver_options=solver_options)
    assert relaxed_result.is_success()

    X = get_X_from_semidefinite_relaxation(sdp_relaxation)
    X_val = relaxed_result.GetSolution(X)

    if plot_eigvals:
        plot_eigenvalues(X_val, output_dir)

    if print_eigvals:
        print_eigenvalues(X_val)

    if print_time:
        logger.info(
            f"Elapsed solver time: {relaxed_result.get_solver_details().optimizer_time:.2f} s"  # type: ignore
        )

    return X_val, relaxed_result.get_optimal_cost()


def get_gaussian_from_sdp_relaxation_solution(
    Y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Follows the sceme from Section 3.3 in
    J. Park and S. P. Boyd, “General Heuristics for
    Nonconvex Quadratically Constrained Quadratic Programming,” 2017

    Y = [X  x
         xᵀ 1]

    @return (mean vector, covariance matrix)
    """
    N = Y.shape[0]
    assert Y.shape == (N, N)
    assert Y.dtype == float

    assert np.isclose(Y[-1, -1], 1)
    X = Y[:-1, :-1]
    x = Y[:-1, -1]

    μ = x
    Σ = X - np.outer(x, x)

    return μ, Σ


def to_symmetric_matrix_from_lower_triangular_columns(
    vec: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    # Determine the size of the matrix
    n = int(np.sqrt(2 * len(vec) + 0.25) - 0.5)

    if len(vec) != (n * (n + 1)) // 2:
        raise ValueError(
            "The length of the vector is not appropriate for forming a symmetric matrix."
        )

    # Create an empty symmetric matrix
    symm_matrix = np.zeros((n, n), dtype=vec.dtype)

    # Fill the lower triangle and diagonal
    index = 0
    for j in range(n):
        for i in range(j, n):
            symm_matrix[i, j] = vec[index]
            if i != j:
                symm_matrix[j, i] = vec[index]
            index += 1

    return symm_matrix
