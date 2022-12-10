from typing import Tuple

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.solvers import MathematicalProgram, Solve


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
