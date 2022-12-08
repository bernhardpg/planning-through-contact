from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


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

    def add_constraint(self, formula: sym.Formula) -> None:
        kind = formula.get_kind()
        lhs, rhs = formula.Unapply()[1]  # type: ignore
        poly = sym.Polynomial(lhs - rhs)

        if poly.TotalDegree() > self.degree:
            raise ValueError(
                f"Constraint degree is {poly.TotalDegree()}, program degree is {self.degree}"
            )

        Q = self._construct_quadratic_constraint(poly, self.mon_basis, self.n)
        constraint_lhs = np.trace(self.X @ Q)
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

    def get_solution(self) -> npt.NDArray[np.float64]:
        result = Solve(self.prog)
        assert result.is_success()
        X_result = result.GetSolution(self.X)
        svd_solution = self._get_sol_from_svd(X_result)
        variable_values = svd_solution[1:]  # first value is 1
        return variable_values

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


def test():
    x = sym.Variable("x")
    y = sym.Variable("y")
    variables = np.array([x, y])

    prog = SdpRelaxation(variables)
    prog.add_constraint(x**2 + y**2 == 1)
    prog.add_constraint(x >= 0.5)
    prog.add_constraint(y >= 0.5)

    result = Solve(prog.prog)
    assert result.is_success()

    X_result = result.GetSolution(prog.X)
    return


@dataclass
class Box2d:
    width: float = 3
    height: float = 2
    mass: float = 1

    # p1 -- p2
    # |     |
    # p4 -- p3

    @property
    def p1(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [self.height / 2]])

    @property
    def p2(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [self.height / 2]])

    @property
    def p3(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [-self.height / 2]])

    @property
    def p4(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [-self.height / 2]])

    @property
    def corners(self) -> List[npt.NDArray[np.float64]]:
        return [self.p1, self.p2, self.p3, self.p4]

    def _construct_plane_from_points(
        self, p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        diff = p2 - p1
        normal_vec = np.array([-diff[1], diff[0]]).reshape((-1, 1))
        a = normal_vec / np.linalg.norm(normal_vec)
        b = a.T.dot(p1)
        return (a, b)

    # p1 - a1 - p2
    # |          |
    # a4         a2
    # |          |
    # p4 --a3--- p3

    @property
    def a1(self) -> npt.NDArray[np.float64]:
        return self._construct_plane_from_points(self.p1, self.p2)

    @property
    def a2(self) -> npt.NDArray[np.float64]:
        return self._construct_plane_from_points(self.p2, self.p3)

    @property
    def a3(self) -> npt.NDArray[np.float64]:
        return self._construct_plane_from_points(self.p3, self.p4)

    @property
    def a4(self) -> npt.NDArray[np.float64]:
        return self._construct_plane_from_points(self.p4, self.p1)


def sdp_relaxation():
    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    box = Box2d(BOX_WIDTH, BOX_HEIGHT, BOX_MASS)

    c_th = sym.Variable("c_th")
    s_th = sym.Variable("s_th")
    p_WB_x = sym.Variable("p_WB_x")
    p_WB_y = sym.Variable("p_WB_y")
    variables = np.array([c_th, s_th, p_WB_x, p_WB_y])

    prog = SdpRelaxation(variables)

    # SO(2) constraints
    so_2_constraint = c_th**2 + s_th**2 == 1
    prog.add_constraint(so_2_constraint)

    # Add Non-penetration
    p_WB = np.array([p_WB_x, p_WB_y]).reshape((-1, 1))
    R_WB = np.array([[c_th, -s_th], [s_th, c_th]])

    # Table hyperplane
    a = np.array([0, 1]).reshape((-1, 1))
    b = -2

    for p_Bmi in box.corners:
        p_Wmi = p_WB + R_WB.dot(p_Bmi)
        non_pen_constraint = (a.T.dot(p_Wmi))[0, 0] >= b
        prog.add_constraint(non_pen_constraint)

    for c in eq(p_WB, 0):
        prog.add_constraint(c[0])
    solution = prog.get_solution()
    breakpoint()

    return


def simple_rotations_test(use_sdp_relaxation: bool = True):
    N_DIMS = 2
    NUM_CTRL_POINTS = 3

    prog = MathematicalProgram()

    BOX_WIDTH = 3
    BOX_HEIGHT = 2
    BOX_MASS = 1
    GRAV_ACC = 9.81

    FINGER_POS = np.array([[-BOX_WIDTH / 2], [BOX_HEIGHT / 2]])
    GROUND_CONTACT_POS = np.array([[BOX_WIDTH / 2], [-BOX_HEIGHT / 2]])

    f_gravity = np.array([[0], [-BOX_MASS * GRAV_ACC]])
    f_finger = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_finger")
    f_contact = prog.NewContinuousVariables(N_DIMS, NUM_CTRL_POINTS, "f_contact")
    cos_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "cos_th")
    sin_th = prog.NewContinuousVariables(1, NUM_CTRL_POINTS, "sin_th")

    # Force and moment balance
    R_f_gravity = np.concatenate(
        (
            cos_th * f_gravity[0] - sin_th * f_gravity[1],
            sin_th * f_gravity[0] + cos_th * f_gravity[1],
        )
    )
    force_balance = eq(f_finger + f_contact + R_f_gravity, 0)
    moment_balance = eq(
        cross(FINGER_POS, f_finger) + cross(GROUND_CONTACT_POS, f_contact), 0
    )

    prog.AddLinearConstraint(force_balance)
    prog.AddLinearConstraint(moment_balance)

    # Force minimization cost
    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_finger.flatten(),
    )
    prog.AddQuadraticCost(
        np.eye(N_DIMS * NUM_CTRL_POINTS),
        np.zeros((N_DIMS * NUM_CTRL_POINTS, 1)),
        f_contact.flatten(),
    )

    # Path length minimization cost
    cos_cost = np.sum(np.diff(cos_th) ** 2)
    sin_cost = np.sum(np.diff(sin_th) ** 2)
    prog.AddQuadraticCost(cos_cost + sin_cost)

    # SO(2) constraint
    if use_sdp_relaxation:
        aux_vars = prog.NewContinuousVariables(3, NUM_CTRL_POINTS, "X")
        Xs = [np.array([[z[0], z[1]], [z[1], z[2]]]) for z in aux_vars.T]
        xs = [np.vstack([c, s]) for c, s in zip(cos_th.T, sin_th.T)]
        Ms = [np.block([[1, x.T], [x, X]]) for X, x in zip(Xs, xs)]
        for X, M in zip(Xs, Ms):
            prog.AddLinearConstraint(X[0, 0] + X[1, 1] - 1 == 0)
            prog.AddPositiveSemidefiniteConstraint(M)
    else:
        cos_th_sq = (cos_th * cos_th)[0]
        sin_th_sq = (sin_th * sin_th)[0]
        prog.AddLorentzConeConstraint(1, cos_th_sq[0] + sin_th_sq[0])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])
        prog.AddLorentzConeConstraint(1, cos_th_sq[1] + sin_th_sq[1])

    # Initial and final condition
    th_initial = 0
    prog.AddLinearConstraint(cos_th[0, 0] == np.cos(th_initial))
    prog.AddLinearConstraint(sin_th[0, 0] == np.sin(th_initial))

    th_final = 0.5
    prog.AddLinearConstraint(cos_th[0, -1] == np.cos(th_final))
    prog.AddLinearConstraint(sin_th[0, -1] == np.sin(th_final))

    # Solve
    result = Solve(prog)
    assert result.is_success()

    # Rounding and projection onto SO(2)
    to_float = np.vectorize(lambda x: x.Evaluate())
    Ms_result = [to_float(result.GetSolution(M)) for M in Ms]
    ws, vs = zip(*[np.linalg.eig(M) for M in Ms_result])
    idx_highest_eigval = [np.argmax(w) for w in ws]
    vks = [v[:, idx] / v[0, idx] for idx, v in zip(idx_highest_eigval, vs)]

    vk_ps = [vk[[1, 2]] for vk in vks]
    xvs = [np.array([[vk_p[0], -vk_p[1]], [vk_p[1], vk_p[0]]]) for vk_p in vk_ps]

    Us, Ss, Vs = zip(*[np.linalg.svd(xv) for xv in xvs])
    R_hats = [
        U.dot(np.diag([1, np.linalg.det(U) * np.linalg.det(V)])).dot(V.T)
        for U, V in zip(Us, Vs)
    ]

    results_finger = result.GetSolution(f_finger)
    results_contact = result.GetSolution(f_contact)
    results_cos_th = np.array([R[0, 0] for R in R_hats])
    results_sin_th = np.array([R[1, 0] for R in R_hats])
    results_th = np.array(
        [
            [np.arccos(cos_th) for cos_th in results_cos_th],
            [np.arcsin(sin_th) for sin_th in results_sin_th],
        ]
    )

    # Plot
    fig, axs = plt.subplots(7, 1)
    axs[0].set_title("Finger force x")
    axs[1].set_title("Finger force y")
    axs[0].plot(results_finger[0, :])
    axs[1].plot(results_finger[1, :])

    axs[2].set_title("Contact force x")
    axs[3].set_title("Contact force y")
    axs[2].plot(results_contact[0, :])
    axs[3].plot(results_contact[1, :])

    axs[4].set_title("cos(th)")
    axs[5].set_title("sin(th)")
    axs[4].plot(results_cos_th)
    axs[5].plot(results_sin_th)

    axs[6].plot(results_th.T)
    axs[6].set_title("theta")

    plt.tight_layout()
    plt.show()
