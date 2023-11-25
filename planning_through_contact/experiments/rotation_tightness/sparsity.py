import numpy as np
import numpy.typing as npt
from pydrake.math import eq, le
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
    SolverOptions,
)
from pydrake.symbolic import (
    DecomposeAffineExpression,
    DecomposeAffineExpressions,
    Expression,
    Variable,
    Variables,
)

from planning_through_contact.convex_relaxation.sdp import (
    create_sdp_relaxation,
    linear_bindings_to_homogenuous_form,
)
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs


class BandSparseSemidefiniteRelaxation:
    def __init__(self, num_groups: int) -> None:
        self.prog = MathematicalProgram()
        self.num_groups = num_groups

        self.groups = {idx: [] for idx in range(num_groups)}
        self.linear_inequality_constraints = {idx: [] for idx in range(num_groups)}
        self.linear_equality_constraints = {idx: [] for idx in range(num_groups)}

        # Quadratic constraints and costs is what couples the groups
        self.quadratic_constraints = {
            (i, j): [] for i, j in zip(range(num_groups - 1), range(1, num_groups))
        }
        self.quadratic_costs = {
            (i, j): [] for i, j in zip(range(num_groups - 1), range(1, num_groups))
        }

        # Include quadratic terms
        for i in range(num_groups):
            self.quadratic_constraints[i, i] = []
            self.quadratic_costs[i, i] = []

    def _find_numbered_idx(self, idx: int) -> int:
        assert idx < self.num_groups
        if idx == -1:
            return self.num_groups - 1
        else:
            return idx

    def new_variables(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        vars = self.prog.NewContinuousVariables(*args)
        self.groups[group_idx].append(vars)

        return vars

    def add_linear_inequality_constraint(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddLinearConstraint(*args)
        self.linear_inequality_constraints[group_idx].append(constraint)

        return constraint

    def add_linear_equality_constraint(self, group_idx: int, *args):
        group_idx = self._find_numbered_idx(group_idx)
        constraint = self.prog.AddLinearEqualityConstraint(*args)
        self.linear_equality_constraints[group_idx].append(constraint)

        return constraint

    def add_quadratic_constraint(self, group_idx_1: int, group_idx_2: int, *args):
        group_idx_1 = self._find_numbered_idx(group_idx_1)
        group_idx_2 = self._find_numbered_idx(group_idx_2)

        constraint = self.prog.AddQuadraticConstraint(*args)
        self.quadratic_constraints[(group_idx_1, group_idx_2)].append(constraint)

        return constraint

    def add_quadratic_cost(self, group_idx_1: int, group_idx_2: int, *args):
        group_idx_1 = self._find_numbered_idx(group_idx_1)
        group_idx_2 = self._find_numbered_idx(group_idx_2)

        cost = self.prog.AddQuadraticCost(*args)
        self.quadratic_costs[(group_idx_1, group_idx_2)].append(cost)

        return cost

    def make_relaxation(self) -> MathematicalProgram:
        relaxed_prog = MathematicalProgram()

        # First gather variables
        self.xs = {
            idx: np.concatenate(self.groups[idx]).reshape((-1, 1))
            for idx in range(self.num_groups)
        }

        # Form smaller PSD cones
        self.Xs = {}
        self.Ys = {}
        for idx in range(self.num_groups - 1):
            x = self.xs[idx]
            x_next = self.xs[idx + 1]

            relaxed_prog.AddDecisionVariables(x)
            relaxed_prog.AddDecisionVariables(x_next)
            num_vars = len(x) + len(x_next)
            X = relaxed_prog.NewSymmetricContinuousVariables(num_vars, f"X_{idx}")
            self.Xs[idx] = X

            # Y = [1; x; x_next] [1; x'; x_next']'
            y = np.concatenate((x, x_next))
            # fmt: off
            Y = np.block([[1, y.T],
                          [y, X]])
            # fmt: on
            relaxed_prog.AddPositiveSemidefiniteConstraint(Y)
            self.Ys[idx] = Y

        # Cones must intersect
        for idx in range(self.num_groups - 2):
            X = self.Xs[idx]
            X_next = self.Xs[idx + 1]

            # X = [x; y][x' y']
            # X = [xx' xy'
            #      yx' yy']
            # X_next = [yy' zy'
            #           zy' zz']
            # We enforce yy' == yy'
            y_len = len(self.xs[idx + 1])
            const = X[-y_len:, -y_len:] - X_next[:y_len, :y_len]
            for c in const.flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

        # Add all linear constraints directly
        for idx in range(self.num_groups):
            for const in self.linear_inequality_constraints[idx]:
                relaxed_prog.AddConstraint(const.evaluator(), const.variables())
            for const in self.linear_equality_constraints[idx]:
                # TODO(bernhardpg): For some reason, the drake API for equality constraints doesn't support bindings
                relaxed_prog.AddConstraint(const.evaluator(), const.variables())

        # Add constraints implied by linear equality constraints
        for idx in range(self.num_groups - 1):
            x = self.xs[idx]
            y = self.xs[idx + 1]
            eqs_i = self.linear_equality_constraints[idx]
            eqs_j = self.linear_equality_constraints[idx + 1]

            # [b A][1; x] = 0
            A = linear_bindings_to_homogenuous_form(eqs_i, np.array([]), x)
            B = linear_bindings_to_homogenuous_form(eqs_j, np.array([]), y)

            x_len = len(x)
            y_len = len(y)

            # X = [xx' xy'
            #      yx' yy']
            X = self.Xs[idx]
            xxT = X[:x_len, :x_len]
            xyT = X[:x_len, -y_len:]
            yyT = X[-y_len:, -y_len:]

            x_bar_x_bar_T = np.block([[1, x.T], [x, xxT]])
            x_bar_y_bar_T = np.block([[1, y.T], [x, xyT]])
            y_bar_y_bar_T = np.block([[1, y.T], [y, yyT]])

            # A[1; x] = 0 => A[1; x][1; x]' = 0 and A[1; x][1; y]' = 0
            for c in A.dot(x_bar_x_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)
            for c in A.dot(x_bar_y_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

            # B[1; y] = 0 => B[1; y][1; y]' = 0 and B[1; y][1; x]' = 0
            for c in B.dot(y_bar_y_bar_T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)
            for c in B.dot(x_bar_y_bar_T.T).flatten():
                relaxed_prog.AddLinearEqualityConstraint(c, 0)

        # # Add products of linear constraints
        # for i, j in zip(range(self.num_groups - 1), range(1, self.num_groups)):
        #     x = self.xs[i]
        #
        #     ineqs_i = self.linear_inequality_constraints[i]
        #     ineqs_j = self.linear_inequality_constraints[j]
        #
        #     self.get_A_b_from_lin_consts(ineqs_i, x)
        #     A, b = DecomposeAffineExpressions(ineqs_i, x)

        self.relaxed_prog = relaxed_prog

        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
        result = Solve(relaxed_prog, solver_options=solver_options)
        assert result.is_success()
        breakpoint()
        return self.relaxed_prog


# This script tries to use the Semidefinite relaxation while exploiting sparsity
# It uses the formulation with the approximate exponential map

NUM_CTRL_POINTS = 20
NUM_DIMS = 2

prog = BandSparseSemidefiniteRelaxation(NUM_CTRL_POINTS)

rs = [prog.new_variables(idx, NUM_DIMS, f"r_{idx}") for idx in range(NUM_CTRL_POINTS)]

# Constrain the points to lie on the unit circle
for i in range(NUM_CTRL_POINTS):
    r_i = rs[i]
    so_2_constraint = r_i.T.dot(r_i) - 1
    prog.add_quadratic_constraint(i, i, so_2_constraint, 1, 1)

# Constrain the cosines and sines
for i in range(NUM_CTRL_POINTS):
    r_i = rs[i]
    prog.add_linear_inequality_constraint(i, le(r_i, 1))
    prog.add_linear_inequality_constraint(i, le(-1, r_i))

# Initial conditions
th_initial = 0
th_final = np.pi - 0.1

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(rs[0], create_r_vec_from_angle(th_initial))
final_cond = eq(rs[-1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.add_linear_equality_constraint(0, c)

for c in final_cond:
    prog.add_linear_equality_constraint(-1, c)

# Add in angular velocity
th_dots = [
    prog.new_variables(idx, 1, f"th_dot_{idx}")[0] for idx in range(NUM_CTRL_POINTS - 1)
]


def skew_symmetric(a):
    return np.array([[0, -a], [a, 0]])


def approximate_exp_map(omega_hat):
    return np.eye(NUM_DIMS) + omega_hat + 0.5 * omega_hat @ omega_hat


def rot_matrix(r):
    return np.array([[r[0], -r[1]], [r[1], r[0]]])


ang_vel_constraints = []
for idx in range(NUM_CTRL_POINTS - 1):
    th_dot_k = th_dots[idx]
    R_k = rot_matrix(rs[idx])
    R_k_next = rot_matrix(rs[idx + 1])
    omega_hat_k = skew_symmetric(th_dot_k)

    exp_om_dt = approximate_exp_map(omega_hat_k)
    constraint = exp_om_dt - R_k.T @ R_k_next
    for c in constraint.flatten():
        prog.add_quadratic_constraint(idx, idx + 1, c, 0, 0)

    ang_vel_constraints.append([expr for expr in constraint.flatten()])

# A = np.array([[1, -3], [-2, -6]])
# b = np.array([2, 3])
#
# for var in r.T:
#     consts = le(A.dot(var), b)
#     prog.AddConstraint(consts)

for i in range(NUM_CTRL_POINTS - 1):
    prog.add_quadratic_cost(i, i, pow(th_dots[i], 2))

relaxed_prog = prog.make_relaxation()

# Solve SDP relaxation
relaxed_prog = MakeSemidefiniteRelaxation(prog)
print("Finished formulating SDP relaxation")

solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

from time import time

start = time()
result = Solve(relaxed_prog, solver_options=solver_options)
elapsed_time = time() - start
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")
print(f"Elapsed time: {elapsed_time}")

r_val = result.GetSolution(rs)
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

# plot_cos_sine_trajs(r_val.T)
# plot_cos_sine_trajs(r_val.T, A, b)
# print(result.get_optimal_cost())
