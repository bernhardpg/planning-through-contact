import numpy as np
from pydrake.math import eq, le
from pydrake.solvers import Solve, SolverOptions

from planning_through_contact.convex_relaxation.band_sparse_semidefinite_relaxation import (
    BandSparseSemidefiniteRelaxation,
)
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs

# This script tries to use the Semidefinite relaxation while exploiting sparsity

NUM_CTRL_POINTS = 400
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

# Minimize squared euclidean distances in rotation parameters
# r_displacements = []
# for i in range(NUM_CTRL_POINTS - 1):
#     r_i = rs[i]
#     r_next = rs[i + 1]
#     r_disp_i = r_next - r_i
#     r_displacements.append(r_disp_i)
#
#     rot_cost_i = r_disp_i.T.dot(r_disp_i)
#     prog.add_quadratic_cost(i, i + 1, rot_cost_i)

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
#
# A = np.array([[1, -3], [-2, -6]])
# b = np.array([2, 3])
#
# for var in r.T:
#     consts = le(A.dot(var), b)
#     prog.AddConstraint(consts)

for i in range(NUM_CTRL_POINTS - 1):
    prog.add_quadratic_cost(i, i, pow(th_dots[i], 2))

# for i in range(NUM_CTRL_POINTS - 1):
#     prog.add_linear_cost(i, -0.2 * th_dots[i])

# Solve SDP relaxation
relaxed_prog = prog.make_relaxation()
print("Finished formulating SDP relaxation")

solver_options = SolverOptions()
# solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

from time import time

start = time()
result = Solve(relaxed_prog, solver_options=solver_options)
elapsed_time = time() - start
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")
print(f"Elapsed time: {elapsed_time}")

r_val = result.GetSolution(rs)
plot_cos_sine_trajs(r_val)
# plot_cos_sine_trajs(r_val.T, A, b)
# print(result.get_optimal_cost())
