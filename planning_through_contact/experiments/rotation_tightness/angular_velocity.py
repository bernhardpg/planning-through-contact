import numpy as np
from pydrake.math import eq, le
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
    SolverOptions,
)

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs

NUM_CTRL_POINTS = 8
NUM_DIMS = 2

prog = MathematicalProgram()

r = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS, "r")

# Constrain the points to lie on the unit circle
for i in range(NUM_CTRL_POINTS):
    r_i = r[:, i]
    so_2_constraint = r_i.T.dot(r_i) == 1
    prog.AddConstraint(so_2_constraint)

# Minimize squared euclidean distances in rotation parameters
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_dot_i = r_next - r_i

    rot_cost_i = r_dot_i.T.dot(r_dot_i)
    prog.AddCost(rot_cost_i)

# Initial conditions
th_initial = np.pi
th_final = 0 + 0.1

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.AddConstraint(c)

for c in final_cond:
    prog.AddConstraint(c)

# Add in angular velocity
th_dots = prog.NewContinuousVariables(NUM_CTRL_POINTS - 1, "th_dot")


def skew_symmetric(th):
    return np.array([[0, -th], [th, 0]])


def rodriguez(om_hat, dt):
    return np.eye(2) + om_hat * np.sin(dt) + om_hat @ om_hat * (1 - np.cos(dt))


def rot_matrix(r):
    return np.array([[r[0], -r[1]], [r[1], r[0]]])


delta_t = 0.2

ang_vel_constraints = []
for k in range(NUM_CTRL_POINTS - 1):
    th_dot_k = th_dots[k]
    R_k = rot_matrix(r[:, k])
    R_k_next = rot_matrix(r[:, k + 1])
    om_hat_k = skew_symmetric(th_dot_k)

    exp_om_dt = rodriguez(om_hat_k, delta_t)
    lhs = R_k.T @ R_k_next

    constraint = eq(exp_om_dt, lhs)
    for c in constraint.flatten():
        prog.AddConstraint(c)

    ang_vel_constraints.append(
        [convert_formula_to_lhs_expression(f) for f in constraint.flatten()]
    )

# prog.AddCost(np.sum(th_dots))


# Solve SDP relaxation
# relaxed_prog, X, mon_basis = create_sdp_relaxation(prog)
relaxed_prog = MakeSemidefiniteRelaxation(prog)
print("Finished formulating SDP relaxation")

# solver_options = SolverOptions()
# solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

result = Solve(relaxed_prog)
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")

# X_val = result.GetSolution(X)
#
# tol = 1e-4
# num_nonzero_eigvals = len(
#     [val for val in np.linalg.eigvals(X_val) if np.abs(val) >= tol]
# )
# print(f"Rank of X: {num_nonzero_eigvals}")
# print(f"cost: {result.get_optimal_cost()}")
#
# x_val = X_val[:, 0]

r_val = result.GetSolution(r)
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

breakpoint()


plot_cos_sine_trajs(r_val.T)
