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

# Initial conditions
th_initial = 0
th_final = np.pi + 0.1

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.AddConstraint(c)

for c in final_cond:
    prog.AddConstraint(c)


def skew_symmetric(a):
    return np.array([[0, -a], [a, 0]])


def rot_matrix(r):
    return np.array([[r[0], -r[1]], [r[1], r[0]]])


# Add in angular velocity
th_dots = prog.NewContinuousVariables(NUM_CTRL_POINTS - 1, "th_dot")
delta_rs = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS - 1, "delta_r")

# Constrain the points to lie on the unit circle
for i in range(NUM_CTRL_POINTS - 1):
    r_i = delta_rs[:, i]
    so_2_constraint = r_i.T.dot(r_i) == 1
    prog.AddConstraint(so_2_constraint)

ang_vel_constraints = []
for k in range(NUM_CTRL_POINTS - 1):
    th_dot_k = th_dots[k]
    R_k = rot_matrix(r[:, k])
    R_k_next = rot_matrix(r[:, k + 1])

    delta_R_k = rot_matrix(delta_rs[:, k])

    constraint = eq(delta_R_k, R_k.T @ R_k_next)
    for c in constraint.flatten():
        prog.AddConstraint(c)

    # constraint = eq(R_k @ delta_R_k, R_k_next)
    # for c in constraint.flatten():
    #     prog.AddConstraint(c)

    omega_hat_k = skew_symmetric(th_dot_k)

    I = np.eye(NUM_DIMS)

    # first side of constraint
    constraint = eq((I - omega_hat_k) @ delta_R_k, (I + omega_hat_k))
    for c in constraint.flatten():
        prog.AddConstraint(c)

    # second side of constraint
    # constraint = eq((I - omega_hat_k), (I + omega_hat_k) @ delta_R_k.T)
    # for c in constraint.flatten():
    #     prog.AddConstraint(c)

    ang_vel_constraints.append(
        [convert_formula_to_lhs_expression(f) for f in constraint.flatten()]
    )

prog.AddCost(th_dots.T @ th_dots)
prog.AddCost(0.5 * np.sum(th_dots))

A = np.array([[1, -3], [-2, -6]])
b = np.array([2, 3])

for var in r.T:
    consts = le(A.dot(var), b)
    prog.AddConstraint(consts)


# Solve SDP relaxation
relaxed_prog = MakeSemidefiniteRelaxation(prog)

print("Finished formulating SDP relaxation")

solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

result = Solve(relaxed_prog, solver_options=solver_options)
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")

r_val = result.GetSolution(r)
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

print(result.GetSolution(ang_vel_constraints))  # type: ignore
print(result.get_optimal_cost())

# plot_cos_sine_trajs(r_val.T)
plot_cos_sine_trajs(r_val.T, A, b)
