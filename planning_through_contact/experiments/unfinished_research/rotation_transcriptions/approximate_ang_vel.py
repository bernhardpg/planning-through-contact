import numpy as np
from pydrake.math import eq
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
    SolverOptions,
)

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
r_displacements = []
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_disp_i = r_next - r_i
    r_displacements.append(r_disp_i)

    rot_cost_i = r_disp_i.T.dot(r_disp_i)
    prog.AddCost(rot_cost_i)

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

# Add in angular velocity
r_dots = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS - 1, "r_dot")


def skew_symmetric(th):
    return np.array([[0, -th], [th, 0]])


# def rodriguez(om_hat, th):
#     return np.eye(2) + om_hat * np.sin(th) + om_hat @ om_hat * (1 - np.cos(th))


def rodriguez(om_hat, cos_th, sin_th):
    return np.eye(2) + om_hat * sin_th + om_hat @ om_hat * (1 - cos_th)


def rot_matrix(r):
    return np.array([[r[0], -r[1]], [r[1], r[0]]])


om_hat_norm = skew_symmetric(1.0)

ang_vel_constraints = []
for k in range(NUM_CTRL_POINTS - 1):
    R_k = rot_matrix(r[:, k])
    R_k_next = rot_matrix(r[:, k + 1])
    cos_th_dot, sin_th_dot = r_dots[0, k], r_dots[1, k]

    exp_om_dt = rodriguez(om_hat_norm, cos_th_dot, sin_th_dot)
    lhs = R_k.T @ R_k_next

    constraint = eq(exp_om_dt, lhs)
    for c in constraint.flatten():
        prog.AddConstraint(c)

    ang_vel_constraints.append(
        [convert_formula_to_lhs_expression(f) for f in constraint.flatten()]
    )

    # Second side of constraint
    lhs = R_k_next
    rhs = R_k @ exp_om_dt
    constraint = eq(lhs, rhs)
    for c in constraint.flatten():
        prog.AddConstraint(c)

    # second_side_ang_vel_constraints.append(
    #     [convert_formula_to_lhs_expression(f) for f in constraint.flatten()]
    # )


# th_dots = r_dots[0, :]  # small angle approx: sin(th) \approx th
# prog.AddConstraint(ge(th_dots, 0))

# prog.AddCost(th_dots.T @ th_dots)
# prog.AddCost(0.1 * np.sum(th_dots))
# With this magic number it seems that (almost) every cut will choose the obstacle free path
# prog.AddCost(-5.0175 * np.sum(th_dots))

# Absolute value cost
# s = prog.NewContinuousVariables(NUM_CTRL_POINTS - 1, "s")
# prog.AddCost(np.sum(s))
#
# for k in range(NUM_CTRL_POINTS - 1):
#     prog.AddLinearConstraint(s[k] >= th_dots[k])
#     prog.AddLinearConstraint(s[k] >= -th_dots[k])


# prog.AddCost(th_dots.T @ th_dots)
# prog.AddCost(-np.sum(th_dots))
# prog.AddCost(10 * np.sum(r_displacements))


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

r_val = result.GetSolution(r)
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

plot_cos_sine_trajs(r_val.T)
# plot_cos_sine_trajs(r_val.T, A, b)
