import numpy as np
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation
from visualize.analysis import plot_cos_sine_trajs

NUM_CTRL_POINTS = 12
NUM_DIMS = 2

prog = MathematicalProgram()

r = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS, "r")

# Constrain the points to lie on the unit circle
for i in range(NUM_CTRL_POINTS):
    r_i = r[:, i]
    so_2_constraint = r_i.T.dot(r_i) == 1
    prog.AddConstraint(so_2_constraint)

# Minimize energy
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_dot_i = r_next - r_i

    rot_cost_i = r_dot_i.T.dot(r_dot_i)
    prog.AddCost(rot_cost_i)

# Initial conditions
th_initial = 0
th_final = np.pi

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.AddConstraint(c)

for c in final_cond:
    prog.AddConstraint(c)

# Solve SDP relaxation
relaxed_prog, X, mon_basis = create_sdp_relaxation(prog)
result = Solve(relaxed_prog)
assert result.is_success()
breakpoint()

X_val = result.GetSolution(X)
x_val = X_val[:, 0]

r_val = x_val[1 : NUM_CTRL_POINTS * NUM_DIMS + 1]
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

# TODO continue here

plot_cos_sine_trajs(r_val.T)
breakpoint()
