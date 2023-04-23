import numpy as np
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation
from visualize.analysis import plot_cos_sine_trajs

NUM_CTRL_POINTS = 5
NUM_PARAMS = 9
NUM_DIMS = 3


def cross(u, v):
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            -(u[0] * v[2] - u[2] * v[0]),
            u[0] * v[1] - u[1] * v[0],
        ]
    )


prog = MathematicalProgram()

r = prog.NewContinuousVariables(NUM_PARAMS, NUM_CTRL_POINTS, "r")

Rs = [r_i.reshape((NUM_DIMS, NUM_DIMS), order="F") for r_i in r.T]
# R = [
#       r1 r4 r7
#       r2 r5 r8
#       r3 r6 r9
#   ]

# Constrain the vectors to have unit norm
for k in range(NUM_CTRL_POINTS):
    R = Rs[k]

    r_vecs = [r for r in R.T]
    for r_vec in r_vecs:
        unit_norm_constraint = r_vec.T.dot(r_vec) == 1
        prog.AddConstraint(unit_norm_constraint)

# Constrain the vectors to be orthogonal
for k in range(NUM_CTRL_POINTS):
    R = Rs[k]

    r_vecs = [r for r in R.T]
    orthogonality_constraints = [
        r_vecs[i].dot(r_vecs[j]) == 0
        for i in range(NUM_DIMS)
        for j in range(NUM_DIMS)
        if i != j
    ]
    for c in orthogonality_constraints:
        prog.AddConstraint(c)

# Constrain vectors to be orthogonal
for k in range(NUM_CTRL_POINTS):
    R = Rs[k]

    r_vecs = [r for r in R.T]
    cross_constraint = eq(cross(r_vecs[0], r_vecs[1]), r_vecs[2])
    prog.AddConstraint(cross_constraint)

# Minimize energy
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_dot_i = r_next - r_i

    r_sq_norm = r_dot_i.T.dot(r_dot_i)
    prog.AddCost(r_sq_norm)

# Initial conditions
th_initial = 0
th_final = np.pi / 2


def R_x(th):
    return np.array(
        [
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th), np.cos(th), 0],
            [0, 0, 1],
        ]
    )


initial_cond = eq(Rs[0], R_x(th_initial)).flatten()
final_cond = eq(Rs[-1], R_x(th_final)).flatten()

prog.AddLinearConstraint(initial_cond)
prog.AddLinearConstraint(final_cond)

import time

start = time.time()
print("Starting to create SDP relaxation...")
relaxed_prog, X, mon_basis = create_sdp_relaxation(prog)
end = time.time()
print(f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds")

print("Solving...")
start = time.time()
result = Solve(relaxed_prog)
end = time.time()
print(f"Solved in {end - start} seconds")
assert result.is_success()
print("Success!")
breakpoint()

X_val = result.GetSolution(X)
x_val = X_val[:, 0]

r_val = x_val[1 : NUM_CTRL_POINTS * NUM_PARAMS + 1]
r_val = r_val.reshape((NUM_PARAMS, NUM_CTRL_POINTS), order="F")

# TODO continue here

plot_cos_sine_trajs(r_val.T)
breakpoint()
