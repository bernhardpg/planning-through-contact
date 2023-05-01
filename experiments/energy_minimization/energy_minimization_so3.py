from typing import Tuple

import numpy as np
import numpy.typing as npt
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import create_sdp_relaxation
from visualize.analysis import plot_cos_sine_trajs

NUM_CTRL_POINTS = 6
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


def R_z(th):
    return np.array(
        [
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th), np.cos(th), 0],
            [0, 0, 1],
        ]
    )


def R_y(th):
    return np.array(
        [
            [np.cos(th), 0, np.sin(th)],
            [0, 1, 0],
            [-np.sin(th), 0, np.cos(th)],
        ]
    )


def R_x(th):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(th), -np.sin(th)],
            [0, np.sin(th), np.cos(th)],
        ]
    )


def create_R(ths: Tuple[float, float, float]) -> npt.NDArray[np.float64]:
    """
    Creates a rotation matrix using the ZYX euler convention
    """
    return R_x(ths[0]).dot(R_y(ths[1])).dot(R_z(ths[2]))


# Initial conditions
th_initial = (0, 0, 0)
eps = 1e-4
th_final = (np.pi - eps, 0, 0)

R_initial = create_R(th_initial)
R_final = create_R(th_final)

initial_cond = eq(Rs[0], R_initial).flatten()
final_cond = eq(Rs[-1], R_final).flatten()

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

X_val = result.GetSolution(X)
tol = 1e-5

num_nonzero_eigvals = len(
    [val for val in np.linalg.eigvals(X_val) if np.abs(val) >= tol]
)
print(f"Rank of X: {num_nonzero_eigvals}")


x_val = X_val[1:, 0]
r_val = x_val.reshape((NUM_PARAMS, NUM_CTRL_POINTS), order="F")

R_vals = [r_i.reshape((NUM_DIMS, NUM_DIMS), order="F") for r_i in r_val.T]

breakpoint()


# Visualize in meshcat
def R_to_transform(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = R
    return T


Ts = [R_to_transform(R) for R in R_vals]

import meshcat
from meshcat.geometry import Box

vis = meshcat.Visualizer()

vis["box1"].set_object(Box([0.1, 0.1, 0.1]))
time.sleep(10)

for T in Ts:
    vis["box1"].set_transform(T)
    time.sleep(1)

breakpoint()
