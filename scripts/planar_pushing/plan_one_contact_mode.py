# Script for solving the SDP relaxation of the planning problem within a single
# face contact mode. Useful for experimenting with the relaxation tightness and
# the effect of different constraints/cuts.

from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MosekSolver,
    SnoptSolver,
    SolverOptions,
)

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.analysis import (
    analyze_mode_result,
    get_constraint_violation_for_face_mode,
)
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    assemble_progs_from_contact_modes,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

# ---- Configuration ----

# Slider geometry: "box", "sugar_box", "tee", "triangle", etc.
slider_type = "sugar_box"
pusher_radius = 0.015

# Adds a half-plane cut on SO(2) derived from the initial/final orientation.
# This is what the GCS planner does by default.
use_so2_cut = False

# When True, keeps translational dynamics in both the world frame
# (v_WB = R_WB @ c_f * f_c_B) and the body frame (R_WB^T @ v_WB = c_f * f_c_B).
# These are redundant for the original nonconvex problem, but can tighten the
# SDP relaxation. When False, only the world-frame version is kept.
use_redundant_constraints = True

config = get_default_plan_config(
    slider_type=slider_type,
    pusher_radius=pusher_radius,
)
config.num_knot_points_contact = 5

# ---- Problem setup ----

# Which face of the slider the pusher is in contact with
contact_location = PolytopeContactLocation(ContactLocation.FACE, 3)

# PlanarPose(x, y, theta)
initial_pose = PlanarPose(0, 0, 0)
final_pose = PlanarPose(0.3, 0.2, 0.4)
config.start_and_goal = PlanarPushingStartAndGoal(initial_pose, final_pose)

mode = FaceContactMode.create_from_plan_spec(contact_location, config)
mode.set_slider_initial_pose(initial_pose)
mode.set_slider_final_pose(final_pose)

if use_so2_cut:
    mode.add_so2_cut_from_boundary_conds()

if not use_redundant_constraints:
    for c in mode.redundant_constraints:
        mode.prog.RemoveConstraint(c)

# ---- Output directory ----

output_dir = (
    Path("outputs") / f"one_contact_mode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir.resolve()}")

# ---- Solve SDP relaxation ----

mode.formulate_convex_relaxation()
print("Finished formulating convex relaxation")

# solver = ClarabelSolver()
solver = MosekSolver()

solver_options = SolverOptions()
solver_log_name = f"{solver.solver_id().name().lower()}_log.txt"
solver_options.SetOption(CommonSolverOption.kPrintFileName, str(output_dir / solver_log_name))  # type: ignore

start = time()
result = solver.Solve(mode.relaxed_prog, solver_options=solver_options)  # type: ignore
relaxation_time = time() - start
assert result.is_success()
print(f"Relaxation cost: {result.get_optimal_cost()}")
print(f"Relaxation solve time: {relaxation_time:.4f}s")

# ---- Tightness analysis ----

# Eigenvalue analysis: for a tight relaxation, each PSD block should be rank-1
TIGHTNESS_THRESHOLD = 100  # sigma_1/sigma_2 ratio above which a block is considered rank-1
Xs = mode.get_Xs()
X_sols = [evaluate_np_expressions_array(X, result) for X in Xs]
eig_ratios = []
for X_sol in X_sols:
    eigs = np.sort(np.abs(np.linalg.eigvalsh(X_sol)))[::-1]
    ratio = eigs[0] / eigs[1] if eigs[1] > 1e-12 else float("inf")
    eig_ratios.append(ratio)
min_ratio = min(eig_ratios)
is_tight = min_ratio > TIGHTNESS_THRESHOLD
print(f"Relaxation is tight: {is_tight} (min eigenvalue ratio: {min_ratio:.1f})")

# Constraint violations: how well the relaxed solution satisfies original constraints
constraint_violations = get_constraint_violation_for_face_mode(mode, result)

# ---- Visualize relaxed trajectory ----

relaxed_vars = mode.variables.eval_result(result)
relaxed_traj = PlanarPushingTrajectory(config, [relaxed_vars])

# Save eigenvalue + constraint violation + cos/sin trajectory plots
analysis_dir = output_dir / "tightness_analysis"
analysis_dir.mkdir(exist_ok=True)
analyze_mode_result(mode, relaxed_traj, result, filename=str(analysis_dir / "analysis"))
print(f"Saved analysis plots to: {analysis_dir}/")

relaxed_video_path = str(output_dir / "trajectory_relaxed")
visualize_planar_pushing_trajectory(
    relaxed_traj, visualize_knot_points=True, save=True, filename=relaxed_video_path
)
print(f"Saved relaxed video to: {relaxed_video_path}.mp4")

# ---- Nonlinear rounding with SNOPT ----
# Solve the original nonconvex program using the relaxed solution as initial guess.

print("Starting nonlinear rounding...")

# Build the nonlinear program from the original (non-relaxed) constraints
rounding_prog = assemble_progs_from_contact_modes([mode])

# Extract initial guess from relaxed solution and project (cos, sin) onto unit circle
initial_guess_vals = result.GetSolution(mode.relaxed_prog.decision_variables())
orig_var_indices = mode.relaxed_prog.FindDecisionVariableIndices(
    mode.prog.decision_variables()
)
initial_guess = initial_guess_vals[orig_var_indices]

for k in range(mode.num_knot_points):
    cos_sin = np.array([
        initial_guess[rounding_prog.FindDecisionVariableIndex(mode.variables.cos_ths[k])],
        initial_guess[rounding_prog.FindDecisionVariableIndex(mode.variables.sin_ths[k])],
    ])
    length = np.linalg.norm(cos_sin)
    if length > 0:
        cos_sin /= length
    initial_guess[rounding_prog.FindDecisionVariableIndex(mode.variables.cos_ths[k])] = cos_sin[0]
    initial_guess[rounding_prog.FindDecisionVariableIndex(mode.variables.sin_ths[k])] = cos_sin[1]

snopt = SnoptSolver()
snopt_options = SolverOptions()
snopt_options.SetOption(snopt.solver_id(), "Major Feasibility Tolerance", 1e-3)
snopt_options.SetOption(snopt.solver_id(), "Major Optimality Tolerance", 1e-4)
snopt_options.SetOption(snopt.solver_id(), "Major iterations limit", 10000)
snopt_log_path = str(output_dir / "snopt_log.txt")
snopt_options.SetOption(snopt.solver_id(), "Print file", snopt_log_path)

start = time()
rounding_result = snopt.Solve(rounding_prog, initial_guess, solver_options=snopt_options)
rounding_time = time() - start

rounding_success = rounding_result.is_success()
print(f"Rounding status: {rounding_result.get_solution_result()}")
print(f"Rounding solve time: {rounding_time:.4f}s")
if rounding_success:
    print(f"Rounded cost: {rounding_result.get_optimal_cost()}")

# ---- Visualize rounded trajectory (if successful) ----

if rounding_success:
    rounded_vars = mode.get_variable_solutions(rounding_result)
    rounded_traj = PlanarPushingTrajectory(config, [rounded_vars])

    rounded_video_path = str(output_dir / "trajectory_rounded")
    visualize_planar_pushing_trajectory(
        rounded_traj, visualize_knot_points=True, save=True, filename=rounded_video_path
    )
    print(f"Saved rounded video to: {rounded_video_path}.mp4")
else:
    print("Rounding failed, skipping rounded trajectory video.")

# ---- Save run log ----

log_path = output_dir / "run_log.txt"
with open(log_path, "w") as f:
    f.write(f"timestamp: {datetime.now().isoformat()}\n")
    f.write(f"\n")
    f.write(f"--- Configuration ---\n")
    f.write(f"slider_type: {slider_type}\n")
    f.write(f"pusher_radius: {pusher_radius}\n")
    f.write(f"contact_location: {contact_location}\n")
    f.write(f"initial_pose: {initial_pose}\n")
    f.write(f"final_pose: {final_pose}\n")
    f.write(f"num_knot_points: {config.num_knot_points_contact}\n")
    f.write(f"use_so2_cut: {use_so2_cut}\n")
    f.write(f"use_redundant_constraints: {use_redundant_constraints}\n")
    f.write(f"\n")
    f.write(f"--- SDP Relaxation ---\n")
    f.write(f"solver: {solver.solver_id().name()}\n")
    f.write(f"solver_status: {result.get_solution_result()}\n")
    f.write(f"optimal_cost: {result.get_optimal_cost()}\n")
    f.write(f"solve_time: {relaxation_time:.4f}s\n")
    f.write(f"\n")
    f.write(f"--- Tightness ---\n")
    f.write(f"Eigenvalue ratios (sigma_1/sigma_2 per PSD block):\n")
    for i, ratio in enumerate(eig_ratios):
        eigs = np.sort(np.abs(np.linalg.eigvalsh(X_sols[i])))[::-1]
        f.write(f"  block {i}: ratio = {ratio:.1f}, top 3 eigs = {eigs[:3]}\n")
    f.write(f"min ratio across blocks: {min_ratio:.1f}\n")
    f.write(f"\n")
    f.write(f"Constraint violations (mean abs per type):\n")
    for key, vals in constraint_violations.items():
        f.write(f"  {key}: {np.mean(vals):.2e}\n")
    f.write(f"\n")
    f.write(f"--- Nonlinear Rounding (SNOPT) ---\n")
    f.write(f"rounding_status: {rounding_result.get_solution_result()}\n")
    f.write(f"rounding_success: {rounding_success}\n")
    f.write(f"rounding_time: {rounding_time:.4f}s\n")
    if rounding_success:
        f.write(f"rounded_cost: {rounding_result.get_optimal_cost()}\n")
print(f"Saved run log to: {log_path}")
