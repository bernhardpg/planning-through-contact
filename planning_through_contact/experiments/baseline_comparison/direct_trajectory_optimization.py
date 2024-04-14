from typing import List, Optional

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge, sqrt
from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    SimplePlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
    visualize_planar_pushing_trajectory_legacy,
)


def direct_trajopt_through_contact(
    start_and_goal: PlanarPushingStartAndGoal,
    config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    output_name: Optional[str] = None,
    output_folder: str = "direct_trajopt",
    visualize: bool = True,
    print_success: bool = False,
) -> MathematicalProgramResult:
    """
    Runs the direct transcription method described in
    M. Posa, C. Cantu, and R. Tedrake, “A direct method for trajectory optimization
    of rigid bodies through contact”
    """

    import os

    output_path = f"{output_folder}/{output_name}"
    output_name = f"direct_trajopt_{output_name}"
    os.makedirs(output_path, exist_ok=True)

    # visualizer = "old"
    visualizer = "new"
    visualize_initial_guess = False
    assert_found_solution = False

    num_time_steps = 16  # TODO: Change
    dt = 0.4  # TODO: Change
    end_time = num_time_steps * dt - dt
    mu = 0.5

    dynamics_config = config.dynamics_config

    slider = dynamics_config.slider
    pusher_radius = dynamics_config.pusher_radius

    slider_initial_pose = start_and_goal.slider_initial_pose
    slider_target_pose = start_and_goal.slider_target_pose
    pusher_initial_pose = start_and_goal.pusher_initial_pose
    pusher_target_pose = start_and_goal.pusher_target_pose

    assert pusher_initial_pose is not None
    assert pusher_target_pose is not None

    config.start_and_goal = PlanarPushingStartAndGoal(
        slider_initial_pose, slider_target_pose, pusher_initial_pose, pusher_target_pose
    )

    NUM_DIMS = 2

    prog = MathematicalProgram()
    # Define decision variables
    p_WBs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_WBs")
    p_BPs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "p_BPs")
    normal_forces = prog.NewContinuousVariables(num_time_steps - 1, 1, "lambda_n")
    friction_forces = prog.NewContinuousVariables(num_time_steps - 1, 1, "lambda_f")
    force_comps = np.hstack((normal_forces, friction_forces))

    # r_WB_k = [cos_th; sin_th]
    # r_WBs = prog.NewContinuousVariables(num_time_steps, NUM_DIMS, "r_WBs")
    theta_WBs = prog.NewContinuousVariables(num_time_steps, "theta_WBs")

    # Create some convenience variables
    def _two_d_rot_matrix_from_cos_sin(cos, sin) -> npt.NDArray:
        return np.array([[cos, -sin], [sin, cos]])

    def _create_p_WP(p_WB, R_WB, p_BP):
        return p_WB + R_WB @ p_BP

    def _calc_f_c_B(force_comp, p_BP):
        J_c = _calc_contact_jacobian(p_BP)
        gen_force = J_c.T @ force_comp
        f_c_B = gen_force[0:2]
        return f_c_B

    def _calc_f_c_W(force_comp, p_BP, R_WB):
        f_c_B = _calc_f_c_B(force_comp, p_BP)
        return R_WB @ f_c_B

    # R_WBs = [_two_d_rot_matrix_from_cos_sin(r_WB[0], r_WB[1]) for r_WB in r_WBs]
    R_WBs = [two_d_rotation_matrix_from_angle(theta) for theta in theta_WBs]
    p_WPs = [
        _create_p_WP(p_WB, R_WB, p_BP) for p_WB, R_WB, p_BP in zip(p_WBs, R_WBs, p_BPs)
    ]

    # Initial and target constraints on slider
    p_WB_initial = slider_initial_pose.pos().flatten()
    p_WB_target = slider_target_pose.pos().flatten()
    R_WB_initial = two_d_rotation_matrix_from_angle(slider_initial_pose.theta)
    R_WB_target = two_d_rotation_matrix_from_angle(slider_target_pose.theta)
    p_WP_initial = _create_p_WP(p_WB_initial, R_WB_initial, p_BPs[0])
    p_WP_target = _create_p_WP(p_WB_target, R_WB_target, p_BPs[-1])

    def _add_slider_equal_pose_constraint(idx: int, target: PlanarPose):
        prog.AddLinearConstraint(eq(p_WBs[idx], target.pos().flatten()))
        # prog.AddLinearConstraint(eq(r_WBs[idx], np.array([target.cos(), target.sin()])))
        prog.AddLinearConstraint(theta_WBs[idx] == target.theta)

    _add_slider_equal_pose_constraint(0, slider_initial_pose)
    _add_slider_equal_pose_constraint(-1, slider_target_pose)

    # Initial and target constraint on pusher
    for c in eq(p_WP_initial, pusher_initial_pose.pos().flatten()):
        prog.AddLinearEqualityConstraint(c)
    for c in eq(p_WP_target, pusher_target_pose.pos().flatten()):
        prog.AddLinearEqualityConstraint(c)

    # SO(2) constraints
    # for cos_th, sin_th in r_WBs:
    #     prog.AddConstraint(cos_th**2 + sin_th**2 == 1)

    # Dynamics
    _calc_contact_jacobian = lambda p_BP: slider.geometry.get_contact_jacobian(p_BP)  # type: ignore

    def _calc_omega_WB(r_WB_curr, r_WB_next):
        R_WB_curr = _two_d_rot_matrix_from_cos_sin(r_WB_curr[0], r_WB_curr[1])
        R_WB_next = _two_d_rot_matrix_from_cos_sin(r_WB_next[0], r_WB_next[1])
        R_WB_dot = (R_WB_next - R_WB_curr) / dt
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        omega_WB = R_WB_dot.dot(R_WB_curr.T)[1, 0]
        return omega_WB

    def _calc_omega_WB_from_theta(theta_WB_curr, theta_WB_next):
        R_WB_curr = two_d_rotation_matrix_from_angle(theta_WB_curr)
        R_WB_next = two_d_rotation_matrix_from_angle(theta_WB_next)
        R_WB_dot = (R_WB_next - R_WB_curr) / dt
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        omega_WB = R_WB_dot.dot(R_WB_curr.T)[1, 0]
        return omega_WB

    # Limit surface constants
    c_f = dynamics_config.f_max**-2
    c_tau = dynamics_config.tau_max**-2

    def _dynamics_constraint(vars: npt.NDArray) -> npt.NDArray:
        p_WB_curr = vars[0:2]
        p_WB_next = vars[2:4]
        theta_WB_curr = vars[4]
        theta_WB_next = vars[5]
        # r_WB_curr = vars[4:6]
        # r_WB_next = vars[6:8]
        # f_comps = vars[8:10]
        # p_BP = vars[10:12]
        f_comps = vars[6:8]
        p_BP = vars[8:10]

        # R_WB = _two_d_rot_matrix_from_cos_sin(r_WB_curr[0], r_WB_curr[1])
        R_WB = two_d_rotation_matrix_from_angle(theta_WB_curr)
        v_WB = (p_WB_next - p_WB_curr) / dt
        # omega_WB = _calc_omega_WB(r_WB_curr, r_WB_next)
        # omega_WB = _calc_omega_WB_from_theta(theta_WB_curr, theta_WB_next)

        omega_WB = (theta_WB_next - theta_WB_curr) / dt

        J_c = _calc_contact_jacobian(p_BP)
        gen_force = J_c.T @ f_comps

        f_c_B = gen_force[:2]
        trans_vel_constraint = v_WB - R_WB @ (c_f * f_c_B)

        tau_c_B = gen_force[2]
        ang_vel_constraint = omega_WB - c_tau * tau_c_B

        constraint_value = np.concatenate([trans_vel_constraint, [ang_vel_constraint]])
        return constraint_value

    dynamics_constraints = []
    for k in range(num_time_steps - 1):
        p_WB_curr = p_WBs[k]
        p_WB_next = p_WBs[k + 1]
        # r_WB_curr = r_WBs[k]
        # r_WB_next = r_WBs[k + 1]
        theta_WB_curr = theta_WBs[k]
        theta_WB_next = theta_WBs[k + 1]
        f_comp_curr = force_comps[k]
        p_BP_curr = p_BPs[k]

        # vars = np.concatenate(
        #     (p_WB_curr, p_WB_next, r_WB_curr, r_WB_next, f_comp_curr, p_BP_curr)
        # )
        vars = np.concatenate(
            (
                p_WB_curr,
                p_WB_next,
                [theta_WB_curr, theta_WB_next],
                f_comp_curr,
                p_BP_curr,
            )
        )
        const = prog.AddConstraint(
            _dynamics_constraint, np.zeros((3,)), np.zeros((3,)), vars=vars
        )
        dynamics_constraints.append(const)

        # TODO: remove?
        # NOTE:
        # We constrain the difference in theta between knot points to be less than pi,
        # as we approximately calculate omega_WB from their difference, and at delta_theta = pi
        # the approximation really breaks down, i.e. if theta_next - theta_curr = pi then
        # omega_WB = 0.
        # Note that in principle this should not be a limiting constraint.
        # delta_theta_WB = theta_WB_next - theta_WB_curr
        # prog.AddConstraint(delta_theta_WB <= np.pi * 2 / 3)
        # prog.AddConstraint(delta_theta_WB >= -np.pi * 2 / 3)

    # Note: Complementarity constraints are encoded similarly to 3.2 in
    # M. Posa, C. Cantu, and R. Tedrake, “A direct method for trajectory optimization
    # of rigid bodies through contact”
    sdf_slacks = prog.NewContinuousVariables(num_time_steps, "sdf_slacks")
    prog.AddLinearConstraint(ge(sdf_slacks, 0))

    # Enforce non-penetration
    calc_sdf = lambda pos: slider.geometry.get_signed_distance(pos) - pusher_radius  # type: ignore

    def _sdf_equal_to_slack(vars: npt.NDArray) -> npt.NDArray:
        pos = vars[:2]
        slack = vars[2]

        sdf = calc_sdf(pos)
        constraint_res = sdf - slack
        return np.array([constraint_res])

    for k in range(num_time_steps):
        prog.AddConstraint(
            _sdf_equal_to_slack,
            np.zeros((1,)),
            np.zeros((1,)),
            vars=np.concatenate([p_BPs[k], [sdf_slacks[k]]]),
        )

    lambda_n_slacks = prog.NewContinuousVariables(num_time_steps, "lambda_n_slacks")
    prog.AddLinearConstraint(ge(lambda_n_slacks, 0))

    # Enforce friction cone
    for (lambda_n, lambda_f), lambda_n_slack in zip(force_comps, lambda_n_slacks):
        prog.AddLinearConstraint(lambda_n == lambda_n_slack)
        prog.AddLinearConstraint(lambda_f <= mu * lambda_n)
        prog.AddLinearConstraint(lambda_f >= -mu * lambda_n)

    # Complementarity constraints
    for k in range(num_time_steps):
        s_sdf = sdf_slacks[k]
        s_lambda_n = lambda_n_slacks[k]
        EPS = 0.0
        prog.AddConstraint(s_sdf * s_lambda_n <= EPS)

    # Non-sliding constraint
    def _nonsliding_constraint(vars: npt.NDArray) -> npt.NDArray:
        p_BP_curr = vars[0:2]
        p_BP_next = vars[2:4]
        s_lambda_n = vars[4]

        J_c = _calc_contact_jacobian(p_BP_curr)
        v_BP = (p_BP_next - p_BP_curr) / dt

        v_in_contact_frame = (J_c.T @ v_BP)[:-1]
        v_tangential = v_in_contact_frame[1]

        res = v_tangential * s_lambda_n
        return np.array([res])

    for k in range(num_time_steps - 1):
        p_BP_curr = p_BPs[k]
        p_BP_next = p_BPs[k + 1]
        s_lambda_n = lambda_n_slacks[k]

        vars = np.hstack([p_BP_curr, p_BP_next, [s_lambda_n]])
        prog.AddConstraint(
            _nonsliding_constraint, np.zeros((1,)), np.zeros((1,)), vars=vars
        )

    # Cost

    cost_config = config.contact_config.cost

    def _get_slider_vertices(slider: CollisionGeometry) -> List[List[npt.NDArray]]:
        """
        Returns a list of slider vertices in the world frame for each knot point. Note that these vertices
        are linear in the decision varibles of the program
        """
        p_Wv_is = [
            [
                slider.get_p_Wv_i(vertex_idx, R_WB, p_WB.reshape((2, 1)))
                for vertex_idx in range(len(slider.vertices))
            ]
            for p_WB, R_WB in zip(p_WBs, R_WBs)
        ]
        return p_Wv_is

    p_Wv_is = _get_slider_vertices(slider.geometry)
    num_keypoints = len(slider.geometry.vertices)

    # Slider keypoint arc length
    EPS = 1e-5
    assert cost_config.keypoint_arc_length is not None
    for k in range(num_time_steps - 1):
        for vertex_k, vertex_k_next in zip(p_Wv_is[k], p_Wv_is[k + 1]):
            diff = vertex_k_next - vertex_k
            dist = sqrt((diff.T @ diff).item() + EPS)
            cost_expr = cost_config.keypoint_arc_length * (1 / num_keypoints) * dist
            prog.AddCost(cost_expr)  # type: ignore

    # Slider keypoint velocity
    assert cost_config.keypoint_velocity_regularization is not None
    for k in range(num_time_steps - 1):
        for vertex_k, vertex_k_next in zip(p_Wv_is[k], p_Wv_is[k + 1]):
            vel = (vertex_k_next - vertex_k) / dt
            squared_vel = (vel.T @ vel).item()
            cost_expr = (
                cost_config.keypoint_velocity_regularization
                * (1 / num_keypoints)
                * squared_vel
            )
            prog.AddCost(cost_expr)  # type: ignore

    # Time in contact cost
    # assert cost_config.time is not None
    #
    #
    # def _only_time_cost_when_contact(p_BP: npt.NDArray) -> Any:
    #     sdf = calc_sdf(p_BP)
    #
    #     if sdf <= 1e-5:
    #         cost = AutoDiffXd(cost_config.time * dt)  # type: ignore
    #         return cost
    #     else:
    #         zero = AutoDiffXd(0.0)
    #         return zero
    #
    #
    # for k in range(num_time_steps):
    #     p_BP = p_BPs[k]
    #     prog.AddCost(_only_time_cost_when_contact, vars=p_BP)

    cost_config_noncoll = config.non_collision_cost
    # Avoid object
    EPS = 1e-5
    assert cost_config_noncoll.distance_to_object_socp is not None
    # cost_config_noncoll.distance_to_object_socp = 1e-2
    distance_to_object_cost = []
    for s in sdf_slacks:
        cost = prog.AddCost(
            cost_config_noncoll.distance_to_object_socp * 1 / (s + pusher_radius + 0.05)
        )
        distance_to_object_cost.append(cost)

    # Pusher velocity cost
    v_BPs = np.vstack(
        [(p_next - p_curr) / dt for p_next, p_curr in zip(p_BPs[1:], p_BPs[:-1])]
    )
    assert cost_config_noncoll.pusher_velocity_regularization is not None
    cost_config_noncoll.pusher_velocity_regularization = 10
    for v_BP in v_BPs:
        squared_vel = v_BP.T @ v_BP
        cost = cost_config_noncoll.pusher_velocity_regularization * squared_vel
        prog.AddCost(cost)

        # prog.AddConstraint(squared_vel <= 0.2**2)

    # Pusher arc length
    cost_config_noncoll.pusher_arc_length = 1
    for k in range(num_time_steps - 1):
        p_BP_curr = p_BPs[k]
        p_BP_next = p_BPs[k + 1]

        diff = p_BP_next - p_BP_curr
        EPS = 1e-5
        dist = sqrt(diff.T @ diff + EPS)
        prog.AddCost(cost_config_noncoll.pusher_arc_length * dist)

    # Squared forces
    sq_forces_cost = []
    cost_config.force_regularization = 100
    for lambda_n, lambda_f in force_comps:
        cost = cost_config.force_regularization * (lambda_n**2 + lambda_f**2)
        sq_forces_cost.append(prog.AddCost(cost))

    # Create initial guess as straight line interpolation
    def _interpolate_traj_1d(
        initial_val: float, target_val: float, duration: float
    ) -> npt.NDArray[np.float64]:
        xs = np.array([0, end_time])
        ys = np.array([initial_val, target_val])
        x_interpolate = np.arange(0, 0 + duration, dt)
        y_interpolated = np.interp(x_interpolate, xs, ys)
        return y_interpolated

    def _interpolate_traj(
        initial_val: npt.NDArray[np.float64],
        target_val: npt.NDArray[np.float64],
        duration: float,
    ) -> npt.NDArray[np.float64]:
        if len(initial_val.shape) == 2:
            initial_val = initial_val.flatten()
            target_val = target_val.flatten()

        num_dims = initial_val.shape[0]
        trajs = np.vstack(
            [
                _interpolate_traj_1d(initial_val[dim], target_val[dim], duration)
                for dim in range(num_dims)
            ]
        ).T  # (num_time_steps, num_dims)
        return trajs

    th_interpolated = _interpolate_traj_1d(
        slider_initial_pose.theta, slider_target_pose.theta, duration=end_time + dt
    )
    prog.SetInitialGuess(theta_WBs, th_interpolated)  # type: ignore
    r_WBs_initial_guess = [np.array([np.cos(th), np.sin(th)]) for th in th_interpolated]
    # prog.SetInitialGuess(r_WBs, r_WBs_initial_guess)  # type: ignore

    p_WBs_initial_guess = _interpolate_traj(
        slider_initial_pose.pos(), slider_target_pose.pos(), duration=end_time + dt
    )
    prog.SetInitialGuess(p_WBs, p_WBs_initial_guess)  # type: ignore

    def find_closest_point_on_geometry(
        point: npt.NDArray[np.float64], geometry: CollisionGeometry
    ) -> npt.NDArray[np.float64]:
        assert isinstance(geometry, Box2d)  # not yet implemented for T-pusher
        _prog = MathematicalProgram()
        closest_point = _prog.NewContinuousVariables(2, 1, "closest_point")
        x, y = closest_point.flatten()

        _prog.AddLinearConstraint(x <= geometry.width / 2)
        _prog.AddLinearConstraint(x >= -geometry.width / 2)
        _prog.AddLinearConstraint(y <= geometry.height / 2)
        _prog.AddLinearConstraint(y >= -geometry.height / 2)

        diff = closest_point - point
        sq_dist = (diff.T @ diff).item()

        _prog.AddQuadraticCost(sq_dist)  # type: ignore
        _result = Solve(_prog)
        assert _result.is_success()

        return _result.GetSolution(closest_point).reshape((2, 1))

    # Initial guess for finger
    use_touching_initial_guess = False
    use_polar_initial_guess = False
    if use_touching_initial_guess:
        closest_point = find_closest_point_on_geometry(
            pusher_initial_pose.pos(), slider.geometry
        )

        p_BPs_interpolated_start = _interpolate_traj(
            pusher_initial_pose.pos(), closest_point, duration=(end_time + dt) / 2
        )
        p_BPs_interpolated_end = _interpolate_traj(
            closest_point, pusher_target_pose.pos(), duration=(end_time + dt) / 2
        )
        # We make the finger touch the object at the middle point as an initial guess
        p_BPs_initial_guess = np.vstack(
            (p_BPs_interpolated_start, p_BPs_interpolated_end)
        )
        prog.SetInitialGuess(p_BPs, p_BPs_initial_guess)  # type: ignore
    elif use_polar_initial_guess:
        # Use polar coordinates to initialize the finger to move around the object in a circle
        p_BP_initial = pusher_initial_pose.pos()
        angle_initial = np.arctan2(p_BP_initial[1], p_BP_initial[0]).item() % (
            np.pi * 2
        )
        radius_initial = np.linalg.norm(p_BP_initial)

        p_BP_target = pusher_target_pose.pos()
        angle_target = np.arctan2(p_BP_initial[1], p_BP_initial[0]).item() % (np.pi * 2)
        radius_target = np.linalg.norm(p_BP_target)

        angle_interpolated = _interpolate_traj_1d(
            angle_initial, angle_target + 2 * np.pi, duration=end_time + dt
        ) % (np.pi * 2)
        radius_interpolated = _interpolate_traj_1d(
            radius_initial, radius_target, duration=end_time + dt  # type: ignore
        )

        p_BPs_initial_guess = np.vstack(
            [
                np.array([r * np.cos(phi), r * np.sin(phi)])
                for r, phi in zip(radius_interpolated, angle_interpolated)
            ]
        )
        prog.SetInitialGuess(p_BPs, p_BPs_initial_guess)  # type: ignore
    else:
        # print("Using straight line interpolation as initial guess")
        p_BPs_initial_guess = _interpolate_traj(
            pusher_initial_pose.pos(), pusher_target_pose.pos(), duration=end_time + dt
        )
        prog.SetInitialGuess(p_BPs, p_BPs_initial_guess)  # type: ignore

    normal_forces_initial_guess = np.ones(normal_forces.shape) * 0.05
    prog.SetInitialGuess(normal_forces, normal_forces_initial_guess)  # type: ignore

    friction_forces_initial_guess = np.ones(friction_forces.shape) * 0
    prog.SetInitialGuess(friction_forces, friction_forces_initial_guess)  # type: ignore

    # Make quantities so we can plot initial guess
    R_WBs_initial_guess = [
        _two_d_rot_matrix_from_cos_sin(cos, sin) for cos, sin in r_WBs_initial_guess
    ]
    p_WPs_initial_guess = np.vstack(
        [
            _create_p_WP(p_WB, R_WB, p_BP)
            for p_WB, R_WB, p_BP in zip(
                p_WBs_initial_guess, R_WBs_initial_guess, p_BPs_initial_guess
            )
        ]
    )
    force_comps_initial_guess = np.hstack(
        (normal_forces_initial_guess, friction_forces_initial_guess)
    )
    f_c_Ws_initial_guess = np.vstack(
        [
            _calc_f_c_W(force_comp, p_BP, R_WB)
            for force_comp, p_BP, R_WB in zip(
                force_comps_initial_guess, p_BPs_initial_guess, R_WBs_initial_guess
            )
        ]
    )

    # Solve program
    # Copied from the convex planning code (PlanarPushingPath)
    snopt = SnoptSolver()

    solver_options = SolverOptions()
    if solver_params.nonl_rounding_save_solver_output:
        import os

        snopt_log_path = f"{output_path}/{output_name}_snopt_output.txt"
        # Delete log file if it already exists as Snopt just keeps writing to the same file
        if os.path.exists(snopt_log_path):
            os.remove(snopt_log_path)

        solver_options.SetOption(snopt.solver_id(), "Print file", snopt_log_path)

    # solver_options.SetOption(
    #     snopt.solver_id(),
    #     "Major Feasibility Tolerance",
    #     solver_params.nonl_round_major_feas_tol,
    # )
    # solver_options.SetOption(
    #     snopt.solver_id(),
    #     "Major Optimality Tolerance",
    #     solver_params.nonl_round_opt_tol,
    # )
    # The performance seems to be better when these (minor step) parameters are left
    # to their default value
    # solver_options.SetOption(
    #     snopt.solver_id(),
    #     "Minor Feasibility Tolerance",
    #     solver_params.nonl_round_minor_feas_tol,
    # )
    # solver_options.SetOption(
    #     snopt.solver_id(),
    #     "Minor Optimality Tolerance",
    #     solver_params.nonl_round_opt_tol,
    # )
    # solver_options.SetOption(
    #     snopt.solver_id(),
    #     "Major iterations limit",
    #     solver_params.nonl_round_major_iter_limit,
    # )

    result = snopt.Solve(prog, solver_options=solver_options)  # type: ignore
    if not visualize_initial_guess and assert_found_solution:
        assert result.is_success()

    if print_success:
        print(f"{output_name}: result.is_success() = {result.is_success()}")
    if result.is_success():
        if solver_params.print_cost:
            print(f"Direct trajopt cost: {result.get_optimal_cost()}")

        normal_force_sols = result.GetSolution(normal_forces)
        force_comps_sols = result.GetSolution(force_comps)
        p_BPs_sols = result.GetSolution(p_BPs)
        R_WBs_sols = [evaluate_np_expressions_array(R_WB, result) for R_WB in R_WBs]
        f_c_Ws_sols = np.vstack(
            [
                _calc_f_c_W(force_comp_sol, p_BP_sol, R_WB_sol)
                for force_comp_sol, p_BP_sol, R_WB_sol in zip(
                    force_comps_sols, p_BPs_sols, R_WBs_sols
                )
            ]
        )

        # theta_WBs_sols = [np.arccos(R[0, 0]) for R in R_WBs_sols]
        theta_WBs_sols = result.GetSolution(theta_WBs)

        # sdfs_sols = [calc_sdf(p_BP_sol).item() for p_BP_sol in p_BPs_sols]
        # compl_consts_vals = [
        #     sdf * lambda_n for sdf, lambda_n in zip(sdfs_sols, normal_force_sols)
        # ]

        p_WBs_sols = result.GetSolution(p_WBs)
        p_BPs_sols = result.GetSolution(p_BPs)

        # TODO: Remove
        c_res = []
        for const in dynamics_constraints:
            eval = const.evaluator()
            vars = const.variables()
            vars_val = result.GetSolution(vars)

            res = eval.Eval(vars_val)
            c_res.append(res)

        omega_WB_sols = [
            _calc_omega_WB_from_theta(th, th_next)
            for th, th_next in zip(theta_WBs_sols[:-1], theta_WBs_sols[1:])
        ]

        J_c_sols = [_calc_contact_jacobian(p_BP) for p_BP in p_BPs_sols]
        gen_force_sols = [
            J_c.T @ f_comps for J_c, f_comps in zip(J_c_sols, force_comps_sols)
        ]

        sq_forces_costs_vals = [
            cost.evaluator().Eval(result.GetSolution(cost.variables()))
            for cost in sq_forces_cost
        ]

        distance_to_object_costs_vals = [
            cost.evaluator().Eval(result.GetSolution(cost.variables()))
            for cost in distance_to_object_cost
        ]

        dynamics_violations = []
        for k in range(num_time_steps - 1):
            # for k in [1]:
            p_WB_curr = p_WBs_sols[k]
            p_WB_next = p_WBs_sols[k + 1]
            theta_WB_curr = theta_WBs_sols[k]
            theta_WB_next = theta_WBs_sols[k + 1]
            f_comp_curr = force_comps_sols[k]
            p_BP_curr = p_BPs_sols[k]

            vars = np.concatenate(
                (
                    p_WB_curr,
                    p_WB_next,
                    [theta_WB_curr, theta_WB_next],
                    f_comp_curr,
                    p_BP_curr,
                )
            )
            dynamics_violations.append(_dynamics_constraint(vars))

            const = dynamics_constraints[k]
            const_vars = const.variables()
            vars_sol = result.GetSolution(const_vars)

        if visualize:
            if visualizer == "new":
                traj = SimplePlanarPushingTrajectory(
                    result.GetSolution(p_WBs),
                    [evaluate_np_expressions_array(R_WB, result) for R_WB in R_WBs],
                    evaluate_np_expressions_array(p_WPs, result),  # type: ignore
                    f_c_Ws_sols,
                    dt,
                    config,
                )
                if output_name is None:
                    output_name = "untitled"

                filename = f"{output_path}/{output_name}"
                visualize_planar_pushing_trajectory(
                    traj,  # type: ignore
                    save=True,
                    filename=filename,
                    visualize_knot_points=True,
                )
            else:

                if visualize_initial_guess:
                    traj_old = OldPlanarPushingTrajectory(
                        dt,
                        R_WBs_initial_guess,
                        p_WBs_initial_guess.T,
                        p_WPs_initial_guess.T,  # type: ignore
                        np.hstack([f_c_Ws_initial_guess.T, np.zeros((2, 1))]),
                        p_BPs_initial_guess.T,
                    )
                else:
                    traj_old = OldPlanarPushingTrajectory(
                        dt,
                        [evaluate_np_expressions_array(R_WB, result) for R_WB in R_WBs],
                        result.GetSolution(p_WBs).T,
                        evaluate_np_expressions_array(p_WPs, result).T,  # type: ignore
                        np.hstack([f_c_Ws_sols.T, np.zeros((2, 1))]),
                        result.GetSolution(p_BPs).T,
                    )
                visualize_planar_pushing_trajectory_legacy(
                    traj_old, slider.geometry, pusher_radius=pusher_radius
                )

    return result
