import argparse
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.solvers import Solve
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.geometry.bezier import BezierCurve
from planning_through_contact.geometry.two_d.box_2d import Box2d
from planning_through_contact.geometry.two_d.contact.contact_pair_2d import (
    ContactPairDefinition,
)
from planning_through_contact.geometry.two_d.contact.contact_scene_2d import (
    ContactScene2d,
)
from planning_through_contact.geometry.two_d.contact.types import (
    ContactLocation,
    ContactMode,
)
from planning_through_contact.geometry.two_d.equilateral_polytope_2d import (
    EquilateralPolytope2d,
)
from planning_through_contact.geometry.two_d.rigid_body_2d import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.contact_mode_motion_planner import (
    ContactModeMotionPlanner,
)
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.visualize.analysis import (
    create_forces_eq_and_opposite_analysis,
    create_static_equilibrium_analysis,
    plot_cos_sine_trajs,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationCone2d,
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def make_so3(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a SO(2) rotation matrix and returns a rotation matrix in SO(3), where the original matrix
    is treated as a rotation about the z-axis.
    """
    R_in_SO3 = np.eye(3)
    R_in_SO3[0:2, 0:2] = R
    return R_in_SO3


def interpolate_so2_using_slerp(
    Rs: List[npt.NDArray[np.float64]],
    start_time: float,
    end_time: float,
    dt: float,
) -> List[npt.NDArray[np.float64]]:
    """
    Assumes evenly spaced knot points R_matrices.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    Rs_in_SO3 = [make_so3(R) for R in Rs]
    knot_point_times = np.linspace(start_time, end_time, len(Rs))
    quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)

    traj_times = np.arange(start_time, end_time, dt)
    R_traj_in_SO2 = [
        quat_slerp_traj.orientation(t).rotation()[0:2, 0:2] for t in traj_times
    ]

    return R_traj_in_SO2


def interpolate_w_first_order_hold(
    values: npt.NDArray[np.float64],  # (NUM_SAMPLES, NUM_DIMS)
    start_time: float,
    end_time: float,
    dt: float,
) -> npt.NDArray[np.float64]:  # (NUM_POINTS, NUM_DIMS)
    """
    Assumes evenly spaced knot points.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    knot_point_times = np.linspace(start_time, end_time, len(values))

    # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
    first_order_hold = PiecewisePolynomial.FirstOrderHold(knot_point_times, values.T)
    traj_times = np.arange(start_time, end_time, dt)
    traj = np.hstack(
        [first_order_hold.value(t) for t in traj_times]
    ).T  # transpose to match format in rest of project

    return traj


def eval_expression_vector_with_traj_values(
    expressions: NpExpressionArray,  # (num_dims, 1)
    all_variables: NpVariableArray,  # (num_variables, )
    all_variable_trajs: npt.NDArray[np.float64],  # (num_timesteps, num_variables)
) -> npt.NDArray[np.float64]:
    # We create one env with all the variable values per timestep
    # NOTE: This is possibly a slow way of doing things
    variable_values_at_each_timestep = [
        {var: val for var, val in zip(all_variables, vals)}
        for vals in all_variable_trajs
    ]
    expr_traj = np.vstack(
        [
            np.array([e.item().Evaluate(variable_values_at_t) for e in expressions])
            for variable_values_at_t in variable_values_at_each_timestep
        ]
    )
    return expr_traj


def _plot_from_sdp_relaxation(
    x_sol,
    planner,
    contact_scene,
    num_ctrl_points,
    plot_ctrl_points: bool = False,
    show_animation: bool = True,
    plot_rotation_curves: bool = False,
):
    decision_var_ctrl_points = planner.get_ctrl_points_for_all_decision_variables()
    # Need this order for the reshaping to be equivalent to the above expression
    decision_var_ctrl_points_vals = x_sol.reshape((-1, num_ctrl_points), order="F")

    if plot_ctrl_points:
        decision_var_trajs = BezierCurve.create_from_ctrl_points(
            decision_var_ctrl_points_vals
        ).ctrl_points.T  # (num_ctrl_points, num_variables)
        frames_per_sec = 0.3
    else:
        decision_var_trajs = BezierCurve.create_from_ctrl_points(
            decision_var_ctrl_points_vals
        ).eval_entire_interval()  # (num_ctrl_points, num_variables)
        frames_per_sec = 20

    # we use the first ctrl point to evaluate all expressions
    # must sort this similarly to SDP monomial basis created in relaxation
    decision_vars = np.array(
        sorted(planner.ctrl_points[0].variables, key=lambda x: x.get_id())
    )

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]
    body_colors = [TABLE_COLOR, BOX_COLOR, FINGER_COLOR]

    if plot_rotation_curves:
        names_of_rotation_vars = ["contact_1_cos_th", "contact_1_sin_th"]
        get_names = np.vectorize(lambda var: var.get_name())
        (idxs_of_rot_vars,) = np.where(
            np.isin(get_names(decision_vars), names_of_rotation_vars)
        )
        rot_trajs = decision_var_trajs[:, idxs_of_rot_vars]
        plot_cos_sine_trajs(rot_trajs)

    if show_animation:
        contact_forces_in_world_frame = [
            eval_expression_vector_with_traj_values(
                force, decision_vars, decision_var_trajs
            )
            for force in planner.ctrl_points[0].get_contact_forces_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        contact_positions_in_world_frame = [
            eval_expression_vector_with_traj_values(
                pos, decision_vars, decision_var_trajs
            )
            for pos in planner.ctrl_points[0].get_contact_positions_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        bodies_com_in_world_frame = [
            eval_expression_vector_with_traj_values(
                com, decision_vars, decision_var_trajs
            )
            for com in planner.ctrl_points[0].get_body_positions_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        # TODO clean up
        flattened_body_rotations = [
            rot.flatten().reshape((-1, 1))
            for rot in planner.ctrl_points[0].get_body_orientations()
        ]
        bodies_rot_in_world_frame = [
            eval_expression_vector_with_traj_values(
                rot, decision_vars, decision_var_trajs
            )
            for rot in flattened_body_rotations
        ]  # List[(N_steps, N_dims)]

        # Reshape to (2, 2) for interpolation
        bodies_rot_in_world_frame = [
            [row.reshape(2, 2) for row in body_traj]
            for body_traj in bodies_rot_in_world_frame
        ]

        # Interpolation
        START_TIME = 1.0
        END_TIME = 3.0
        DT = 0.01

        frames_per_sec = 1 / DT

        bodies_com_in_world_frame = [
            interpolate_w_first_order_hold(traj, START_TIME, END_TIME, DT)
            for traj in bodies_com_in_world_frame
        ]
        num_frames = len(bodies_com_in_world_frame[0])  # TODO change

        contact_positions_in_world_frame = [
            interpolate_w_first_order_hold(traj, START_TIME, END_TIME, DT)
            for traj in contact_positions_in_world_frame
        ]

        contact_forces_in_world_frame = [
            interpolate_w_first_order_hold(traj, START_TIME, END_TIME, DT)
            for traj in contact_forces_in_world_frame
        ]

        bodies_rot_in_world_frame = [
            interpolate_so2_using_slerp(traj, START_TIME, END_TIME, DT)
            for traj in bodies_rot_in_world_frame
        ]

        #######
        # Create relaxation error analysis
        #######

        eq_and_opposite_forces = contact_forces_in_world_frame[0:2]
        force_discrepancy = np.array(
            [
                sum([force[k] for force in eq_and_opposite_forces])
                for k in range(num_frames)
            ]
        )
        # create_forces_eq_and_opposite_analysis(force_discrepancy, num_ctrl_points)
        # plt.show()

        forces_acting_on_object = contact_forces_in_world_frame[
            1:
        ]  # TODO: Generalize this. This is just specific to this setup!
        GRAV_VEC = np.array([0, -9.81])
        sum_of_forces = np.vstack(
            [
                sum([force[k] for force in forces_acting_on_object]) + GRAV_VEC
                for k in range(num_frames)
            ]
        )

        contact_points_in_object_frame = contact_positions_in_world_frame[
            1:
        ]  # TODO: Generalize this. This is just specific to this setup!

        def _cross_2d(v1, v2):
            return (
                v1[0] * v2[1] - v1[1] * v2[0]
            )  # copied because the other one assumes the result is a np array, here it is just a scalar. Clean up!

        object_com = bodies_com_in_world_frame[1]
        sum_of_torques = np.vstack(
            [
                sum(
                    [
                        _cross_2d(pos[k] - object_com[k], force[k])
                        for pos, force in zip(
                            contact_points_in_object_frame, forces_acting_on_object
                        )
                    ]
                )
                for k in range(num_frames)
            ]
        )

        # create_static_equilibrium_analysis(
        #     sum_of_forces, sum_of_torques, num_ctrl_points  # type: ignore
        # )
        # plt.show()

        ########

        # Flatten bodies rot
        bodies_rot_in_world_frame = [
            np.vstack([R.flatten() for R in body_traj])
            for body_traj in bodies_rot_in_world_frame
        ]

        viz_com_points = [
            VisualizationPoint2d(com, GRAVITY_COLOR)  # type: ignore
            for com in bodies_com_in_world_frame
        ]

        viz_contact_positions = [
            VisualizationPoint2d(pos, CONTACT_COLOR)  # type: ignore
            for pos in contact_positions_in_world_frame
        ]
        viz_contact_forces = [
            VisualizationForce2d(pos, CONTACT_COLOR, force)  # type: ignore
            for pos, force in zip(
                contact_positions_in_world_frame, contact_forces_in_world_frame
            )
        ]

        # TODO: A bit hacky visualization of gravity
        box_com = bodies_com_in_world_frame[1]
        grav_vec = planner.ctrl_points[0].get_gravitational_forces_in_world_frame()[
            0
        ]  # I know there is only one gravity vec for this problem
        grav_force_traj = np.ones(box_com.shape) * grav_vec.T
        viz_gravitional_forces = [
            VisualizationForce2d(
                box_com,
                GRAVITY_COLOR,
                grav_force_traj,
            )
        ]

        # TODO: Bring back friction cones in visuzliation

        # friction_cone_angle = np.arctan(FRICTION_COEFF)
        # (
        #     fc_normals,
        #     fc_positions,
        #     fc_orientations,
        # ) = motion_plan.contact_point_friction_cones
        # viz_friction_cones = [
        #     VisualizationCone2d.from_ctrl_points(
        #         evaluate_np_expressions_array(pos, motion_plan.result),
        #         [
        #             motion_plan.result.GetSolution(R_ctrl_point)
        #             for R_ctrl_point in orientation
        #         ],
        #         normal_vec,
        #         friction_cone_angle,
        #     )
        #     for normal_vec, pos, orientation in zip(
        #         fc_normals, fc_positions, fc_orientations
        #     )
        # ]
        # viz_friction_cones_mirrored = [
        #     VisualizationCone2d.from_ctrl_points(
        #         evaluate_np_expressions_array(pos, motion_plan.result),
        #         [
        #             motion_plan.result.GetSolution(R_ctrl_point)
        #             for R_ctrl_point in orientation
        #         ],
        #         -normal_vec,
        #         friction_cone_angle,
        #     )
        #     for normal_vec, pos, orientation in zip(
        #         fc_normals, fc_positions, fc_orientations
        #     )
        # ]

        viz_polygons = [
            VisualizationPolygon2d.from_trajs(
                com,
                rotation,
                body,
                color,
            )
            for com, rotation, body, color in zip(
                bodies_com_in_world_frame,
                bodies_rot_in_world_frame,
                contact_scene.rigid_bodies,
                body_colors,
            )
        ]

        TARGET_POLYGON_IDX = 1
        TARGET_COLOR = COLORS["firebrick1"]
        target_polygon = VisualizationPolygon2d.from_trajs(
            bodies_com_in_world_frame[TARGET_POLYGON_IDX],
            bodies_rot_in_world_frame[TARGET_POLYGON_IDX],
            contact_scene.rigid_bodies[TARGET_POLYGON_IDX],
            TARGET_COLOR,
        )

        viz = Visualizer2d()
        viz.visualize(
            viz_contact_positions + viz_com_points,
            viz_contact_forces + viz_gravitional_forces,
            viz_polygons,
            frames_per_sec,
            target_polygon,
        )


def plan_polytope_flipup(
    num_vertices: int,
    contact_vertex: int,
    th_initial: Optional[float],
    th_target: Optional[float],
    sliding: bool = False,
) -> None:
    FRICTION_COEFF = 0.4
    TABLE_HEIGHT = 0.5
    TABLE_WIDTH = 2

    FINGER_HEIGHT = 0.1
    FINGER_WIDTH = 0.2

    POLYTOPE_MASS = 1

    # TODO:
    # 1. Make sure that contact locations for both forces always stay within the polytope
    # 2. contact force displacements should be calculated automatically

    polytope = EquilateralPolytope2d(
        actuated=False,
        name="polytope",
        mass=POLYTOPE_MASS,
        vertex_distance=0.2,
        num_vertices=num_vertices,
    )
    table = Box2d(
        actuated=True,
        name="table",
        mass=None,
        width=TABLE_WIDTH,
        height=TABLE_HEIGHT,
    )
    finger = Box2d(
        actuated=True,
        name="finger",
        mass=None,
        width=FINGER_WIDTH,
        height=FINGER_HEIGHT,
    )
    table_polytope = ContactPairDefinition(
        "contact_1",
        table,
        PolytopeContactLocation(ContactLocation.FACE, 1),
        polytope,
        PolytopeContactLocation(ContactLocation.VERTEX, contact_vertex),
        FRICTION_COEFF,
    )
    polytope_finger = ContactPairDefinition(
        "contact_2",
        polytope,
        PolytopeContactLocation(ContactLocation.FACE, 0),
        finger,
        PolytopeContactLocation(ContactLocation.FACE, 3),  # for line contact
        # PolytopeContactLocation(ContactLocation.VERTEX, 1),  # for vertex contact
        FRICTION_COEFF,
    )
    contact_scene = ContactScene2d(
        [table, polytope, finger],
        [table_polytope, polytope_finger],
        table,
    )

    if sliding:
        contact_modes = {
            "contact_1": ContactMode.SLIDING_LEFT,
            "contact_2": ContactMode.ROLLING,
        }
        lam_target = 0.6
    else:
        contact_modes = {
            "contact_1": ContactMode.ROLLING,
            "contact_2": ContactMode.ROLLING,
        }
        lam_target = None

    NUM_CTRL_POINTS = 3
    planner = ContactModeMotionPlanner(
        contact_scene,
        NUM_CTRL_POINTS,
        contact_modes,
    )
    if th_initial is not None:
        planner.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=0, theta=th_initial
        )
    if th_target is not None:
        planner.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, theta=th_target
        )
    planner.constrain_contact_position_at_ctrl_point(
        table_polytope, ctrl_point_idx=0, lam_target=0.4
    )
    if lam_target is not None:
        planner.constrain_contact_position_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, lam_target=lam_target
        )

    import time

    start = time.time()
    print("Starting to create SDP relaxation...")
    relaxed_prog, X, basis = create_sdp_relaxation(planner.prog)
    end = time.time()
    print(f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds")

    print("Solving...")
    start = time.time()
    result = Solve(relaxed_prog)
    end = time.time()
    print(f"Solved in {end - start} seconds")
    assert result.is_success()
    print("Success!")

    X_sol = result.GetSolution(X)
    x_sol = X_sol[1:, 0]
    breakpoint()

    _plot_from_sdp_relaxation(
        x_sol,
        planner,
        contact_scene,
        NUM_CTRL_POINTS,
        plot_ctrl_points=True,
        show_animation=True,
        plot_rotation_curves=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sliding", help="Use sliding", action="store_true", default=False
    )
    parser.add_argument(
        "--num_vertices", help="Number of vertices for polytope", type=int, default=3
    )
    parser.add_argument("--th_initial", help="Initial angle", type=float, default=0.0)
    parser.add_argument(
        "--th_target", help="Target angle", type=float, default=np.pi / 4
    )
    args = parser.parse_args()

    num_vertices = args.num_vertices
    th_initial = args.th_initial
    th_target = args.th_target
    sliding = args.sliding

    # Set some feasible conditions for the different polytopes
    if sliding:
        if num_vertices == 3:
            contact_vertex = 2
            th_initial = np.pi / 4 - 0.3
            th_target = np.pi / 4
        elif num_vertices == 4:
            contact_vertex = 2
            th_initial = -0.6
            th_target = -0.4
        elif num_vertices == 5:
            contact_vertex = 3
            th_initial = 0.1
            th_target = 0.2
        else:
            contact_vertex = 3
    else:
        if num_vertices == 3:
            contact_vertex = 2
            th_initial = 0.2
            th_target = np.pi / 4 + 0.4
        elif num_vertices == 4:
            contact_vertex = 2
            th_initial = -0.6
            th_target = np.pi / 6 - 0.6
        elif num_vertices == 5:
            contact_vertex = 3
            th_target = np.pi / 6
        else:
            contact_vertex = 3

    plan_polytope_flipup(num_vertices, contact_vertex, th_initial, th_target, sliding)
