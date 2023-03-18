import argparse
from typing import Optional

import numpy as np
from pydrake.solvers import Solve

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.bezier import BezierCurve
from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.contact_pair_2d import ContactPairDefinition
from geometry.two_d.contact.contact_scene_2d import ContactScene2d
from geometry.two_d.contact.types import ContactLocation, ContactMode
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from planning.contact_mode_motion_planner import ContactModeMotionPlanner
from tools.types import NpExpressionArray, NpVariableArray
from tools.utils import evaluate_np_expressions_array
from visualize.colors import COLORS
from visualize.visualizer_2d import (
    VisualizationCone2d,
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def plan_polytope_flipup(
    num_vertices: int,
    contact_vertex: int,
    th_initial: Optional[float],
    th_target: Optional[float],
    sliding: bool = False,
) -> None:
    FRICTION_COEFF = 1.0
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
        PolytopeContactLocation(ContactLocation.VERTEX, 1),
        FRICTION_COEFF,
    )
    contact_scene = ContactScene2d(
        [table, polytope, finger],
        [table_polytope, polytope_finger],
        table,
    )

    # TODO: this should be cleaned up
    MAX_FORCE = POLYTOPE_MASS * 9.81 * 2.0  # only used for mccorimick constraints
    CONSTANT = 2
    variable_bounds = {
        "contact_1_polytope_c_n": (0.0, MAX_FORCE),
        "contact_1_polytope_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_c_n": (0.0, MAX_FORCE),
        "contact_1_table_c_f": (
            -FRICTION_COEFF * MAX_FORCE,
            FRICTION_COEFF * MAX_FORCE,
        ),
        "contact_1_table_lam": (0.0, 1.0),
        "contact_1_sin_th": (-1, 1),
        "contact_1_cos_th": (-1, 1),
        "contact_2_polytope_lam": (0.0, 1.0),
        "contact_2_polytope_c_n": (0, MAX_FORCE / CONSTANT),
        "contact_2_polytope_c_f": (0, MAX_FORCE / CONSTANT),
        "contact_2_sin_th": (-1, 1),
        "contact_2_cos_th": (-1, 1),
        "contact_2_finger_c_n": (0.0, MAX_FORCE / CONSTANT),
        "contact_2_finger_c_f": (
            -FRICTION_COEFF * MAX_FORCE / CONSTANT,
            FRICTION_COEFF * MAX_FORCE / CONSTANT,
        ),
    }

    if sliding:
        contact_modes = {
            "contact_1": ContactMode.ROLLING,
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
    motion_plan = ContactModeMotionPlanner(
        contact_scene,
        NUM_CTRL_POINTS,
        contact_modes,
        variable_bounds,
    )
    if th_initial is not None:
        motion_plan.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=0, theta=th_initial
        )
    if th_target is not None:
        motion_plan.constrain_orientation_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, theta=th_target
        )
    motion_plan.constrain_contact_position_at_ctrl_point(
        table_polytope, ctrl_point_idx=0, lam_target=0.5
    )
    if lam_target is not None:
        motion_plan.constrain_contact_position_at_ctrl_point(
            table_polytope, ctrl_point_idx=NUM_CTRL_POINTS - 1, lam_target=lam_target
        )

    import time

    start = time.time()
    print("Starting to create SDP relaxation...")
    relaxed_prog, X = create_sdp_relaxation(motion_plan.prog)
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
    decision_var_ctrl_points = motion_plan.get_ctrl_points_for_all_decision_variables()
    # Need this order for the reshaping to be equivalent to the above expression
    decision_var_ctrl_points_vals = x_sol.reshape((-1, NUM_CTRL_POINTS), order="F")
    decision_var_trajs = BezierCurve.create_from_ctrl_points(
        decision_var_ctrl_points_vals
    ).eval_entire_interval()  # (num_steps, num_variables)

    decision_vars = motion_plan.ctrl_points[
        0
    ].variables  # we use the first ctrl point to evaluate all expressions

    # TODO move
    import numpy.typing as npt
    import pydrake.symbolic as sym

    def eval_expressions_at_traj(
        exprs: NpExpressionArray, vars: NpVariableArray, trajs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # NOTE! Likely very slow
        envs = [
            {var: val for var, val in zip(vars, vals)} for vals in trajs
        ]  # one env per timestep
        expr_traj = np.vstack(
            [np.array([e.item().Evaluate(env) for e in exprs]) for env in envs]
        )
        return expr_traj

    if True:
        CONTACT_COLOR = COLORS["dodgerblue4"]
        GRAVITY_COLOR = COLORS["blueviolet"]
        BOX_COLOR = COLORS["aquamarine4"]
        TABLE_COLOR = COLORS["bisque3"]
        FINGER_COLOR = COLORS["firebrick3"]
        body_colors = [TABLE_COLOR, BOX_COLOR, FINGER_COLOR]

        contact_forces_in_world_frame = [
            eval_expressions_at_traj(force, decision_vars, decision_var_trajs)
            for force in motion_plan.ctrl_points[0].get_contact_forces_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        contact_positions_in_world_frame = [
            eval_expressions_at_traj(pos, decision_vars, decision_var_trajs)
            for pos in motion_plan.ctrl_points[0].get_contact_positions_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        bodies_com_in_world_frame = [
            eval_expressions_at_traj(com, decision_vars, decision_var_trajs)
            for com in motion_plan.ctrl_points[0].get_body_positions_in_world_frame()
        ]  # List[(N_steps, N_dims)]

        # TODO clean up
        flattened_body_rotations = [
            rot.flatten().reshape((-1, 1))
            for rot in motion_plan.ctrl_points[0].get_body_orientations()
        ]
        bodies_rot_in_world_frame = [
            eval_expressions_at_traj(rot, decision_vars, decision_var_trajs)
            for rot in flattened_body_rotations
        ]  # List[(N_steps, N_dims)]

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

        # box_com_ctrl_points = bodies_com_ctrl_points[1]
        # # TODO: should not depend explicitly on box
        # viz_gravitional_forces = [
        #     VisualizationForce2d
        #         box_com_ctrl_points,
        #         evaluate_np_expressions_array(force_ctrl_points, motion_plan.result),
        #         GRAVITY_COLOR,
        #     )
        #     for force_ctrl_points in motion_plan.gravitational_forces_in_world_frame
        # ]
        #
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

        breakpoint()
        viz = Visualizer2d()
        viz.visualize(
            viz_contact_positions + viz_com_points,
            viz_contact_forces,
            viz_polygons,
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
            th_target = np.pi / 4
        elif num_vertices == 4:
            contact_vertex = 2
            th_initial = -np.pi / 4
            th_target = 0
        elif num_vertices == 5:
            contact_vertex = 3
            th_target = np.pi / 5
        else:
            contact_vertex = 3
    else:
        if num_vertices == 3:
            contact_vertex = 2
        elif num_vertices == 4:
            contact_vertex = 2
            th_target = np.pi / 6
        elif num_vertices == 5:
            contact_vertex = 3
            th_target = np.pi / 6
        else:
            contact_vertex = 3

    plan_polytope_flipup(num_vertices, contact_vertex, th_initial, th_target, sliding)