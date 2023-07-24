
import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult

from geometry.rigid_body import RigidBody
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation, RigidBody2d
from tools.types import NpVariableArray


def plan_planar_pushing(experiment_number: int):
    if experiment_number == 0:
        rot_initial = 0
        rot_target = 0.5
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[0.2, 0.2]])
    else:
        raise NotImplementedError(f"Experiment {experiment_number} not implemented")

    num_knot_points = 4
    time_in_contact = 2
    time_moving = 0.5

    MASS = 1.0
    DIST_TO_CORNERS = 0.2
    num_vertices = 4

    slider = EquilateralPolytope2d(
        actuated=False,
        name="Slider",
        mass=MASS,
        vertex_distance=DIST_TO_CORNERS,
        num_vertices=num_vertices,
    )
    planner = GcsPlanarPushingPlanner(slider)
    planner.set_source_pose(pos_initial, rot_initial)
    planner.set_target_pose(pos_target, rot_target)

    contact_modes = {
        face_name(face_idx): PlanarPushingContactMode(
            object,
            num_knot_points=num_knot_points,
            contact_face_idx=face_idx,
            end_time=time_in_contact,
        )
        for face_idx in faces_to_consider
    }
    spectrahedrons = {
        key: mode.get_spectrahedron() for key, mode in contact_modes.items()
    }
    contact_vertices = {
        key: gcs.AddVertex(s, name=str(key)) for key, s in spectrahedrons.items()
    }

    # Add costs
    for mode, vertex in zip(contact_modes.values(), contact_vertices.values()):
        prog = mode.relaxed_prog
        for cost in prog.linear_costs():
            vars = vertex.x()[prog.FindDecisionVariableIndices(cost.variables())]
            a = cost.evaluator().a()
            vertex.AddCost(a.T.dot(vars))

    for v in source_connections:
        vertex = contact_vertices[face_name(v)]
        mode = contact_modes[face_name(v)]

        add_source_or_target_edge(
            vertex,
            source_vertex,
            mode,
            initial_config,
            gcs,
            source_or_target="source",
        )

    for v in target_connections:
        vertex = contact_vertices[face_name(v)]
        mode = contact_modes[face_name(v)]

        add_source_or_target_edge(
            vertex,
            target_vertex,
            mode,
            target_config,
            gcs,
            source_or_target="target",
        )

    num_knot_points_for_non_collision = 2

    chains = [
        GraphChain.from_contact_connection(incoming_idx, outgoing_idx, paths)
        for incoming_idx, outgoing_idx in face_connections
    ]
    for chain in chains:
        chain.create_contact_modes(
            object, num_knot_points_for_non_collision, time_moving
        )

    for chain in chains:
        incoming_vertex = contact_vertices[face_name(chain.start_contact_idx)]
        outgoing_vertex = contact_vertices[face_name(chain.end_contact_idx)]
        incoming_mode = contact_modes[face_name(chain.start_contact_idx)]
        outgoing_mode = contact_modes[face_name(chain.end_contact_idx)]
        chain.create_edges(
            incoming_vertex, outgoing_vertex, incoming_mode, outgoing_mode, gcs
        )
        chain.add_costs()

    # Collect all modes and vertices in one big lookup table for trajectory retrieval
    all_modes = contact_modes.copy()
    all_vertices = contact_vertices.copy()

    for chain in chains:
        mode_chain = chain.get_all_non_collision_modes()
        vertex_chain = chain.get_all_non_collision_vertices()

        for modes, vertices in zip(mode_chain, vertex_chain):
            for mode, vertex in zip(modes, vertices):
                all_modes[mode.name] = mode  # type: ignore
                all_vertices[mode.name] = vertex

    graphviz = gcs.GetGraphvizString()
    data = pydot.graph_from_dot_data(graphviz)[0]
    data.write_svg("graph.svg")

    options = opt.GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.solver_options = SolverOptions()
    options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # options.solver_options.SetOption(
    #     MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3
    # )
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
    # options.solver_options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
    # options.solver_options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)
    if options.convex_relaxation is True:
        options.preprocessing = True  # TODO Do I need to deal with this?
        options.max_rounded_paths = 1
    result = gcs.SolveShortestPath(source_vertex, target_vertex, options)

    assert result.is_success()
    print("Success!")

    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [result.GetSolution(p) for p in flow_variables]
    active_edges = [
        edge for edge, flow in zip(gcs.Edges(), flow_results) if flow >= 0.55
    ]

    full_path = _find_path_to_target(active_edges, target_vertex, source_vertex)
    vertex_names_on_path = [
        v.name() for v in full_path if v.name() not in ["source", "target"]
    ]

    vertices_on_path = [all_vertices[name] for name in vertex_names_on_path]
    modes_on_path = [all_modes[name] for name in vertex_names_on_path]

    mode_vars_on_path = [
        mode.get_vars_from_gcs_vertex(vertex)
        for mode, vertex in zip(modes_on_path, vertices_on_path)
    ]
    vals = [mode.eval_result(result) for mode in mode_vars_on_path]

    DT = 0.01
    interpolate = True
    R_traj = sum(
        [val.get_R_traj(DT, interpolate=interpolate) for val in vals],
        [],
    )
    com_traj = np.vstack(
        [val.get_p_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    force_traj = np.vstack(
        [val.get_f_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    contact_pos_traj = np.vstack(
        [val.get_p_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )

    traj_length = len(R_traj)
    breakpoint()

    compute_violation = False
    if compute_violation:
        # NOTE: SHOULD BE MOVED!
        # compute quasi-static dynamic violation
        def _cross_2d(v1, v2):
            return (
                v1[0] * v2[1] - v1[1] * v2[0]
            )  # copied because the other one assumes the result is a np array, here it is just a scalar. clean up!

        quasi_static_violation = []
        for k in range(traj_length - 1):
            v_WB = (com_traj[k + 1] - com_traj[k]) / DT
            R_dot = (R_traj[k + 1] - R_traj[k]) / DT
            R = R_traj[k]
            omega_WB = R_dot.dot(R.T)[1, 0]
            f_c_B = force_traj[k]
            p_c_B = contact_pos_traj[k]
            com = com_traj[k]

            # Contact torques
            tau_c_B = _cross_2d(p_c_B - com, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions

            R_padded = np.zeros((3, 3))
            R_padded[2, 2] = 1
            R_padded[0:2, 0:2] = R
            violation = x_dot - R_padded.dot(A).dot(wrench)
            quasi_static_violation.append(violation)

        quasi_static_violation = np.vstack(quasi_static_violation)
        create_quasistatic_pushing_analysis(quasi_static_violation, num_knot_points)
        plt.show()

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]

    flattened_rotation = np.vstack([R.flatten() for R in R_traj])
    box_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        object,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        object,
        TARGET_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = len(R_traj) / DT
    viz.visualize(
        [contact_point_viz],
        [contact_force_viz],
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        help="Which experiment to run",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    experiment_number = args.exp

    plan_planar_pushing()
