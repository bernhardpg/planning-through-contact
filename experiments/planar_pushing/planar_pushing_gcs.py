import argparse
import time
from typing import List, Literal, NamedTuple, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.math import eq
from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    MixedIntegerBranchAndBound,
    Solve,
)
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.two_d.contact.types import ContactLocation
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpVariableArray
from visualize.analysis import create_quasistatic_pushing_analysis
from visualize.colors import COLORS
from visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

T = TypeVar("T")


def forward_differences(
    vars: List[NpExpressionArray], dt: float
) -> List[NpExpressionArray]:
    # TODO: It is cleaner to implement this using a forward diff matrix, but as a first step I do this the simplest way
    forward_diffs = [
        (var_next - var_curr) / dt for var_curr, var_next in zip(vars[0:-1], vars[1:])
    ]
    return forward_diffs


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
    breakpoint()
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


# TODO: should probably have a better name
class ModeVars(NamedTuple):
    lam: NpVariableArray
    normal_forces: NpVariableArray
    friction_forces: NpVariableArray
    cos_th: NpVariableArray
    sin_th: NpVariableArray
    p_WBs: List[NpVariableArray]


class PlanarPushingContactMode:
    def __init__(
        self,
        contact_face_idx: int,
        num_knot_points: int = 4,
        end_time: float = 3,
        th_initial: Optional[float] = None,
        th_target: Optional[float] = None,
        pos_initial: Optional[npt.NDArray[np.float64]] = None,
        pos_target: Optional[npt.NDArray[np.float64]] = None,
    ):
        self.num_knot_points = num_knot_points
        num_vertices = 4

        FRICTION_COEFF = 0.5
        G = 9.81
        MASS = 1.0
        DIST_TO_CORNERS = 0.2

        f_max = FRICTION_COEFF * G * MASS
        tau_max = f_max * DIST_TO_CORNERS

        A = np.diag(
            [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
        )  # Ellipsoidal Limit surface approximation

        dt = end_time / num_knot_points
        self.dt = dt

        prog = MathematicalProgram()

        self.polytope = EquilateralPolytope2d(
            actuated=False,
            name="Slider",
            mass=MASS,
            vertex_distance=DIST_TO_CORNERS,
            num_vertices=num_vertices,
        )

        contact_face = PolytopeContactLocation(
            pos=ContactLocation.FACE, idx=contact_face_idx
        )

        # Contact positions
        lams = prog.NewContinuousVariables(num_knot_points, "lam")
        for lam in lams:
            prog.AddLinearConstraint(lam >= 0)
            prog.AddLinearConstraint(lam <= 1)

        self.pv1, self.pv2 = self.polytope.get_proximate_vertices_from_location(
            contact_face
        )
        p_c_Bs = [lam * self.pv1 + (1 - lam) * self.pv2 for lam in lams]

        # Contact forces
        normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            self.normal_vec,
            self.tangent_vec,
        ) = self.polytope.get_norm_and_tang_vecs_from_location(contact_face)
        f_c_Bs = [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(normal_forces, friction_forces)
        ]

        # Rotations
        cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")
        R_WBs = [
            np.array([[cos, sin], [-sin, cos]]) for cos, sin in zip(cos_ths, sin_ths)
        ]

        # Box position relative to world frame
        p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")
        p_WBs = [np.array([x, y]) for x, y in zip(p_WB_xs, p_WB_ys)]

        # Compute velocities
        v_WBs = forward_differences(p_WBs, dt)
        cos_th_dots = forward_differences(cos_ths, dt)
        sin_th_dots = forward_differences(sin_ths, dt)
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
        ]
        v_c_Bs = forward_differences(p_c_Bs, dt)

        # In 2D, omega_z = theta_dot will be at position (0,1) in R_dot * R'
        omega_WBs = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(R_WBs, R_WB_dots)]

        # SO(2) constraints
        for c, s in zip(cos_ths, sin_ths):
            prog.AddConstraint(c**2 + s**2 == 1)

        # # Friction cone constraints
        for c_n in normal_forces:
            prog.AddLinearConstraint(c_n >= 0)
        for c_n, c_f in zip(normal_forces, friction_forces):
            prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Quasi-static dynamics
        for k in range(num_knot_points - 1):
            v_WB = v_WBs[k]
            omega_WB = omega_WBs[k]
            f_c_B = (f_c_Bs[k] + f_c_Bs[k + 1]) / 2
            p_c_B = (p_c_Bs[k] + p_c_Bs[k + 1]) / 2

            R = np.zeros((3, 3), dtype="O")
            R[2, 2] = 1
            R[0:2, 0:2] = R_WBs[
                k
            ]  # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"

            # Contact torques
            tau_c_B = cross_2d(p_c_B, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions

            quasi_static_dynamic_constraint = eq(x_dot, A.dot(wrench))
            for row in quasi_static_dynamic_constraint:
                prog.AddConstraint(row)

        # Ensure sticking on the contact point
        for v_c_B in v_c_Bs:
            prog.AddLinearConstraint(
                eq(v_c_B, 0)
            )  # no velocity on contact points in body frame

        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in v_WBs])
        sq_angular_vels = sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
            ]
        )
        prog.AddQuadraticCost(sq_linear_vels)
        prog.AddQuadraticCost(sq_angular_vels)

        def create_R(th: float) -> npt.NDArray[np.float64]:
            return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        # Initial conditions (only first and last vertex will have this)
        if th_initial is not None:
            R_WB_I = create_R(th_initial)
            prog.AddLinearConstraint(eq(R_WBs[0], R_WB_I))
        if th_target is not None:
            R_WB_T = create_R(th_target)
            prog.AddLinearConstraint(eq(R_WBs[-1], R_WB_T))
        if pos_initial is not None:
            prog.AddLinearConstraint(eq(p_WBs[0], pos_initial))
        if pos_target is not None:
            prog.AddLinearConstraint(eq(p_WBs[-1], pos_target))

        start = time.time()
        print("Starting to create SDP relaxation...")
        self.relaxed_prog, self.X, _ = create_sdp_relaxation(prog)
        end = time.time()
        print(
            f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds"
        )

        # Retrieve original variables from X
        # We have n = 7*num_knot_points variables
        # This gives X dimensions (n+1,n+1)
        # where the 1 is added because the first element of x is 1.
        # This gives ((n+1)^2 - (n+1))/2 + (n+1) decision variables
        # (all entries - diagonal entries)/2 (because X symmetric) + add back diagonal)
        x = self.X[1:, 0]
        self.lam = x[0:num_knot_points]
        self.normal_forces = x[num_knot_points : 2 * num_knot_points]
        self.friction_forces = x[2 * num_knot_points : 3 * num_knot_points]
        self.cos_th = x[3 * num_knot_points : 4 * num_knot_points]
        self.sin_th = x[4 * num_knot_points : 5 * num_knot_points]
        self.p_WB_xs = x[5 * num_knot_points : 6 * num_knot_points]
        self.p_WB_ys = x[6 * num_knot_points : 7 * num_knot_points]

        self.num_variables = 7 * num_knot_points + 1  # TODO: 7 is hardcoded, fix this

    def get_spectrahedron(self) -> opt.Spectrahedron:
        # Variables should be the stacked columns of the lower
        # triangular part of the symmetric matrix from relaxed_prog, see
        # https://robotlocomotion.github.io/doxygen_cxx/classdrake_1_1solvers_1_1_mathematical_program.html#a8f718351922bc149cb6e7fa6d82288a5
        spectrahedron = opt.Spectrahedron(self.relaxed_prog)
        return spectrahedron

    def get_vars_from_gcs_vertex(
        self, gcs_vertex: opt.GraphOfConvexSets.Vertex
    ) -> ModeVars:
        x = gcs_vertex.x()[1 : self.num_variables + 2]

        lam = x[0 : self.num_knot_points]
        normal_forces = x[self.num_knot_points : 2 * self.num_knot_points]
        friction_forces = x[2 * self.num_knot_points : 3 * self.num_knot_points]
        cos_th = x[3 * self.num_knot_points : 4 * self.num_knot_points]
        sin_th = x[4 * self.num_knot_points : 5 * self.num_knot_points]
        p_WB_xs = x[5 * self.num_knot_points : 6 * self.num_knot_points]
        p_WB_ys = x[6 * self.num_knot_points : 7 * self.num_knot_points]

        p_WBs = [np.expand_dims(np.array([x, y]), 1) for x, y in zip(p_WB_xs, p_WB_ys)]

        return ModeVars(
            lam,
            normal_forces,
            friction_forces,
            cos_th,
            sin_th,
            p_WBs,
        )

    def solve(self):
        print("Solving...")
        start = time.time()
        result = Solve(self.relaxed_prog)
        end = time.time()
        print(f"Solved in {end - start} seconds")
        assert result.is_success()
        print("Success!")

        x_val = result.GetSolution(self.X[1:, 0])
        lam_vals = result.GetSolution(self.lam)
        normal_forces_vals = result.GetSolution(self.normal_forces)
        friction_forces_vals = result.GetSolution(self.friction_forces)
        cos_th_vals = result.GetSolution(self.cos_th)
        sin_th_vals = result.GetSolution(self.sin_th)
        p_WB_xs_vals = result.GetSolution(self.p_WB_xs)
        p_WB_ys_vals = result.GetSolution(self.p_WB_ys)

        return (
            lam_vals,
            normal_forces_vals,
            friction_forces_vals,
            cos_th_vals,
            sin_th_vals,
            p_WB_xs_vals,
            p_WB_ys_vals,
        )


def get_vals(result: MathematicalProgramResult, mode_vars: List[ModeVars]) -> ModeVars:
    lam_vals = [result.GetSolution(mode.lam) for mode in mode_vars]
    normal_force_vals = [result.GetSolution(mode.normal_forces) for mode in mode_vars]
    friction_forces_vals = [
        result.GetSolution(mode.friction_forces) for mode in mode_vars
    ]
    cos_th_vals = [result.GetSolution(mode.cos_th) for mode in mode_vars]
    sin_th_vals = [result.GetSolution(mode.sin_th) for mode in mode_vars]
    p_WB_vals = [
        [result.GetSolution(p_WB) for p_WB in mode.p_WBs] for mode in mode_vars
    ]
    return ModeVars(
        lam_vals,
        normal_force_vals,
        friction_forces_vals,
        cos_th_vals,
        sin_th_vals,
        p_WB_vals,
    )


def _create_obj_config(
    pos: npt.NDArray[np.float64], th: float
) -> npt.NDArray[np.float64]:
    """
    Concatenates unactauted object config for source and target vertex
    """
    obj_pose = np.concatenate([pos.flatten(), [np.cos(th), np.sin(th)]])
    return obj_pose


def add_source_or_target_edge(
    vertex: opt.GraphOfConvexSets.Vertex,
    point_vertex: opt.GraphOfConvexSets.Vertex,
    vertex_mode: PlanarPushingContactMode,
    obj_config: npt.NDArray[np.float64],
    gcs: opt.GraphOfConvexSets,
    source_or_target: Literal["source", "target"],
) -> opt.GraphOfConvexSets.Edge:
    vars = vertex_mode.get_vars_from_gcs_vertex(vertex)
    if source_or_target == "source":
        edge = gcs.AddEdge(point_vertex, vertex)
        pos_vars = vars.p_WBs[0]
        cos_var = vars.cos_th[0]
        sin_var = vars.sin_th[0]
    else:  # target
        edge = gcs.AddEdge(vertex, point_vertex)
        pos_vars = vars.p_WBs[-1]
        cos_var = vars.cos_th[-1]
        sin_var = vars.sin_th[-1]

    continuity_constraints = eq(pos_vars.flatten(), obj_config[0:2])
    for c in continuity_constraints.flatten():
        edge.AddConstraint(c)

    continuity_constraint = cos_var == obj_config[2]
    edge.AddConstraint(continuity_constraint)

    continuity_constraint = sin_var == obj_config[3]
    edge.AddConstraint(continuity_constraint)

    return edge


def add_edge_with_continuity_constraint(
    u: opt.GraphOfConvexSets.Vertex,
    v: opt.GraphOfConvexSets.Vertex,
    u_mode: PlanarPushingContactMode,
    v_mode: PlanarPushingContactMode,
    gcs: opt.GraphOfConvexSets,
) -> opt.GraphOfConvexSets.Edge:
    edge = gcs.AddEdge(u, v)

    u_vars = u_mode.get_vars_from_gcs_vertex(u)
    v_vars = v_mode.get_vars_from_gcs_vertex(v)

    continuity_constraints = eq(u_vars.p_WBs[-1], v_vars.p_WBs[0])
    for c in continuity_constraints.flatten():
        edge.AddConstraint(c)

    continuity_constraints = eq(u_vars.cos_th[-1], v_vars.cos_th[0])
    for c in continuity_constraints.flatten():
        edge.AddConstraint(c)

    continuity_constraints = eq(u_vars.sin_th[-1], v_vars.sin_th[0])
    for c in continuity_constraints.flatten():
        edge.AddConstraint(c)

    return edge


def plan_planar_pushing():
    th_initial = 0
    th_target = 0.5
    pos_initial = np.array([[0.0, 0.5]])
    pos_target = np.array([[-0.4, 0.2]])
    end_time = 3

    initial_config = _create_obj_config(pos_initial, th_initial)
    target_config = _create_obj_config(pos_target, th_target)

    source = opt.Point(initial_config)
    target = opt.Point(target_config)

    face_0_contact = PlanarPushingContactMode(
        num_knot_points=6,
        contact_face_idx=0,
        end_time=end_time,
    )

    face_1_contact = PlanarPushingContactMode(
        num_knot_points=6,
        contact_face_idx=1,
        end_time=end_time,
    )

    gcs = opt.GraphOfConvexSets()
    source_vertex = gcs.AddVertex(source)
    target_vertex = gcs.AddVertex(target)
    face_0_vertex = gcs.AddVertex(face_0_contact.get_spectrahedron())
    face_1_vertex = gcs.AddVertex(face_1_contact.get_spectrahedron())

    modes = [face_0_contact, face_1_contact]
    vertices = [face_0_vertex, face_1_vertex]
    for mode, vertex in zip(modes, vertices):
        prog = mode.relaxed_prog
        for cost in prog.linear_costs():
            vars = vertex.x()[prog.FindDecisionVariableIndices(cost.variables())]
            a = cost.evaluator().a()
            vertex.AddCost(a.T.dot(vars))

    add_edge_with_continuity_constraint(
        face_0_vertex, face_1_vertex, face_0_contact, face_1_contact, gcs
    )

    add_source_or_target_edge(
        face_0_vertex,
        source_vertex,
        face_0_contact,
        initial_config,
        gcs,
        source_or_target="source",
    )

    add_source_or_target_edge(
        face_1_vertex,
        target_vertex,
        face_1_contact,
        target_config,
        gcs,
        source_or_target="target",
    )

    options = opt.GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    if options.convex_relaxation is True:
        options.preprocessing = True  # TODO Do I need to deal with this?
        options.max_rounded_paths = 10
    result = gcs.SolveShortestPath(source_vertex, target_vertex, options)
    assert result.is_success()

    # TODO: Pick this sequence from optimal path
    all_mode_vars = [
        mode.get_vars_from_gcs_vertex(vertex) for mode, vertex in zip(modes, vertices)
    ]
    vals = get_vals(result, all_mode_vars)
    lam_vals = vals.lam
    normal_forces_vals = vals.normal_forces
    friction_forces_vals = vals.friction_forces
    cos_th_vals = np.concatenate(vals.cos_th)
    sin_th_vals = np.concatenate(vals.sin_th)
    p_WBs_vals = np.concatenate(vals.p_WBs)

    normal_vecs = [mode.normal_vec for mode in modes]
    tangent_vecs = [mode.tangent_vec for mode in modes]
    pv1s = [mode.pv1 for mode in modes]
    pv2s = [mode.pv2 for mode in modes]

    polytope = face_0_contact.polytope

    # Reconstruct quantities
    p_c_Bs_vals = np.concatenate(
        [
            np.hstack([lam * pv1 + (1 - lam) * pv2 for lam in lams]).T
            for (pv1, pv2), lams in zip(zip(pv1s, pv2s), lam_vals)
        ]
    )
    f_c_Bs_vals = np.concatenate(
        [
            np.hstack(
                [c_n * normal_vec + c_f * tangent_vec for c_n, c_f in zip(c_ns, c_fs)]
            ).T
            for (normal_vec, tangent_vec), (c_ns, c_fs) in zip(
                zip(normal_vecs, tangent_vecs),
                zip(normal_forces_vals, friction_forces_vals),
            )
        ]
    )
    R_WBs_vals = [
        np.array([[cos, -sin], [sin, cos]])
        for cos, sin in zip(cos_th_vals, sin_th_vals)
    ]

    contact_pos_in_W = np.vstack(
        [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(p_WBs_vals, R_WBs_vals, p_c_Bs_vals)
        ]
    )

    contact_force_in_W = np.vstack(
        [R_WB.dot(f_c_B) for R_WB, f_c_B in zip(R_WBs_vals, f_c_Bs_vals)]
    )

    DT = 0.01

    interpolate = False
    if interpolate:
        # Interpolate quantities
        R_traj = interpolate_so2_using_slerp(R_WBs_vals, 0, end_time, DT)
        com_traj = interpolate_w_first_order_hold(p_WBs_vals, 0, end_time, DT)
        force_traj = interpolate_w_first_order_hold(contact_force_in_W, 0, end_time, DT)
        contact_pos_traj = interpolate_w_first_order_hold(
            contact_pos_in_W, 0, end_time, DT
        )
    else:
        R_traj = R_WBs_vals
        com_traj = p_WBs_vals
        force_traj = contact_force_in_W
        contact_pos_traj = contact_pos_in_W

    traj_length = len(R_traj)

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
        polytope,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        polytope,
        TARGET_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = len(R_traj) / end_time
    viz.visualize(
        [com_points_viz, contact_point_viz],
        [contact_force_viz],
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
    )


def plan_planar_pushing_one_mode():
    th_initial = 0
    th_target = 0.5
    pos_initial = np.array([[0.0, 0.5]])
    pos_target = np.array([[-0.1, 0.2]])
    end_time = 3

    mode = PlanarPushingContactMode(
        contact_face_idx=0,
        end_time=end_time,
        th_initial=th_initial,
        th_target=th_target,
        pos_initial=pos_initial,
        pos_target=pos_target,
    )

    # Extract valus
    (
        lam_vals,
        normal_forces_vals,
        friction_forces_vals,
        cos_th_vals,
        sin_th_vals,
        p_WB_xs_vals,
        p_WB_ys_vals,
    ) = mode.solve()

    # Reconstruct quantities
    p_c_Bs_vals = np.hstack(
        [lam * mode.pv1 + (1 - lam) * mode.pv2 for lam in lam_vals]
    ).T
    f_c_Bs_vals = np.hstack(
        [
            c_n * mode.normal_vec + c_f * mode.tangent_vec
            for c_n, c_f in zip(normal_forces_vals, friction_forces_vals)
        ]
    ).T
    R_WBs_vals = [
        np.array([[cos, -sin], [sin, cos]])
        for cos, sin in zip(cos_th_vals, sin_th_vals)
    ]

    p_WBs_vals = np.vstack(
        [np.array([x, y]) for x, y in zip(p_WB_xs_vals, p_WB_ys_vals)]
    )  # TODO: rename!

    contact_pos_in_W = np.vstack(
        [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(p_WBs_vals, R_WBs_vals, p_c_Bs_vals)
        ]
    )

    contact_force_in_W = np.vstack(
        [R_WB.dot(f_c_B) for R_WB, f_c_B in zip(R_WBs_vals, f_c_Bs_vals)]
    )
    breakpoint()

    DT = 0.01

    # Interpolate quantities
    R_traj = interpolate_so2_using_slerp(R_WBs_vals, 0, end_time, DT)
    com_traj = interpolate_w_first_order_hold(p_WBs_vals, 0, end_time, DT)
    force_traj = interpolate_w_first_order_hold(contact_force_in_W, 0, end_time, DT)
    contact_pos_traj = interpolate_w_first_order_hold(contact_pos_in_W, 0, end_time, DT)

    traj_length = len(R_traj)

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
        mode.polytope,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        com_traj,
        flattened_rotation,
        mode.polytope,
        TARGET_COLOR,
    )

    com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
    contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
    contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

    viz = Visualizer2d()
    FRAMES_PER_SEC = len(R_traj) / end_time
    viz.visualize(
        [com_points_viz, contact_point_viz],
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
