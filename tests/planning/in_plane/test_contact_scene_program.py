import os

import numpy as np
import numpy.typing as npt
import pytest
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    Solve,
    SolverOptions,
)
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from planning_through_contact.geometry.bezier import BezierCurve
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    ContactMode,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.in_plane.contact_pair import (
    ContactPairDefinition,
)
from planning_through_contact.geometry.in_plane.contact_scene import (
    ContactSceneDefinition,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.in_plane.contact_scene_program import (
    ContactSceneProgram,
)
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
DEBUG = True


@pytest.fixture
def contact_scene_def() -> ContactSceneDefinition:
    box = RigidBody("box", Box2d(0.15, 0.15), mass=0.1)
    robot = RigidBody("robot", Box2d(0.05, 0.03), mass=0.03, is_actuated=True)
    loc_robot = PolytopeContactLocation(ContactLocation.FACE, 2)
    loc_box_robot = PolytopeContactLocation(ContactLocation.FACE, 3)

    table = RigidBody("table", Box2d(0.5, 0.03), mass=0.03, is_actuated=True)
    loc_table = PolytopeContactLocation(ContactLocation.FACE, 0)
    loc_box_table = PolytopeContactLocation(ContactLocation.VERTEX, 2)

    box_table = ContactPairDefinition("box_table", box, loc_box_table, table, loc_table)
    box_robot = ContactPairDefinition("box_robot", box, loc_box_robot, robot, loc_robot)

    scene_def = ContactSceneDefinition(
        [table, box, robot],
        [box_table, box_robot],
        table,
    )
    return scene_def


def test_contact_scene_program_construction_rolling(
    contact_scene_def: ContactSceneDefinition,
) -> None:
    num_ctrl_points = 4
    contact_modes = {
        contact_scene_def.contact_pairs[0]: ContactMode.ROLLING,
        contact_scene_def.contact_pairs[1]: ContactMode.ROLLING,
    }

    scene_prob = ContactSceneProgram(contact_scene_def, num_ctrl_points, contact_modes)
    prog = scene_prob.prog

    # point contact: cos_th, sin_th, f_x, f_y, c_n, c_f, rel positions (4 vals)
    # line contact: c_n, c_f * 2
    # x num_ctrl_points
    # + lam for each pair
    assert len(prog.decision_variables()) == (10 + 4) * num_ctrl_points + 2

    # Friction cone for each control point
    assert len(prog.linear_constraints()) == 1 * num_ctrl_points

    # one for lams, one for trig terms, for each control point
    assert len(prog.bounding_box_constraints()) == 2 * num_ctrl_points

    # Should not be any generic constraints
    assert len(prog.generic_constraints()) == 0

    # 1 * for so(2) constraint
    # 1 * for torque balance for each control point
    # 2 (dims) * equal contact points for each control point
    # 2 * 2 (dims) * equal rel pos for each control point
    # 2 * 2 (dims) * equal opposite forces for each control point
    assert len(prog.quadratic_constraints()) == 12 * num_ctrl_points

    # 1 * force balance for each control point
    # 1 * 2 (dims) for equal contact points
    assert len(prog.linear_equality_constraints()) == 3 * num_ctrl_points

    assert len(prog.linear_costs()) == 0
    assert len(prog.generic_costs()) == 0

    unactuated_body = scene_prob.contact_scene_def.unactuated_bodies[0]
    assert (
        len(scene_prob._get_sq_rot_param_dots(unactuated_body)) == num_ctrl_points - 1
    )
    assert len(scene_prob._get_sq_body_vels(unactuated_body)) == num_ctrl_points - 1

    # two cost term per velocity, one for ang and one for trans vel
    assert len(prog.quadratic_costs()) == 2 * (num_ctrl_points - 1)

    for cost in prog.quadratic_costs():
        # cos, sin for both knot points involved in computing vel
        # or
        # p_x, p_y for both knot points involved in computing vel
        assert len(cost.variables()) == 2 * 2


def test_contact_scene_program_construction_sliding(
    contact_scene_def: ContactSceneDefinition,
) -> None:
    num_ctrl_points = 4
    contact_modes = {
        contact_scene_def.contact_pairs[0]: ContactMode.SLIDING_LEFT,
        contact_scene_def.contact_pairs[1]: ContactMode.SLIDING_LEFT,
    }

    scene_prob = ContactSceneProgram(contact_scene_def, num_ctrl_points, contact_modes)

    prog = scene_prob.prog

    pos = scene_prob._get_nonfixed_contact_pos_for_pair(
        contact_scene_def.contact_pairs[1]
    )
    assert len(pos) == num_ctrl_points
    assert pos[0].shape == (2, 1)

    vels = scene_prob._get_vel_from_pos_by_fe(pos)
    assert len(vels) == num_ctrl_points - 1
    assert vels[0].shape == (2, 1)

    # velocity constraint for each sliding pair for each control point - 1
    assert len(prog.linear_constraints()) == 2 * (num_ctrl_points - 1)

    # Friction cone (only normal force) for each control point
    # one for lams, one for trig terms, for each control point
    assert len(prog.bounding_box_constraints()) == 3 * num_ctrl_points

    # Should not be any generic constraints
    assert len(prog.generic_constraints()) == 0

    # 1 * for so(2) constraint
    # 1 * for torque balance for each control point
    # 2 (dims) * equal contact points for each control point
    # 2 * 2 (dims) * equal rel pos for each control point
    # 2 * 2 (dims) * equal opposite forces for each control point
    assert len(prog.quadratic_constraints()) == 12 * num_ctrl_points

    # 1 * force balance for each control point
    # 1 * 2 (dims) for equal contact points
    assert len(prog.linear_equality_constraints()) == 3 * num_ctrl_points


def test_contact_scene_initial_and_target(
    contact_scene_def: ContactSceneDefinition,
) -> None:
    box_table = contact_scene_def.contact_pairs[0]
    box_robot = contact_scene_def.contact_pairs[1]

    num_ctrl_points = 4
    contact_modes = {
        box_table: ContactMode.ROLLING,
        box_robot: ContactMode.ROLLING,
    }

    scene_prob = ContactSceneProgram(contact_scene_def, num_ctrl_points, contact_modes)

    original_num_lin_eqs = len(scene_prob.prog.linear_equality_constraints())
    scene_prob.constrain_contact_position_at_ctrl_point(box_table, 0, 0.5)

    assert (
        len(scene_prob.prog.linear_equality_constraints()) == original_num_lin_eqs + 1
    )

    scene_prob.constrain_orientation_at_ctrl_point(box_table, 0, 0)
    scene_prob.constrain_orientation_at_ctrl_point(box_table, num_ctrl_points - 1, 0.2)

    assert (
        len(scene_prob.prog.linear_equality_constraints()) == original_num_lin_eqs + 3
    )


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS == True,
    reason="Too slow",
)
def test_contact_scene_program_planning_w_semidefinite_relaxation(
    contact_scene_def: ContactSceneDefinition,
) -> None:
    box_table = contact_scene_def.contact_pairs[0]
    box_robot = contact_scene_def.contact_pairs[1]

    num_ctrl_points = 4
    contact_modes = {
        box_table: ContactMode.ROLLING,
        box_robot: ContactMode.ROLLING,
    }

    scene_prob = ContactSceneProgram(contact_scene_def, num_ctrl_points, contact_modes)

    scene_prob.constrain_contact_position_at_ctrl_point(box_table, 0, 0.5)

    scene_prob.constrain_orientation_at_ctrl_point(box_table, 0, 0)
    scene_prob.constrain_orientation_at_ctrl_point(box_table, num_ctrl_points - 1, 0.2)

    relaxed_prog = MakeSemidefiniteRelaxation(scene_prob.prog)

    solver_options = SolverOptions()

    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    result = Solve(relaxed_prog, solver_options=solver_options)
    assert result.is_success()
