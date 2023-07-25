import numpy as np
import pydrake.symbolic as sym

from planning_through_contact.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    non_collision_mode,
    non_collision_vars,
    rigid_body_box,
)


def test_non_collision_vars(non_collision_vars: NonCollisionVariables) -> None:
    num_knot_points = non_collision_vars.num_knot_points

    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.cos_th, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.sin_th, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.p_WB_x, float
    )
    assert isinstance(non_collision_vars.cos_th, sym.Variable) or isinstance(
        non_collision_vars.p_WB_y, float
    )
    assert non_collision_vars.p_BF_xs.shape == (num_knot_points,)
    assert non_collision_vars.p_BF_ys.shape == (num_knot_points,)

    assert len(non_collision_vars.R_WBs) == num_knot_points
    for R in non_collision_vars.R_WBs:
        assert R.shape == (2, 2)

    assert len(non_collision_vars.p_WBs) == num_knot_points
    for p in non_collision_vars.p_WBs:
        assert p.shape == (2, 1)

    assert len(non_collision_vars.p_BFs) == num_knot_points
    for p_BF in non_collision_vars.p_BFs:
        assert p_BF.shape == (2, 1)

    assert len(non_collision_vars.v_WBs) == num_knot_points - 1
    for v in non_collision_vars.v_WBs:
        assert v.shape == (2, 1)

    assert len(non_collision_vars.omega_WBs) == num_knot_points - 1
    for o in non_collision_vars.omega_WBs:
        assert isinstance(o, float)

    assert len(non_collision_vars.p_c_Ws) == num_knot_points
    for p in non_collision_vars.p_c_Ws:
        assert p.shape == (2, 1)
        assert isinstance(p[0, 0], sym.Expression)

    assert len(non_collision_vars.f_c_Ws) == num_knot_points
    for f in non_collision_vars.f_c_Ws:
        assert f.shape == (2, 1)
        assert np.all(f == 0)


def test_non_collision_mode(non_collision_mode: NonCollisionMode) -> None:
    mode = non_collision_mode
    num_knot_points = mode.num_knot_points

    # We should have three planes for a collision free region for a normal box
    num_planes = len(mode.planes)
    assert num_planes == 3

    # One linear constraint per plane, per knot point
    num_linear_constraints = len(mode.prog.linear_constraints()) + len(
        mode.prog.bounding_box_constraints()
    )
    assert num_linear_constraints == num_knot_points * num_planes

    # The next two tests may fail for more complex geometries than boxes. If so, update them!
    assert len(mode.prog.bounding_box_constraints()) == 2
    assert len(mode.prog.linear_constraints()) == 4

    assert len(mode.prog.linear_equality_constraints()) == 0

    assert len(mode.prog.linear_costs()) == 0

    # One quadratic cost for squared eucl distances
    assert len(mode.prog.quadratic_costs()) == 1

    lin_vel_vars = sym.Variables(mode.prog.quadratic_costs()[0].variables())
    target_lin_vel_vars = sym.Variables(np.concatenate(mode.variables.p_BFs))
    assert lin_vel_vars.EqualTo(target_lin_vel_vars)
