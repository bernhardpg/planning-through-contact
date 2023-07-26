from pathlib import Path

import pytest
from pydrake.solvers import LinearCost
from pydrake.symbolic import Variables

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.analysis import save_gcs_graph_diagram
from tests.geometry.planar.fixtures import box_geometry, rigid_body_box


# @pytest.fixture
def planar_pushing_planner(rigid_body_box: RigidBody) -> PlanarPushingPlanner:
    specs = PlanarPlanSpecs()
    return PlanarPushingPlanner(rigid_body_box, specs)


def test_planar_pushing_planner_construction(
    planar_pushing_planner: PlanarPushingPlanner,
) -> None:
    # One contact mode per face
    assert len(planar_pushing_planner.contact_modes) == 4

    # One contact mode per face
    assert len(planar_pushing_planner.contact_vertices) == 4

    # One subgraph between each contact mode:
    # 4 choose 2 = 6
    assert len(planar_pushing_planner.subgraphs) == 6

    for v, m in zip(
        planar_pushing_planner.contact_vertices, planar_pushing_planner.contact_modes
    ):
        costs = v.GetCosts()

        # angular velocity and linear velocity
        assert len(costs) == 2

        lin_vel, ang_vel = costs

        target_lin_vars = Variables(v.x()[m.get_cost_terms()[0][0]])
        assert target_lin_vars.EqualTo(Variables(lin_vel.variables()))

        target_ang_vars = Variables(v.x()[m.get_cost_terms()[0][1]])
        assert target_ang_vars.EqualTo(Variables(ang_vel.variables()))

        # Costs should be linear in SDP relaxation
        assert isinstance(lin_vel.evaluator(), LinearCost)
        assert isinstance(ang_vel.evaluator(), LinearCost)

    DEBUG = True
    if DEBUG:
        save_gcs_graph_diagram(
            planar_pushing_planner.gcs, Path("planar_pushing_graph.svg")
        )

    breakpoint()


# # TODO
# # def test_set_box_initial_and_target(plan_components: PlanComponents) -> None:
# #     planner = PlanarPushingPlanner(plan_components.box, plan_components.specs)
# #     planner.set_slider_initial_pose(plan_components.box_initial_pose)
# #     planner.set_slider_target_pose(plan_components.box_target_pose)
#
#
if __name__ == "__main__":
    test_planar_pushing_planner_construction(
        planar_pushing_planner(rigid_body_box(box_geometry()))
    )
    # test_set_box_initial_and_target(plan_components(rigid_body_box(box_geometry())))
