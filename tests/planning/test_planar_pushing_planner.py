import pytest

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from tests.geometry.collision_geometry.test_box2d import box_geometry

# @pytest.fixture
# def planner(rigid_body_box: RigidBody) -> PlanarPushingPlanner:
#     specs = PlanarPlanSpecs()
#     return PlanarPushingPlanner(rigid_body_box, specs)
#
#
# def test_formulate_contact_modes(planner: PlanarPushingPlanner) -> None:
#     # One contact mode per face
#     assert len(planner.contact_modes) == 4
#     assert len(planner.non_collision_modes) == 4
#
#
# def test_build_graph(planner: PlanarPushingPlanner) -> None:
#     # One contact mode per face
#     assert len(planner.contact_vertices) == 4
#
#     # One subgraph between each contact mode:
#     # 4 choose 2 = 6
#     assert len(planner.subgraphs) == 6
#
#
# def test_add_costs(planner: PlanarPushingPlanner) -> None:
#     ...
#
#
# # TODO
# # def test_set_box_initial_and_target(plan_components: PlanComponents) -> None:
# #     planner = PlanarPushingPlanner(plan_components.box, plan_components.specs)
# #     planner.set_slider_initial_pose(plan_components.box_initial_pose)
# #     planner.set_slider_target_pose(plan_components.box_target_pose)
#
#
# if __name__ == "__main__":
#     test_formulate_contact_modes(planner(rigid_body_box(box_geometry())))
#     test_build_graph(planner(rigid_body_box(box_geometry())))
#     # test_set_box_initial_and_target(plan_components(rigid_body_box(box_geometry())))
