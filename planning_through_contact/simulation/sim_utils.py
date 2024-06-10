import copy
import os
import time
from typing import List, Literal

import numpy as np
from lxml import etree
from pydrake.all import Box as DrakeBox
from pydrake.all import (
    ContactModel,
    DiscreteContactApproximation,
    GeometryInstance,
    LoadModelDirectives,
    MakePhongIllustrationProperties,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    Rgba,
)
from pydrake.all import RigidBody as DrakeRigidBody
from pydrake.all import RigidTransform, RollPitchYaw, Transform

from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.visualize.colors import COLORS

package_xml_file = os.path.join(os.path.dirname(__file__), "models/package.xml")
models_folder = os.path.join(os.path.dirname(__file__), "models")


def GetParser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    return parser


def ConfigureParser(parser):
    """Add the manipulation/package.xml index to the given Parser."""
    parser.package_map().AddPackageXml(filename=package_xml_file)
    AddPackagePaths(parser)


def AddPackagePaths(parser):
    parser.package_map().PopulateFromFolder(str(models_folder))


def LoadRobotOnly(sim_config, robot_plant_file) -> MultibodyPlant:
    robot = MultibodyPlant(sim_config.time_step)
    parser = GetParser(robot)
    # Load the controller plant, i.e. the plant without the box
    directives = LoadModelDirectives(f"{models_folder}/{robot_plant_file}")
    ProcessModelDirectives(directives, robot, parser)  # type: ignore
    robot.Finalize()
    return robot


def AddSliderAndConfigureContact(sim_config, plant, scene_graph) -> ModelInstanceIndex:
    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)
    use_hydroelastic = sim_config.contact_model == ContactModel.kHydroelastic

    if not use_hydroelastic:
        raise NotImplementedError()

    directives = LoadModelDirectives(
        f"{models_folder}/{sim_config.scene_directive_name}"
    )
    ProcessModelDirectives(directives, plant, parser)  # type: ignore

    slider_sdf_url = GetSliderUrl(sim_config)

    (slider,) = parser.AddModels(url=slider_sdf_url)

    if use_hydroelastic:
        plant.set_contact_model(ContactModel.kHydroelastic)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    plant.Finalize()
    return slider


def GetSliderUrl(sim_config, format: Literal["sdf", "yaml"] = "sdf"):
    if isinstance(sim_config.slider.geometry, Box2d):
        slider_sdf_url = f"package://planning_through_contact/box_hydroelastic.{format}"
    elif isinstance(sim_config.slider.geometry, TPusher2d):
        slider_sdf_url = f"package://planning_through_contact/t_pusher.{format}"
    elif isinstance(sim_config.slider.geometry, ArbitraryShape2D):
        slider_sdf_url = f"package://planning_through_contact/arbitrary_shape.{format}"
    else:
        raise NotImplementedError(f"Body '{sim_config.slider}' not supported")
    return slider_sdf_url


def get_slider_sdf_path(sim_config, models_folder: str) -> str:
    if isinstance(sim_config.slider.geometry, Box2d):
        slider_sdf_url = f"{models_folder}/box_hydroelastic.sdf"
    elif isinstance(sim_config.slider.geometry, TPusher2d):
        slider_sdf_url = f"{models_folder}/t_pusher.sdf"
    elif isinstance(sim_config.slider.geometry, ArbitraryShape2D):
        slider_sdf_url = f"{models_folder}/arbitrary_shape.sdf"
    else:
        raise NotImplementedError(f"Body '{sim_config.slider}' not supported")
    return slider_sdf_url


## Domain Randomization Functions


def AddRandomizedSliderAndConfigureContact(
    sim_config,
    plant,
    scene_graph,
    default_color=[0.2, 0.2, 0.2],
    color_range=0.02,
) -> ModelInstanceIndex:
    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)
    use_hydroelastic = sim_config.contact_model == ContactModel.kHydroelastic

    if not use_hydroelastic:
        raise NotImplementedError()

    scene_directive_name = (
        f"{sim_config.scene_directive_name.split('.')[0]}_randomized.yaml"
    )
    directives = LoadModelDirectives(f"{models_folder}/{scene_directive_name}")
    ProcessModelDirectives(directives, plant, parser)  # type: ignore

    sdf_file = get_slider_sdf_path(sim_config, models_folder)
    safe_parse = etree.XMLParser(recover=True)
    tree = etree.parse(sdf_file, safe_parse)
    root = tree.getroot()

    diffuse_elements = root.xpath("//model/link/visual/material/diffuse")

    R = clamp(default_color[0] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    G = clamp(default_color[1] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    B = clamp(default_color[2] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    A = 1  # assuming fully opaque

    new_diffuse_value = f"{R} {G} {B} {A}"
    for diffuse in diffuse_elements:
        diffuse.text = new_diffuse_value

    sdf_as_string = etree.tostring(tree, encoding="utf8").decode()

    (slider,) = parser.AddModelsFromString(sdf_as_string, "sdf")

    if use_hydroelastic:
        plant.set_contact_model(ContactModel.kHydroelastic)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    plant.Finalize()
    return slider


# def randomize_table(
#     default_color = [0.55, 0.55, 0.55],
#     color_range = 0.02,
#     table_urdf: str = "small_table_hydroelastic.urdf"
# ) -> None:
#     base_urdf = f'{models_folder}/{table_urdf}'
#     parser = etree.XMLParser(recover=True)
#     tree = etree.parse(base_urdf, parser)
#     root = tree.getroot()

#     R = clamp(default_color[0] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
#     G = clamp(default_color[1] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
#     B = clamp(default_color[2] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
#     A = 1 # assuming fully opaque

#     new_color_value = f"{R} {G} {B} {A}"
#     models = root.xpath('//material[@name="LightGrey"]')
#     for model in models:
#         for color in model:
#             color.set("rgba", new_color_value)

#     new_urdf_location = f'{models_folder}/small_table_hydroelastic_randomized.urdf'

#     tree.write(new_urdf_location, pretty_print=True, xml_declaration=True, encoding='UTF-8')


def randomize_table(
    default_color=[0.55, 0.55, 0.55],
    color_range=0.02,
    table_urdf: str = "small_table_hydroelastic.urdf",
) -> None:
    base_urdf = f"{models_folder}/{table_urdf}"
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(base_urdf, parser)
    root = tree.getroot()

    rv = np.random.uniform(0, 1)
    import random

    if rv < 0.3:
        image_dir = f"{models_folder}/images"
        image_files = os.listdir(image_dir)
        image_file = random.choice(image_files)
        material = root.xpath('//link[@name="TableTop"]/visual/material')
        material[0].set("name", "")
        texture = etree.SubElement(material[0], "texture")
        texture.set("filename", f"{models_folder}/images/{image_file}")
    else:
        R = clamp(
            default_color[0] + np.random.uniform(-color_range, color_range), 0.0, 1.0
        )
        G = clamp(
            default_color[1] + np.random.uniform(-color_range, color_range), 0.0, 1.0
        )
        B = clamp(
            default_color[2] + np.random.uniform(-color_range, color_range), 0.0, 1.0
        )
        A = 1  # assuming fully opaque
        new_color_value = f"{R} {G} {B} {A}"
        models = root.xpath('//materialz[@name="LightGrey"]')
        for model in models:
            for color in model:
                color.set("rgba", new_color_value)
    # else:
    #     image_dir = f'{models_folder}/images'
    #     image_files = os.listdir(image_dir)
    #     image_file = "russ.png"
    #     material = root.xpath('//link[@name="TableTop"]/visual/material')
    #     material[0].set("name", "")
    #     texture = etree.SubElement(material[0], "texture")
    #     texture.set("filename", f'{models_folder}/{image_file}')
    #     # new_urdf_location = f'{models_folder}/small_table_hydroelastic_randomized.urdf'
    new_urdf_location = f"{models_folder}/small_table_hydroelastic_randomized.urdf"
    tree.write(
        new_urdf_location, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    )


def randomize_pusher(
    default_color=[1.0, 0.45, 0.14],
    color_range=0.02,
    pusher_sdf: str = "pusher_floating_hydroelastic.sdf",
) -> None:
    base_sdf = f"{models_folder}/{pusher_sdf}"

    safe_parse = etree.XMLParser(recover=True)
    tree = etree.parse(base_sdf, safe_parse)
    root = tree.getroot()

    diffuse_elements = root.xpath("//model/link/visual/material/diffuse")

    R = clamp(default_color[0] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    G = clamp(default_color[1] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    B = clamp(default_color[2] + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    A = 1  # assuming fully opaque

    new_diffuse_value = (
        f"{R} {G} {B} {A}"  # Example: changing diffuse to white (R G B A)
    )
    for diffuse in diffuse_elements:
        diffuse.text = new_diffuse_value

    new_sdf_location = f"{models_folder}/pusher_floating_hydroelastic_randomized.sdf"

    tree.write(
        new_sdf_location, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    )


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def randomize_camera_config(camera_config):
    # Randomize camera location
    new_camera_config = copy.deepcopy(camera_config)
    camera_pose = camera_config.X_PB.GetDeterministicValue()

    new_xyz = np.random.normal(camera_pose.translation(), 0.01, 3)
    rpy = camera_pose.rotation().ToRollPitchYaw()
    rot_std = 2 * np.pi / 180
    new_rpy = RollPitchYaw(
        np.random.normal(rpy.roll_angle(), rot_std),
        np.random.normal(rpy.pitch_angle(), rot_std),
        np.random.normal(rpy.yaw_angle(), rot_std),
    )

    new_camera_config.X_PB = Transform(RigidTransform(new_rpy, new_xyz))

    # randomize the background color
    new_rgb = np.random.uniform(0, 1, 3)
    new_camera_config.background = Rgba(new_rgb[0], new_rgb[1], new_rgb[2], 1)

    return new_camera_config


## Collision checkers for computing initial slider and pusher poses


def get_slider_start_poses(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    pusher_pose: PlanarPose,
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(seed)
    slider = config.slider_geometry
    slider_initial_poses = []
    for _ in range(num_plans):
        slider_initial_pose = get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )
        slider_initial_poses.append(slider_initial_pose)

    return slider_initial_poses


def get_slider_pose_within_workspace(
    workspace: PlanarPushingWorkspace,
    slider: CollisionGeometry,
    pusher_pose: PlanarPose,
    config: PlanarPlanConfig,
    limit_rotations: bool = False,
    enforce_entire_slider_within_workspace: bool = True,
    timeout_s: float = 10.0,
) -> PlanarPose:
    valid_pose = False

    start_time = time.time()
    slider_pose = None
    while not valid_pose:
        if time.time() - start_time > timeout_s:
            raise ValueError("Could not find a valid slider pose within the timeout.")

        x_initial = np.random.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = np.random.uniform(workspace.slider.y_min, workspace.slider.y_max)
        EPS = 0.01
        if limit_rotations:
            th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
            # th_initial = np.random.uniform(-np.pi / 4 + EPS, np.pi / 4 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = check_collision(pusher_pose, slider_pose, config)
        within_workspace = slider_within_workspace(workspace, slider_pose, slider)

        if enforce_entire_slider_within_workspace:
            valid_pose = within_workspace and not collides_with_pusher
        else:
            valid_pose = not collides_with_pusher

    assert slider_pose is not None  # fix LSP errors

    return slider_pose


# TODO: refactor
def check_collision(
    pusher_pose_world: PlanarPose,
    slider_pose_world: PlanarPose,
    config: PlanarPlanConfig,
) -> bool:
    p_WP = pusher_pose_world.pos()
    R_WB = slider_pose_world.two_d_rot_matrix()
    p_WB = slider_pose_world.pos()

    # We need to compute the pusher pos in the frame of the slider
    p_BP = R_WB.T @ (p_WP - p_WB)
    pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(pusher_pose_body, loc, config)
    ]
    if len(matching_locs) == 0:
        return True
    else:
        return False


def slider_within_workspace(
    workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: CollisionGeometry
) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    p_Wv_s = [
        slider.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all(
        [v >= lb for v in p_Wv_s]
    )
    return vertices_within_workspace


## Meshcat visualizations


def get_slider_body(robot_system: RobotSystemBase) -> DrakeRigidBody:
    slider_body = robot_system.station_plant.GetUniqueFreeBaseBodyOrThrow(
        robot_system.slider
    )
    return slider_body


def get_slider_shapes(robot_system: RobotSystemBase) -> List[DrakeBox]:
    slider_body = get_slider_body(robot_system)
    collision_geometries_ids = robot_system.station_plant.GetCollisionGeometriesForBody(
        slider_body
    )

    inspector = robot_system._scene_graph.model_inspector()
    shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

    # for now we only support Box shapes
    assert all([isinstance(shape, DrakeBox) for shape in shapes])

    return shapes


def get_slider_shape_poses(robot_system: RobotSystemBase) -> List[DrakeBox]:
    slider_body = get_slider_body(robot_system)
    collision_geometries_ids = robot_system.station_plant.GetCollisionGeometriesForBody(
        slider_body
    )

    inspector = robot_system._scene_graph.model_inspector()
    poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

    return poses


def create_goal_geometries(
    robot_system: RobotSystemBase,
    desired_planar_pose: PlanarPose,
    box_color=COLORS["emeraldgreen"],
    desired_pose_alpha=0.3,
) -> List[str]:
    shapes = get_slider_shapes(robot_system)
    poses = get_slider_shape_poses(robot_system)
    heights = [shape.height() for shape in shapes]
    min_height = min(heights)
    desired_pose = desired_planar_pose.to_pose(min_height / 2, z_axis_is_positive=True)

    source_id = robot_system._scene_graph.RegisterSource()

    goal_geometries = []
    for idx, (shape, pose) in enumerate(zip(shapes, poses)):
        geom_instance = GeometryInstance(
            desired_pose.multiply(pose),
            shape,
            f"shape_{idx}",
        )
        curr_shape_geometry_id = robot_system._scene_graph.RegisterAnchoredGeometry(
            source_id,
            geom_instance,
        )
        robot_system._scene_graph.AssignRole(
            source_id,
            curr_shape_geometry_id,
            MakePhongIllustrationProperties(box_color.diffuse(desired_pose_alpha)),
        )
        geom_name = f"goal_shape_{idx}"
        goal_geometries.append(geom_name)
        robot_system._meshcat.SetObject(
            geom_name, shape, rgba=Rgba(*box_color.diffuse(desired_pose_alpha))
        )
    return goal_geometries


def visualize_desired_slider_pose(
    robot_system: RobotSystemBase,
    desired_planar_pose: PlanarPose,
    goal_geometries: List[str],
    time_in_recording: float = 0.0,
) -> None:
    shapes = get_slider_shapes(robot_system)
    poses = get_slider_shape_poses(robot_system)

    heights = [shape.height() for shape in shapes]
    min_height = min(heights)
    desired_pose = desired_planar_pose.to_pose(min_height / 2, z_axis_is_positive=True)

    for pose, geom_name in zip(poses, goal_geometries):
        robot_system._meshcat.SetTransform(
            geom_name, desired_pose.multiply(pose), time_in_recording
        )
