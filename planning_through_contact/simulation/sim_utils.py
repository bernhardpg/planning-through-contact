import os
import sys
from typing import Literal

from pydrake.all import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
    MultibodyPlant,
    ContactModel,
    DiscreteContactApproximation,
    ModelInstanceIndex,
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d

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
    else:
        raise NotImplementedError(f"Body '{sim_config.slider}' not supported")
    return slider_sdf_url

## Meshcat visualizations
