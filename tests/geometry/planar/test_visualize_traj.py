import numpy as np
import pydot
from pydrake.all import ConnectPlanarSceneGraphVisualizer, PlanarSceneGraphVisualizer
from pydrake.geometry import MeshcatVisualizer, SceneGraph, StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.simulation.dynamics.slider_pusher.general_slider_pusher_geometry import (
    GeneralSliderPusherGeometry,
)

DEBUG = False


# TODO(bernhardpg): Use this everywhere
def connect_planar_visualizer(
    builder: DiagramBuilder, scene_graph: SceneGraph
) -> PlanarSceneGraphVisualizer:
    T_VW = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    LIM = 0.8
    visualizer = ConnectPlanarSceneGraphVisualizer(
        builder,
        scene_graph,
        T_VW=T_VW,
        xlim=[-LIM, LIM],
        ylim=[-LIM, LIM],
        show=True,
    )
    return visualizer


def test_visualize_2d() -> None:
    slider = TPusher2d()

    builder = DiagramBuilder()

    state = builder.AddNamedSystem(
        "state", ConstantVectorSource(np.array([0, 0, 0, 0]))
    )

    PUSHER_RADIUS = 0.015
    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher_geometry = GeneralSliderPusherGeometry.add_to_builder(
        builder,
        state.get_output_port(),
        slider,
        slider.contact_locations[0],
        scene_graph,
        PUSHER_RADIUS,
    )

    # Connect planar visualizer
    if DEBUG:
        connect_planar_visualizer(builder, scene_graph)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Create the simulator, and simulate for 10 seconds.
    SIMULATION_END = 7
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(SIMULATION_END)

    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore


def test_visualize_3d() -> None:
    slider = TPusher2d()

    builder = DiagramBuilder()

    state = builder.AddNamedSystem(
        "state", ConstantVectorSource(np.array([0, 0, 0, 0]))
    )

    PUSHER_RADIUS = 0.015

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher_geometry = GeneralSliderPusherGeometry.add_to_builder(
        builder,
        state.get_output_port(),
        slider,
        slider.contact_locations[0],
        scene_graph,
        pusher_radius=PUSHER_RADIUS,
    )

    # Connect planar visualizer
    if DEBUG:
        meshcat = StartMeshcat()  # type: ignore
        meshcat.Delete()  # remove everything from visualizer
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Create the simulator, and simulate for 10 seconds.
    SIMULATION_END = 7
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(SIMULATION_END)

    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore
