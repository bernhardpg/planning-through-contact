from typing import List

import numpy as np
import pydot
import pytest
from pydrake.geometry import SceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
)

from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.simulation.controllers.hybrid_mpc import (
    HybridModelPredictiveControl,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_geometry import (
    SliderPusherGeometry,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)
from planning_through_contact.simulation.systems.slider_pusher_trajectory_feeder import (
    SliderPusherTrajectoryFeeder,
)
from tests.geometry.planar.fixtures import face_contact_mode
from tests.simulation.dynamics.test_slider_pusher_system import (
    box_geometry,
    rigid_body_box,
    slider_pusher_system,
)
from tests.simulation.systems.test_slider_pusher_trajectory_feeder import (
    one_contact_mode_vars,
)


@pytest.fixture
def hybrid_mpc(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
) -> HybridModelPredictiveControl:
    mpc = HybridModelPredictiveControl(slider_pusher_system)
    return mpc


def test_get_linear_system(hybrid_mpc: HybridModelPredictiveControl) -> None:  # type: ignore
    linear_system = hybrid_mpc._get_linear_system(
        np.array([0, 0, 0, 0.5]), np.array([1.0, 0, 0])
    )
    assert np.allclose(linear_system.A(), np.zeros((4, 4)))
    assert sum(linear_system.B().flatten() != 0) == 4


def test_hybrid_mpc(
    face_contact_mode: FaceContactMode,
    one_contact_mode_vars: List[AbstractModeVariables],
    hybrid_mpc: HybridModelPredictiveControl,
) -> None:  # type: ignore
    slider_geometry = face_contact_mode.object.geometry
    contact_location = face_contact_mode.contact_location

    builder = DiagramBuilder()

    builder.AddNamedSystem("mpc", hybrid_mpc)
    feeder = builder.AddNamedSystem(
        "feedforward", SliderPusherTrajectoryFeeder(one_contact_mode_vars)
    )
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher = builder.AddNamedSystem(
        "slider_pusher", SliderPusherSystem(slider_geometry, contact_location)
    )

    slider_pusher_geometry = SliderPusherGeometry.add_to_builder(
        builder,
        slider_pusher.get_output_port(),
        slider_pusher.slider_geometry,
        slider_pusher.contact_location,
        scene_graph,
    )

    DEBUG = True
    if DEBUG:
        T_VW = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        LIM = 0.8
        visualizer = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            T_VW=T_VW,
            xlim=[-LIM, LIM],
            ylim=[-LIM, LIM],
            show=False,
        )

    builder.Connect(slider_pusher.get_output_port(), hybrid_mpc.get_state_port())
    builder.Connect(hybrid_mpc.get_control_port(), slider_pusher.get_input_port())
    builder.Connect(
        feeder.get_state_feedforward_port(), hybrid_mpc.get_desired_state_port()
    )
    builder.Connect(
        feeder.get_control_feedforward_port(), hybrid_mpc.get_desired_input_port()
    )

    diagram = builder.Build()
    diagram.set_name("diagram")

    context = diagram.CreateDefaultContext()

    if DEBUG:
        visualizer.start_recording()  # type: ignore

    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(face_contact_mode.time_in_mode)

    DEBUG = True

    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore

        visualizer.stop_recording()  # type: ignore
        ani = visualizer.get_recording_as_animation()  # type: ignore
        # Playback the recording and save the output.
        ani.save("test.mp4", fps=30)
