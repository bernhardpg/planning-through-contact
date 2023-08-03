from typing import List

import numpy as np
import pytest
from pydrake.all import ConnectPlanarSceneGraphVisualizer
from pydrake.geometry import SceneGraph
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
    assemble_progs_from_contact_modes,
)
from planning_through_contact.planning.planar.planar_plan_specs import PlanarPlanSpecs
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_geometry import (
    SliderPusherGeometry,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)
from planning_through_contact.simulation.systems.slider_pusher_trajectory_feeder import (
    SliderPusherTrajectoryFeeder,
)
from tests.geometry.planar.fixtures import (
    box_geometry,
    face_contact_mode,
    rigid_body_box,
)


@pytest.fixture
def one_contact_mode_vars(
    face_contact_mode: FaceContactMode,
) -> List[AbstractModeVariables]:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.8)

    face_contact_mode.set_slider_initial_pose(initial_pose)
    face_contact_mode.set_slider_final_pose(final_pose)

    face_contact_mode.formulate_convex_relaxation()
    assert face_contact_mode.relaxed_prog is not None
    relaxed_result = Solve(face_contact_mode.relaxed_prog)
    assert relaxed_result.is_success()

    prog = assemble_progs_from_contact_modes([face_contact_mode])
    initial_guess = relaxed_result.GetSolution(
        face_contact_mode.relaxed_prog.decision_variables()[: prog.num_vars()]
    )
    prog.SetInitialGuess(prog.decision_variables(), initial_guess)
    result = Solve(prog)

    vars = face_contact_mode.variables.eval_result(result)
    return [vars]


def test_feeder(
    face_contact_mode: FaceContactMode,
    one_contact_mode_vars: List[AbstractModeVariables],
) -> None:
    slider_geometry = face_contact_mode.object.geometry
    contact_location = face_contact_mode.contact_location

    builder = DiagramBuilder()

    feeder = builder.AddNamedSystem(
        "feedforward", SliderPusherTrajectoryFeeder(one_contact_mode_vars)
    )
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher_geometry = SliderPusherGeometry.add_to_builder(
        builder,
        feeder.GetOutputPort("state"),
        slider_geometry,
        contact_location,
        scene_graph,
    )

    # Connect planar visualizer
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
            show=True,
        )

    diagram = builder.Build()
    diagram.set_name("diagram")

    context = diagram.CreateDefaultContext()

    if DEBUG:
        visualizer.start_recording()  # type: ignore

    SIMULATION_END = 7
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(SIMULATION_END)

    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore

        visualizer.stop_recording()  # type: ignore
        ani = visualizer.get_recording_as_animation()  # type: ignore
        # Playback the recording and save the output.
        ani.save("test.mp4", fps=30)

    breakpoint()

    # slider_pusher = builder.AddNamedSystem(
    #     "slider_pusher",
    #     SliderPusherSystem(
    #         face_contact_mode.object.geometry, face_contact_mode.contact_location
    #     ),
    # )
