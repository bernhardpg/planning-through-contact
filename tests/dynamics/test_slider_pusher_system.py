import numpy as np
import numpy.typing as npt
import pydot
import pytest
from pydrake.geometry import SceneGraph
from pydrake.systems.all import ConstantVectorSource, Linearize
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
)
from pydrake.systems.primitives import VectorLogSink

from planning_through_contact.dynamics.slider_pusher.slider_pusher_geometry import (
    SliderPusherGeometry,
)
from planning_through_contact.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.rigid_body import RigidBody


@pytest.fixture
def box_geometry() -> Box2d:
    return Box2d(width=0.3, height=0.3)


@pytest.fixture
def rigid_body_box(box_geometry: Box2d) -> RigidBody:
    mass = 0.3
    box = RigidBody("box", box_geometry, mass)
    return box


@pytest.fixture
def slider_pusher_system(rigid_body_box: RigidBody) -> SliderPusherSystem:
    slider_pusher = SliderPusherSystem(
        rigid_body_box.geometry, PolytopeContactLocation(ContactLocation.FACE, 1)
    )
    return slider_pusher


def test_get_jacobian(slider_pusher_system: SliderPusherSystem) -> None:
    lam = 0.0
    J_c = slider_pusher_system._get_contact_jacobian(lam)
    J_c_target = np.array([[1.0, 0.0, 0.15], [0.0, 1.0, 0.15]])
    assert np.allclose(J_c, J_c_target)

    lam = 1.0
    J_c = slider_pusher_system._get_contact_jacobian(lam)
    J_c_target = np.array([[1.0, 0.0, -0.15], [0.0, 1.0, 0.15]])
    assert np.allclose(J_c, J_c_target)

    lam = 0.5
    J_c = slider_pusher_system._get_contact_jacobian(lam)
    J_c_target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.15]])
    assert np.allclose(J_c, J_c_target)


def test_get_wrench(slider_pusher_system: SliderPusherSystem) -> None:
    c_n = 0.0
    c_f = 0.0
    lam = 0.0
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert w.shape == (3, 1)  # v_x, v_y, omega
    assert np.allclose(w, np.zeros(w.shape))

    c_n = 0.0
    c_f = 0.0
    lam = 1.0
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert np.allclose(w, np.zeros(w.shape))

    c_n = 1.0
    c_f = 0.0
    lam = 0.5
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert w[0] == -1  # should be a generalized force along the negative x-axis
    assert w[1] == 0  # should be a generalized force along the negative x-axis
    assert w[2] == 0  # shouldn't be any torque

    c_n = 1.0
    c_f = -0.5
    lam = 0.5
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert w[2] >= 0  # should be a positive torque
    assert w[0] == -1  # should be a generalized force along the negative x-axis
    assert w[1] == 0.5  # should be a generalized force along the positive y-axis

    c_n = 1.0
    c_f = 0.5
    lam = 0.2
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert w[2] <= 0

    c_n = 1.0
    c_f = 0.5
    lam = 0.8
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    assert w[2] >= 0


def test_get_twist(slider_pusher_system: SliderPusherSystem) -> None:
    check_parallel = lambda u, v: np.isclose(
        u.T.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)), 1
    )
    c_n = 1.0
    c_f = 0.5
    lam = 0.8
    w = slider_pusher_system._get_wrench(lam, c_n, c_f)
    t = slider_pusher_system._get_twist(lam, c_n, c_f)
    assert check_parallel(w[:2], t[:2])  # x and y components should be parallel
    assert t.shape == (3, 1)


def test_calc_dynamics(slider_pusher_system: SliderPusherSystem) -> None:
    x = np.array([0, 0, 0.5, 0])
    u = np.array([0, 0, 0])
    x_dot = slider_pusher_system._calc_dynamics(x, u)
    assert np.allclose(x_dot, np.zeros(x_dot.shape))
    assert x_dot.shape == (4, 1)

    x = np.array([0, 0, 0, 0])
    u = np.array([1, 0, 0])
    x_dot = slider_pusher_system._calc_dynamics(x, u)
    assert x_dot[0] <= 0
    assert x_dot[2] <= 0

    x = np.array([0, 0, 0, 1])
    u = np.array([1, 0, 0])
    x_dot = slider_pusher_system._calc_dynamics(x, u)
    assert x_dot[0] <= 0
    assert x_dot[2] >= 0

    x = np.array([0, 0, 0, 0])
    u = np.array([0, 0, 1])
    x_dot = slider_pusher_system._calc_dynamics(x, u)
    assert x_dot[3] == 1


@pytest.mark.parametrize(
    "slider_pusher_system, input",
    [({}, np.array([0, 0, 0]))],
    indirect=["slider_pusher_system"],
)
def test_slider_pusher(
    slider_pusher_system: SliderPusherSystem, input: npt.NDArray[np.float64]
) -> None:
    slider_pusher = slider_pusher_system

    builder = DiagramBuilder()
    builder.AddNamedSystem("slider_pusher", slider_pusher)

    # input
    constant_input = ConstantVectorSource(np.array([0, 0, 0]))
    builder.AddNamedSystem("input", constant_input)
    builder.Connect(constant_input.get_output_port(), slider_pusher.get_input_port())

    scene_graph = builder.AddSystem(SceneGraph())
    source_id = scene_graph.RegisterSource()

    # logger
    state_logger = VectorLogSink(slider_pusher.num_continuous_states())
    builder.AddNamedSystem("state_logger", state_logger)
    builder.Connect(slider_pusher.get_output_port(), state_logger.get_input_port())

    input_logger = VectorLogSink(slider_pusher.input.size())
    builder.AddNamedSystem("input_logger", input_logger)
    builder.Connect(constant_input.get_output_port(), input_logger.get_input_port())

    diagram = builder.Build()
    diagram.set_name("diagram")

    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore

    context = diagram.CreateDefaultContext()

    x_initial = np.array([0, 0, 0, 0.5])
    context.SetContinuousState(x_initial)

    # Create the simulator, and simulate for 10 seconds.
    SIMULATION_END = 10
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(SIMULATION_END)

    state_log = state_logger.FindLog(context)
    input_log = input_logger.FindLog(context)

    # TODO


@pytest.mark.parametrize(
    "slider_pusher_system, input",
    [({}, np.array([0, 0, 0]))],
    indirect=["slider_pusher_system"],
)
def test_slider_pusher_visualization(
    slider_pusher_system: SliderPusherSystem, input: npt.NDArray[np.float64]
) -> None:
    slider_pusher = slider_pusher_system

    builder = DiagramBuilder()
    builder.AddNamedSystem("slider_pusher", slider_pusher)

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher_geometry = SliderPusherGeometry.add_to_builder(
        builder,
        slider_pusher.get_output_port(),
        slider_pusher.slider_geometry,
        slider_pusher.contact_location,
        scene_graph,
    )

    # input
    constant_input = ConstantVectorSource(np.array([0, 0, 0.1]))
    builder.AddNamedSystem("input", constant_input)
    builder.Connect(constant_input.get_output_port(), slider_pusher.get_input_port())

    # Connect planar visualizer
    T_VW = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    LIM = 0.8
    visualizer = ConnectPlanarSceneGraphVisualizer(
        builder, scene_graph, T_VW=T_VW, xlim=[-LIM, LIM], ylim=[-LIM, LIM], show=True
    )

    diagram = builder.Build()
    diagram.set_name("diagram")

    context = diagram.CreateDefaultContext()
    x_initial = np.array([0, 0, 0, 0.5])
    context.SetContinuousState(x_initial)

    visualizer.start_recording()

    # Create the simulator, and simulate for 10 seconds.
    SIMULATION_END = 10
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    # simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(SIMULATION_END)

    visualizer.stop_recording()
    ani = visualizer.get_recording_as_animation()
    # Playback the recording and save the output.
    ani.save("test.mp4", fps=30)

    DEBUG = True
    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore


def test_linearize_slider_pusher(slider_pusher_system: SliderPusherSystem) -> None:
    system = slider_pusher_system

    context = system.CreateDefaultContext()
    context.SetContinuousState([0, 0, 0, 0])
    system.get_input_port(0).FixValue(context, [0, 0, 0])
    lin_system = Linearize(system, context)

    assert lin_system.A().shape == (4, 4)
    assert lin_system.B().shape == (4, 3)
