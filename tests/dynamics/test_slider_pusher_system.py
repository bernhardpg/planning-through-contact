import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pytest
from pydrake.autodiffutils import AutoDiffXd
from pydrake.systems.all import ConstantVectorSource, Linearize
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    Context,
    ContinuousState,
    ContinuousState_,
    DiagramBuilder,
    LeafSystem,
)
from pydrake.systems.primitives import (
    ConstantVectorSource_,
    LogVectorOutput,
    VectorLogSink,
)

from planning_through_contact.dynamics.slider_pusher_system import (
    SliderPusherSystem,
    SliderPusherSystem_,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
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


def test_slider_pusher(slider_pusher_system: SliderPusherSystem) -> None:
    slider_pusher = slider_pusher_system

    builder = DiagramBuilder()
    builder.AddNamedSystem("slider_pusher", slider_pusher)

    # input
    constant_input = ConstantVectorSource(
        np.full(slider_pusher.get_input_port().size(), 1),
    )
    builder.AddNamedSystem("input", constant_input)
    builder.Connect(constant_input.get_output_port(), slider_pusher.get_input_port())

    # logger
    logger = VectorLogSink(slider_pusher.num_continuous_states())
    builder.AddNamedSystem("logger", logger)
    builder.Connect(slider_pusher.get_output_port(), logger.get_input_port())

    diagram = builder.Build()
    diagram.set_name("diagram")

    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram.png")  # type: ignore

    context = diagram.CreateDefaultContext()
    system_context = slider_pusher.GetMyContextFromRoot(context)

    x_initial = np.array([0, 0, 0, 0])
    context.SetContinuousState(x_initial)

    # Create the simulator, and simulate for 10 seconds.
    SIMULATION_END = 10
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(SIMULATION_END)

    log = logger.FindLog(context)

    # Plot the results
    plt.figure()
    plt.plot(log.sample_times(), log.data().transpose())
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.show()


def test_linearize_slider_pusher(slider_pusher_system: SliderPusherSystem) -> None:
    system = slider_pusher_system

    context = system.CreateDefaultContext()
    context.SetContinuousState([0, 0, 0, 0])
    system.get_input_port(0).FixValue(context, [0, 0, 0])
    lin_system = Linearize(system, context)

    assert lin_system.A().shape == (4, 4)
    assert lin_system.B().shape == (4, 3)
