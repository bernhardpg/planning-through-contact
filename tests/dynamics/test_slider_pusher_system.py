import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pytest
from pydrake.systems.all import ConstantVectorSource
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

from planning_through_contact.dynamics.slider_pusher_system import SliderPusherSystem
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
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


def test_slider_pusher(rigid_body_box: RigidBody) -> None:
    builder = DiagramBuilder()
    slider_pusher = SliderPusherSystem(rigid_body_box.geometry)
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
    # slider_pusher.get_wrench(system_context)

    breakpoint()

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
