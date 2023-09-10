from typing import List

import numpy as np
import pydot
import pydrake.symbolic as sym
import pytest
from pydrake.geometry import SceneGraph
from pydrake.solvers import Solve
from pydrake.systems.all import ZeroOrderHold
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
)
from pydrake.systems.primitives import VectorLogSink, ZeroOrderHold_

from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import (
    HybridModelPredictiveControlSystem,
    HybridModes,
    HybridMpc,
    HybridMpcConfig,
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
from planning_through_contact.visualize.analysis import (
    PlanarPushingLog,
    plot_planar_pushing_logs,
    plot_planar_pushing_trajectory,
)
from tests.geometry.planar.fixtures import (
    dynamics_config,
    face_contact_mode,
    plan_config,
    t_pusher,
)
from tests.simulation.dynamics.test_slider_pusher_system import (
    box_geometry,
    config,
    face_idx,
    rigid_body_box,
    slider_pusher_system,
)
from tests.simulation.systems.test_slider_pusher_trajectory_feeder import (
    contact_mode_example,
    one_contact_mode_vars,
)

DEBUG = True


@pytest.fixture
def hybrid_mpc(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
) -> HybridMpc:
    config = HybridMpcConfig(
        step_size=0.1,
        horizon=10,
        num_sliding_steps=5,
    )
    mpc = HybridMpc(slider_pusher_system, config, slider_pusher_system.config)
    return mpc


@pytest.fixture
def hybrid_mpc_controller_system(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
) -> HybridModelPredictiveControlSystem:
    config = HybridMpcConfig(
        step_size=0.1,
        horizon=10,
        num_sliding_steps=5,
    )
    mpc = HybridModelPredictiveControlSystem(slider_pusher_system, config)
    return mpc


def test_get_linear_system(hybrid_mpc: HybridMpc) -> None:
    check_same_nonzero_elements = lambda A, B: np.all((A == 0) == (B == 0))

    linear_system = hybrid_mpc._get_linear_system(
        np.array([0, 0, 0, 0.5]), np.array([1.0, 0, 0])
    )
    A_target = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.15644441, 0.0],
            [0.0, 0.0, 0.0, 153.9423856],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert check_same_nonzero_elements(A_target, linear_system.A())

    # Linearized around th = 0 and with only a normal force,
    # theta should only impact y_dot
    assert np.all(linear_system.A()[0, :] == 0)
    assert linear_system.A()[1, 2] != 0

    B_target = np.array(
        [
            [4.15644441, -0.0, 0.0],
            [0.0, 4.15644441, 0.0],
            [0.0, -76.9711928, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert check_same_nonzero_elements(B_target, linear_system.B())


def test_get_control_no_movement(hybrid_mpc: HybridMpc) -> None:
    N = hybrid_mpc.config.horizon
    current_state = np.array([0, 0, 0, 0.5])
    desired_state = [current_state] * N
    desired_control = [np.zeros((3,))] * (N - 1)
    prog, x, u = hybrid_mpc._setup_QP(
        current_state, desired_state, desired_control, mode=HybridModes.STICKING
    )

    dt = hybrid_mpc.config.step_size
    times = np.arange(0, dt * N, dt)

    result = Solve(prog)
    assert result.is_success()

    # must evaluate to get rid of expression type
    state_sol = sym.Evaluate(result.GetSolution(x))  # type: ignore
    control_sol = sym.Evaluate(result.GetSolution(u))  # type: ignore

    actual = PlanarPushingLog.from_np(times, state_sol, control_sol)
    desired = PlanarPushingLog.from_np(
        times, np.vstack(desired_state).T, np.vstack(desired_control).T
    )

    assert control_sol.shape == (3, N - 1)
    assert state_sol.shape == (4, N)

    # No deviation should happen
    for state, state_desired in zip(state_sol.T, desired_state):
        assert np.allclose(state, state_desired)

    # No control should be applied
    for control, control_desired in zip(control_sol.T, desired_control):
        assert np.allclose(control, control_desired)

    if DEBUG:
        plot_planar_pushing_trajectory(actual, desired)


def test_get_control_with_plan(
    one_contact_mode_vars: List[FaceContactVariables],
    hybrid_mpc: HybridMpc,
) -> None:
    """
    The plan should follow the desired trajectory exactly.
    """

    feeder = SliderPusherTrajectoryFeeder(one_contact_mode_vars, hybrid_mpc.config)
    context = feeder.CreateDefaultContext()

    desired_state_traj = feeder.get_state_traj_feedforward_port().Eval(context)
    desired_control_traj = feeder.get_control_traj_feedforward_port().Eval(context)[:-1]  # type: ignore
    initial_state = feeder.get_state(0)

    prog, x, u = hybrid_mpc._setup_QP(initial_state, desired_state_traj, desired_control_traj, mode=HybridModes.STICKING)  # type: ignore

    result = Solve(prog)
    assert result.is_success()

    # must evaluate to get rid of expression type
    state_sol = sym.Evaluate(result.GetSolution(x))  # type: ignore
    control_sol = sym.Evaluate(result.GetSolution(u))  # type: ignore

    N = hybrid_mpc.config.horizon
    dt = hybrid_mpc.config.step_size
    times = np.arange(0, dt * N, dt)

    actual = PlanarPushingLog.from_np(times, state_sol, control_sol)
    desired = PlanarPushingLog.from_np(
        times, np.vstack(desired_state_traj).T, np.vstack(desired_control_traj).T  # type: ignore
    )

    # No deviation should happen
    state_treshold = 0.1
    for state, state_desired in zip(state_sol.T, desired_state_traj):  # type: ignore
        assert np.all(
            np.abs(state - state_desired) <= np.full(state.shape, state_treshold)
        )

    # No control should be applied
    control_treshold = 0.4
    for control, control_desired in zip(control_sol.T, desired_control_traj):  # type: ignore
        assert np.all(
            np.abs(control - control_desired)
            <= np.full(control.shape, control_treshold)
        )

    if DEBUG:
        plot_planar_pushing_trajectory(actual, desired)


def test_hybrid_mpc_controller(
    face_contact_mode: FaceContactMode,
    one_contact_mode_vars: List[FaceContactVariables],
    hybrid_mpc_controller_system: HybridModelPredictiveControlSystem,
) -> None:  # type: ignore
    mpc_controller = hybrid_mpc_controller_system

    slider_geometry = face_contact_mode.config.slider_geometry
    contact_location = face_contact_mode.contact_location
    config = face_contact_mode.config

    builder = DiagramBuilder()

    builder.AddNamedSystem("mpc", mpc_controller)

    feeder = builder.AddNamedSystem(
        "feedforward",
        SliderPusherTrajectoryFeeder(one_contact_mode_vars, mpc_controller.config),
    )
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher = builder.AddNamedSystem(
        "slider_pusher",
        SliderPusherSystem(contact_location, face_contact_mode.config.dynamics_config),
    )

    # state logger
    state_logger = builder.AddNamedSystem(
        "state_logger", VectorLogSink(slider_pusher.num_continuous_states())
    )
    builder.Connect(slider_pusher.get_output_port(), state_logger.get_input_port())
    state_desired_logger = builder.AddNamedSystem(
        "state_desired_logger", VectorLogSink(slider_pusher.num_continuous_states())
    )
    builder.Connect(
        feeder.get_state_feedforward_port(), state_desired_logger.get_input_port()
    )

    # control logger
    control_logger = builder.AddNamedSystem(
        "control_logger", VectorLogSink(slider_pusher.get_input_port().size())
    )
    builder.Connect(mpc_controller.get_control_port(), control_logger.get_input_port())
    control_desired_logger = builder.AddNamedSystem(
        "control_desired_logger", VectorLogSink(slider_pusher.get_input_port().size())
    )
    builder.Connect(
        feeder.get_control_feedforward_port(), control_desired_logger.get_input_port()
    )

    slider_pusher_geometry = SliderPusherGeometry.add_to_builder(
        builder,
        slider_pusher.get_output_port(),
        slider_pusher.slider_geometry,
        slider_pusher.pusher_radius,
        slider_pusher.contact_location,
        scene_graph,
    )
    slider_pusher_desired_geometry = SliderPusherGeometry.add_to_builder(
        builder,
        feeder.get_state_feedforward_port(),
        slider_geometry,
        slider_pusher.pusher_radius,
        contact_location,
        scene_graph,
        "desired_slider_pusher_geometry",
        alpha=0.1,
    )

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

    builder.Connect(slider_pusher.get_output_port(), mpc_controller.get_state_port())
    zero_order_hold = builder.AddNamedSystem(
        "zero_order_hold", ZeroOrderHold(mpc_controller.config.step_size, 3)
    )
    builder.Connect(mpc_controller.get_control_port(), zero_order_hold.get_input_port())
    builder.Connect(zero_order_hold.get_output_port(), slider_pusher.get_input_port())
    builder.Connect(
        feeder.get_state_traj_feedforward_port(),
        mpc_controller.get_desired_state_port(),
    )
    builder.Connect(
        feeder.get_control_traj_feedforward_port(),
        mpc_controller.get_desired_control_port(),
    )

    diagram = builder.Build()
    diagram.set_name("diagram")

    context = diagram.CreateDefaultContext()
    x_initial = feeder.get_state(0) + np.array([0.05, 0.05, -0.2, 0])
    context.SetContinuousState(x_initial)

    if DEBUG:
        visualizer.start_recording()  # type: ignore

    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(face_contact_mode.time_in_mode)

    if DEBUG:
        pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_pdf("hybrid_mpc_diagram.pdf")  # type: ignore

        visualizer.stop_recording()  # type: ignore
        ani = visualizer.get_recording_as_animation()  # type: ignore
        # Playback the recording and save the output.
        ani.save("hybrid_mpc.mp4", fps=30)

        state_log = state_logger.FindLog(context)
        desired_state_log = state_desired_logger.FindLog(context)
        control_log = control_logger.FindLog(context)
        desired_control_log = control_desired_logger.FindLog(context)

        plot_planar_pushing_logs(
            state_log, desired_state_log, control_log, desired_control_log
        )
