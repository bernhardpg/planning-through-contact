from typing import List, Optional

import numpy as np
import pydot
import pydrake.symbolic as sym
import pytest
import logging
from pydrake.geometry import SceneGraph
from pydrake.solvers import CommonSolverOption, Solve, SolverOptions
from pydrake.systems.all import ZeroOrderHold
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
)
from pydrake.systems.primitives import VectorLogSink, ZeroOrderHold_

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import (
    FaceContactMode,
    FaceContactVariables,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    assemble_progs_from_contact_modes,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    SliderPusherSystemConfig,
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
    analyze_mode_result,
    plot_control_sols_vs_time,
    plot_planar_pushing_logs,
    plot_planar_pushing_trajectory,
    plot_velocities,
)

from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_start_and_goal,
    visualize_planar_pushing_trajectory,
)

DEBUG = False
logging.getLogger(
    "planning_through_contact.simulation.controllers.hybrid_mpc"
).setLevel(logging.DEBUG)


@pytest.fixture
def mpc_config() -> HybridMpcConfig:
    # config = HybridMpcConfig(
    #     step_size=0.1,
    #     horizon=10,
    #     num_sliding_steps=5,
    #     rate_Hz=20,
    # )
    config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=30,
        Q=np.diag([3, 3, 0.5, 0]) * 10,
        Q_N=np.diag([3, 3, 0.5, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
    )
    # config = HybridMpcConfig(
    #     step_size=0.03,
    #     horizon=50,
    #     num_sliding_steps=1,
    #     rate_Hz=200,
    #     Q=np.diag([3, 3, 0.01, 0]) * 100,
    #     Q_N=np.diag([3, 3, 1, 0]) * 2000,
    #     R=np.diag([1, 1, 0]) * 0.5,
    # )
    return config


@pytest.fixture
def slider_pusher_system() -> SliderPusherSystem:  # type: ignore
    mass = 0.1
    box_geometry = Box2d(width=0.15, height=0.15)
    box = RigidBody("box", box_geometry, mass)

    config = SliderPusherSystemConfig(slider=box, friction_coeff_slider_pusher=0.5)

    contact_idx = 3

    sys = SliderPusherSystem(
        contact_location=PolytopeContactLocation(ContactLocation.FACE, contact_idx),
        config=config,
    )
    return sys


@pytest.fixture
def hybrid_mpc(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
    mpc_config: HybridMpcConfig,
) -> HybridMpc:
    mpc = HybridMpc(slider_pusher_system, mpc_config, slider_pusher_system.config)
    return mpc


@pytest.fixture
def hybrid_mpc_controller_system(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
    mpc_config: HybridMpcConfig,
) -> HybridModelPredictiveControlSystem:
    mpc = HybridModelPredictiveControlSystem(slider_pusher_system, mpc_config)
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

    feeder = SliderPusherTrajectoryFeeder(
        one_contact_mode_vars, hybrid_mpc.dynamics_config, hybrid_mpc.config
    )
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


def test_get_control_with_disturbance(
    one_contact_mode_vars: List[FaceContactVariables],
    hybrid_mpc: HybridMpc,
) -> None:
    feeder = SliderPusherTrajectoryFeeder(
        one_contact_mode_vars, hybrid_mpc.dynamics_config, hybrid_mpc.config
    )
    context = feeder.CreateDefaultContext()

    desired_state_traj = feeder.get_state_traj_feedforward_port().Eval(context)
    desired_control_traj = feeder.get_control_traj_feedforward_port().Eval(context)[:-1]  # type: ignore
    initial_state = feeder.get_state(0) + np.array([0.1, 0.1, 0.1, 0])

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

    if DEBUG:
        plot_planar_pushing_trajectory(actual, desired)

    # Make sure we are able to stabilize the system in one plan
    # Edit: We wont, once we have to choose a contact mode
    # assert np.isclose(actual.x[-1], desired.x[-1], atol=0.01)
    # assert np.isclose(actual.y[-1], desired.y[-1], atol=0.01)
    # assert np.isclose(actual.theta[-1], desired.theta[-1], atol=0.01)


@pytest.fixture
def one_contact_mode(
    slider_pusher_system: SliderPusherSystem,  # type:ignore
) -> FaceContactMode:
    config = PlanarPlanConfig(dynamics_config=slider_pusher_system.config)
    mode = FaceContactMode.create_from_plan_spec(
        slider_pusher_system.contact_location, config
    )
    return mode


def generate_path(
    one_contact_mode: FaceContactMode, initial_pose: PlanarPose, final_pose: PlanarPose
) -> List[FaceContactVariables]:
    one_contact_mode.set_slider_initial_pose(initial_pose)
    one_contact_mode.set_slider_final_pose(final_pose)

    one_contact_mode.formulate_convex_relaxation()
    assert one_contact_mode.relaxed_prog is not None
    relaxed_result = Solve(one_contact_mode.relaxed_prog)
    assert relaxed_result.is_success()

    # NOTE: We don't use nonlinear rounding, but rather return the relaxed solution
    # TODO(bernhardpg): Change this when nonlinear rounding is working again
    # prog = assemble_progs_from_contact_modes([one_contact_mode])
    # initial_guess = relaxed_result.GetSolution(
    #     one_contact_mode.relaxed_prog.decision_variables()[: prog.num_vars()]
    # )
    # prog.SetInitialGuess(prog.decision_variables(), initial_guess)
    #
    # solver_options = SolverOptions()
    # if DEBUG:
    #     solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
    #
    # result = Solve(prog, solver_options=solver_options)
    # assert result.is_success()

    vars = one_contact_mode.variables.eval_result(relaxed_result)
    return [vars]


@pytest.fixture
def one_contact_mode_vars(
    one_contact_mode: FaceContactMode,
) -> List[FaceContactVariables]:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0.2, 2.5)
    return generate_path(one_contact_mode, initial_pose, final_pose)


def execute_hybrid_mpc_controller(
    one_contact_mode: FaceContactMode,
    one_contact_mode_vars: List[FaceContactVariables],
    hybrid_mpc_controller_system: HybridModelPredictiveControlSystem,
    initial_disturbance: np.ndarray,
    traj: Optional[PlanarPushingTrajectory] = None,
) -> None:
    mpc_controller = hybrid_mpc_controller_system

    slider_geometry = one_contact_mode.config.slider_geometry
    contact_location = one_contact_mode.contact_location

    builder = DiagramBuilder()

    builder.AddNamedSystem("mpc", mpc_controller)

    feeder = builder.AddNamedSystem(
        "feedforward",
        SliderPusherTrajectoryFeeder(
            one_contact_mode_vars,
            one_contact_mode.config.dynamics_config,
            mpc_controller.config,
        ),
    )
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    slider_pusher = builder.AddNamedSystem(
        "slider_pusher",
        SliderPusherSystem(contact_location, one_contact_mode.config.dynamics_config),
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
            show=True,
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
    x_initial = feeder.get_state(0) + initial_disturbance
    context.SetContinuousState(x_initial)

    if DEBUG:
        visualizer.start_recording()  # type: ignore

    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.AdvanceTo(one_contact_mode.time_in_mode + 0.3)

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
        plot_control_sols_vs_time(mpc_controller.mpc.control_log)
        # plot_velocities(
        #     mpc_controller.mpc.desired_velocity_log,
        #     mpc_controller.mpc.commanded_velocity_log,
        # )


def test_hybrid_mpc_controller_curve_tracking(
    one_contact_mode: FaceContactMode,
    one_contact_mode_vars: List[FaceContactVariables],
    hybrid_mpc_controller_system: HybridModelPredictiveControlSystem,
) -> None:  # type: ignore
    execute_hybrid_mpc_controller(
        one_contact_mode,
        one_contact_mode_vars,
        hybrid_mpc_controller_system,
        np.array([-0.02, 0.02, 0.1, 0]),
    )


def test_hybrid_mpc_controller_curve_tracking_2(
    one_contact_mode: FaceContactMode,
    hybrid_mpc_controller_system: HybridModelPredictiveControlSystem,
) -> None:  # type: ignore
    path = generate_path(
        one_contact_mode, PlanarPose(0, 0, 0), PlanarPose(0.3, -0.2, -2.5)
    )
    execute_hybrid_mpc_controller(
        one_contact_mode,
        path,
        hybrid_mpc_controller_system,
        np.array([0.0, 0.02, 0.1, 0]),
    )


def test_hybrid_mpc_controller_curve_tracking_B_2(
    mpc_config: HybridMpcConfig,
) -> None:  # type: ignore
    mass = 0.1
    box_geometry = Box2d(width=0.07, height=0.07)
    box = RigidBody("box", box_geometry, mass)

    config = SliderPusherSystemConfig(slider=box, friction_coeff_slider_pusher=0.5)

    contact_idx = 2

    sys = SliderPusherSystem(
        contact_location=PolytopeContactLocation(ContactLocation.FACE, contact_idx),
        config=config,
    )
    mpc_sys = HybridModelPredictiveControlSystem(sys, mpc_config)
    plan_config = PlanarPlanConfig(dynamics_config=sys.config, time_in_contact=4)
    mode = FaceContactMode.create_from_plan_spec(sys.contact_location, plan_config)
    path = generate_path(
        mode, PlanarPose(-0.0564, -0.0463, -0.6279825839830266), PlanarPose(0.0, 0, 0)
    )
    execute_hybrid_mpc_controller(mode, path, mpc_sys, np.array([0.0, 0.0, 0.0, 0.0]))


def test_hybrid_mpc_controller_curve_tracking_C_4() -> None:  # type: ignore
    mass = 0.1
    box_geometry = Box2d(width=0.07, height=0.07)
    box = RigidBody("box", box_geometry, mass)

    config = SliderPusherSystemConfig(
        slider=box,
        friction_coeff_table_slider=0.5,
        friction_coeff_slider_pusher=0.25,
        integration_constant=0.02,
    )

    contact_idx = 2

    sys = SliderPusherSystem(
        contact_location=PolytopeContactLocation(ContactLocation.FACE, contact_idx),
        config=config,
    )
    mpc_config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=50,
        Q=np.diag([3, 3, 0.01, 0]) * 100,
        Q_N=np.diag([3, 3, 1, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
    )
    mpc_sys = HybridModelPredictiveControlSystem(sys, mpc_config)
    start_and_goal = PlanarPushingStartAndGoal(
        slider_initial_pose=PlanarPose(
            x=0.5378211006117899, y=0.04524750758737649, theta=-2.2577135799479247
        ),
        slider_target_pose=PlanarPose(
            x=0.6130126810603981, y=-0.00039604693591686023, theta=-0.4058877726442287
        ),
        pusher_initial_pose=PlanarPose(x=0.0, y=0.0, theta=0.0),
        pusher_target_pose=PlanarPose(x=0.0, y=0.1, theta=0.0),
    )
    plan_config = PlanarPlanConfig(
        dynamics_config=sys.config,
        time_in_contact=2,
        start_and_goal=start_and_goal,
    )
    mode = FaceContactMode.create_from_plan_spec(sys.contact_location, plan_config)

    path = generate_path(
        mode, start_and_goal.slider_initial_pose, start_and_goal.slider_target_pose
    )
    execute_hybrid_mpc_controller(mode, path, mpc_sys, np.array([0.0, 0.0, 0.0, 0.0]))


@pytest.mark.skip(reason="Requires saved plan file")
def test_hybrid_mpc_controller_curve_tracking_C_1() -> None:  # type: ignore
    traj = PlanarPushingTrajectory.load(
        "trajectories/box_pushing_demos/hw_demo_C_1_rounded.pkl"
    )
    print("start test_C_1")
    for idx, seg in enumerate(traj.traj_segments):
        if seg.mode != PlanarPushingContactMode.NO_CONTACT:
            first_contact_seg = seg
            first_contact_seg_idx = idx
            break

    sys = SliderPusherSystem(
        contact_location=first_contact_seg.mode.to_contact_location(),
        config=traj.config.dynamics_config,
    )
    mpc_config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=50,
        Q=np.diag([3, 3, 0.01, 0]) * 100,
        Q_N=np.diag([3, 3, 1, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
    )
    mpc_sys = HybridModelPredictiveControlSystem(sys, mpc_config)

    face_contact_mode = FaceContactMode.create_from_plan_spec(
        sys.contact_location, traj.config
    )

    make_traj_figure(traj, filename="test_C_1_traj.png")
    execute_hybrid_mpc_controller(
        face_contact_mode,
        [traj.path_knot_points[first_contact_seg_idx]],
        mpc_sys,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )


""" The following fixtures and test replicate the straight line tracking simulation experiment from
"Feedback Control of the Pusher-Slider System: A Story of Hybrid and Underactuated Contact Dynamics"
Francois Robert Hogan and Alberto Rodriguez, 2016, https://arxiv.org/abs/1611.08268
"""


@pytest.fixture
def hogan_sim_mpc_config() -> HybridMpcConfig:
    # For weights, see page 12 of the paper
    # config = HybridMpcConfig(
    #     step_size=0.03,
    #     horizon=35,
    #     num_sliding_steps=1,
    #     rate_Hz=30,
    #     Q=np.diag([1, 3, 0.1, 0]) * 10,
    #     Q_N=np.diag([1, 3, 0.1, 0]) * 200,
    #     R=np.diag([1, 1, 0]) * 0.5,
    # )
    # The best performing weights I found
    config = HybridMpcConfig(
        step_size=0.03,
        horizon=35,
        num_sliding_steps=1,
        rate_Hz=30,
        Q=np.diag([3, 3, 0.5, 0]) * 10,
        Q_N=np.diag([3, 3, 0.5, 0]) * 2000,
        R=np.diag([1, 1, 0]) * 0.5,
    )
    return config


@pytest.fixture
def hogan_sim_hybrid_mpc(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
    hogan_sim_mpc_config: HybridMpcConfig,
) -> HybridMpc:
    mpc = HybridMpc(
        slider_pusher_system, hogan_sim_mpc_config, slider_pusher_system.config
    )
    return mpc


@pytest.fixture
def hogan_sim_hybrid_mpc_controller_system(
    slider_pusher_system: SliderPusherSystem,  # type: ignore
    hogan_sim_mpc_config: HybridMpcConfig,
) -> HybridModelPredictiveControlSystem:
    mpc = HybridModelPredictiveControlSystem(slider_pusher_system, hogan_sim_mpc_config)
    return mpc


@pytest.fixture
def straight_line_path(
    one_contact_mode: FaceContactMode,
) -> List[FaceContactVariables]:
    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.45, 0, 0)

    one_contact_mode.set_slider_initial_pose(initial_pose)
    one_contact_mode.set_slider_final_pose(final_pose)

    one_contact_mode.formulate_convex_relaxation()
    assert one_contact_mode.relaxed_prog is not None
    relaxed_result = Solve(one_contact_mode.relaxed_prog)
    assert relaxed_result.is_success()

    vars = one_contact_mode.variables.eval_result(relaxed_result)
    return [vars]


def test_hybrid_mpc_controller_straight_line_tracking(
    one_contact_mode: FaceContactMode,
    straight_line_path: List[FaceContactVariables],
    hogan_sim_hybrid_mpc_controller_system: HybridModelPredictiveControlSystem,
) -> None:  # type: ignore
    execute_hybrid_mpc_controller(
        one_contact_mode,
        straight_line_path,
        hogan_sim_hybrid_mpc_controller_system,
        np.array([0, 0.01, 15 * np.pi / 180, 0]),
    )
