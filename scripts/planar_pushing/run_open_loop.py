import numpy as np
import pydot
from pydrake.multibody.plant import ContactModel
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingDiagram,
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim import (
    solve_ik,
)
from planning_through_contact.simulation.systems.open_loop_pushing_controller import (
    OpenLoopPushingController,
)


# TODO(bernhardpg): Generalize this
def get_slider_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.15, height=0.15)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def run_sim(plan: str, save_recording: bool = False, debug: bool = False):
    traj = PlanarPushingTrajectory.load(plan)

    slider = get_slider_box()

    config = PlanarPushingSimConfig(
        body="box",
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose,
        slider_goal_pose=traj.target_slider_planar_pose,
        visualize_desired=True,
        time_step=1e-3,
        use_realtime=False,
        delay_before_execution=2.0,
        use_diff_ik=True,
        closed_loop=False,
        mpc_config=HybridMpcConfig(rate_Hz=50, pusher_radius=traj.pusher_radius),
    )

    builder = DiagramBuilder()

    open_loop = builder.AddNamedSystem(
        "OpenLoopController", OpenLoopPushingController(traj, slider, config)
    )

    station = builder.AddNamedSystem(
        "PlanarPushingDiagram",
        PlanarPushingDiagram(add_visualizer=True, sim_config=config),
    )

    # No feedforward torque
    constant_source = builder.AddNamedSystem("const", ConstantVectorSource(np.zeros(7)))
    builder.Connect(
        constant_source.get_output_port(),
        station.GetInputPort("iiwa_feedforward_torque"),
    )

    builder.Connect(
        open_loop.GetOutputPort("iiwa_position_cmd"),
        station.GetInputPort("iiwa_position"),
    )

    diagram = builder.Build()

    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
        "open_loop_diagram.pdf"
    )

    sim = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)

    context = sim.get_mutable_context()
    mbp_context = station.mbp.GetMyContextFromRoot(context)

    # Set box starting position
    min_height = min([shape.height() for shape in station.get_slider_shapes()])

    # add a small height to avoid the box penetrating the table
    q = config.slider_start_pose.to_generalized_coords(
        min_height + 1e-2, z_axis_is_positive=True
    )
    station.mbp.SetPositions(mbp_context, station.slider, q)

    # Set iiwa starting position
    BUFFER_TO_TABLE = 0.02
    start_joint_positions = solve_ik(
        diagram,
        station,
        config.pusher_start_pose.to_pose(BUFFER_TO_TABLE),
        config.slider_start_pose.to_pose(station.get_slider_min_height()),
        config.default_joint_positions,
    )
    station.mbp.SetPositions(mbp_context, station.iiwa, start_joint_positions)

    # Must init diff ik explicitly as it does not have any robot states!
    open_loop.pusher_pose_to_joint_pos.init_diff_ik(start_joint_positions, context)

    sim.Initialize()
    sim.AdvanceTo(1e8)


if __name__ == "__main__":
    run_sim(plan="trajectories/box_pushing_4.pkl", save_recording=True, debug=True)
