from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.hardware.planar_pushing_position_control import (
    PlanarPushingPositionControlNode,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)


def run_position_control_node(debug: bool = False):
    traj_name = "trajectories/box_pushing_2.pkl"
    traj = PlanarPushingTrajectory.load(traj_name)

    config = PlanarPushingSimConfig()
    position_control_node = PlanarPushingPositionControlNode(traj, 3, config)
    if debug:
        position_control_node.export_diagram("position_control_diagram.png")

    position_control_node.run()


if __name__ == "__main__":
    run_position_control_node(debug=True)
