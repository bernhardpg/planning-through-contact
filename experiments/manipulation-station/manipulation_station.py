import numpy as np
import pydot
from pydrake.common import FindResourceOrThrow
from pydrake.examples import ManipulationStation
from pydrake.geometry import DrakeVisualizer, Meshcat, MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import FirstOrderLowPassFilter, VectorLogSink

from simulation.planar_pushing.planar_pushing_iiwa import (
    PlanarPose,
    PlanarPushingSimulation,
)


def planar_pushing_station():
    sim = PlanarPushingSimulation()
    box = sim.get_box()

    box_initial_pose = PlanarPose(x=1.0, y=0.0, theta=0.0)
    box_target_pose = PlanarPose(x=0.5, y=0.0, theta=0.2)
    sim.set_box_planar_pose(box_initial_pose)

    finger_initial_pose = PlanarPose(x=0.7, y=0.3, theta=0.0)
    sim.set_pusher_planar_pose(finger_initial_pose)

    # finger_pose = sim.get_pusher_planar_pose()
    sim.run()


if __name__ == "__main__":
    # manipulation_station()
    # simple_iiwa_and_brick()
    planar_pushing_station()
    # teleop()
