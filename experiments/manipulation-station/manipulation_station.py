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
    # station.export_diagram("deleteme")
    box = sim.get_box()
    initial_pose = sim.get_box_planar_pose()
    target_pose = PlanarPose(x=0.5, y=0.5, theta=0.2)

    finger_pose = sim.get_pusher_planar_pose()
    sim.run()

    # Make motion plan:
    # 1. Need object geometry and mass (not inertia)
    # 2. Need initial object position
    # 3. Need initial manipulator position
    # 4. Need target object position


if __name__ == "__main__":
    # manipulation_station()
    # simple_iiwa_and_brick()
    planar_pushing_station()
    # teleop()
