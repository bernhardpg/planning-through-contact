import numpy as np
import pytest

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
    ConstantValueSource,
    RigidTransform,
    Quaternion,
    MultibodyPlant,
    AbstractValue,
)

from planning_through_contact.simulation.systems.diff_ik_system import DiffIKSystem
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    LoadRobotOnly
)
from planning_through_contact.experiments.utils import get_box
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystemConfig,
)

@pytest.fixture
def desired_pose() -> RigidTransform:
    return RigidTransform(
        Quaternion(0.0, 1.0, 0.0, 0.0),
        np.array([0.5, 0.25, 0.3])
    )

@pytest.fixture
def diff_ik_plant() -> MultibodyPlant:
    # Create dummy sim_config
    slider = get_box()
    sim_config = PlanarPushingSimConfig(
        slider=get_box(),
        dynamics_config=SliderPusherSystemConfig(
            slider=slider, friction_coeff_slider_pusher=0.05
        ),
        time_step=0.01,
    )
    return LoadRobotOnly(
        sim_config, 'planar_pushing_iiwa_plant_hydroelastic.yaml'
    )

@pytest.fixture
def state() -> np.ndarray:
    state = np.zeros(14)
    state[:7] = np.array([0.41, 0.88, -0.65, -1.45, 0.59, 1.01, 2.76])
    return state

@pytest.fixture
def default_joint_positions() -> np.ndarray:
    return np.array([0.41, 0.88, -0.65, -1.45, 0.59, 1.01, 2.76])

@pytest.fixture
def unreachable_pose() -> RigidTransform:
    return RigidTransform(
        Quaternion(0.0, 1.0, 0.0, 0.0),
        np.array([10.0, 0.0, 0.0])
    )

def test_diff_ik_system(
    diff_ik_plant, 
    default_joint_positions, 
    desired_pose, 
    state
):
    # Build diagram and simulator
    builder = DiagramBuilder()
    
    # Create dummy sim_config
    diff_ik = builder.AddSystem(
        DiffIKSystem(
            plant=diff_ik_plant,
            time_step=0.001,
            default_joint_positions=default_joint_positions,
            log_path = 'tests/simulation/systems/test_diff_ik_system.txt',
        )
    )

    pose_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(desired_pose)))
    state_source = builder.AddSystem(ConstantVectorSource(state))

    builder.Connect(pose_source.get_output_port(0), diff_ik.get_input_port(0))
    builder.Connect(state_source.get_output_port(0), diff_ik.get_input_port(1))
    builder.ExportOutput(diff_ik.get_output_port(0), "q_output")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(0.1)

    # Check output
    context = simulator.get_context()
    output = diagram.GetOutputPort("q_output").Eval(context)
    desired_q = np.array([0.4121, 0.8786, -0.6505, -1.4512, 0.5895, 1.0103, 2.7615])
    assert np.allclose(output, desired_q, atol=1e-4)

def test_diff_ik_system_consequtive_failures(
    diff_ik_plant, 
    default_joint_positions, 
    unreachable_pose, 
    state
):
    # Build diagram and simulator
    builder = DiagramBuilder()
    
    # Create dummy sim_config
    diff_ik = builder.AddSystem(
        DiffIKSystem(
            plant=diff_ik_plant,
            time_step=0.001,
            default_joint_positions=default_joint_positions,
            log_path = 'tests/simulation/systems/test_diff_ik_system.txt',
        )
    )

    pose_source = builder.AddSystem(ConstantValueSource(AbstractValue.Make(unreachable_pose)))
    state_source = builder.AddSystem(ConstantVectorSource(state))

    builder.Connect(pose_source.get_output_port(0), diff_ik.get_input_port(0))
    builder.Connect(state_source.get_output_port(0), diff_ik.get_input_port(1))
    builder.ExportOutput(diff_ik.get_output_port(0), "q_output")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # Run simulation
    desired_consequtive_failures = 10
    simulator.Initialize()
    for i in range(desired_consequtive_failures):
        simulator.AdvanceTo(0.1)
        # Check output
        context = simulator.get_context()
        output = diagram.GetOutputPort("q_output").Eval(context)

    assert(diff_ik._consequtive_ik_fails == desired_consequtive_failures)
