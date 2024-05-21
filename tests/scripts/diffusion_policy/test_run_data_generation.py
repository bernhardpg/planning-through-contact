import os
import shutil

import pytest

PYTHON_SCRIPT = "scripts/planar_pushing/diffusion_policy/run_data_generation.py"


@pytest.fixture
def config_path() -> str:
    return "test_sim_config.yaml"


@pytest.fixture
def config_dir() -> str:
    return "tests/scripts/diffusion_policy"


def test_run_data_generation(config_path: str, config_dir: str):
    # Run a command on the command line using os
    command = (
        f"python {PYTHON_SCRIPT} --config-dir {config_dir} --config-name {config_path}"
    )
    os.system(command)
    passed = _passed_test()

    # Clean up test
    shutil.rmtree("tests/scripts/diffusion_policy/temp")
    assert passed


def _passed_test() -> bool:
    # check if required paths exist in generate plans
    trajectory_path = "tests/scripts/diffusion_policy/temp/test_trajectories"
    if not os.path.exists(
        f"{trajectory_path}/traj_0_0/analysis/rounded_traj_trajectory.pdf"
    ):
        return False
    if not os.path.exists(f"{trajectory_path}/traj_0_0/trajectory/traj_rounded.pkl"):
        return False
    if not os.path.exists(f"{trajectory_path}/config.yaml"):
        return False

    # check if required paths exist in render plans
    rendered_path = "tests/scripts/diffusion_policy/temp/test_trajectories_rendered"
    if not os.path.exists(f"{rendered_path}/0/images/0.png"):
        return False
    if not os.path.exists(f"{rendered_path}/0/combined_logs.pkl"):
        return False
    if not os.path.exists(f"{rendered_path}/0/log.txt"):
        return False
    if not os.path.exists(f"{rendered_path}/config.yaml"):
        return False

    # check if required paths exist in convert to zarr
    zarr_path = (
        "tests/scripts/diffusion_policy/temp/test_trajectories_rendered/data.zarr"
    )
    data_paths = [
        "data/action",
        "data/img",
        "data/slider_state",
        "data/state",
        "data/target",
    ]
    for data_path in data_paths:
        if not os.path.exists(f"{zarr_path}/{data_path}"):
            return False
    if not os.path.exists(f"{zarr_path}/meta/episode_ends"):
        return False

    return True
