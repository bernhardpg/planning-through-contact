import pytest

from geometry.rigid_body import RigidBody
from simulation.planar_pushing.planar_pushing_iiwa import PlanarPushingDiagram


# @pytest.fixture
def box() -> RigidBody:
    # Get the default sugar box from the manipulation simulation
    station = PlanarPushingDiagram()
    return station.get_box()


def test_1(box: RigidBody):
    breakpoint()


if __name__ == "__main__":
    test_1(box())
