from geometry.two_d.box_2d import Box2d
from geometry.two_d.contact.types import ContactMode, ContactPosition, ContactType
from geometry.two_d.equilateral_polytope_2d import EquilateralPolytope2d
from geometry.two_d.rigid_body_2d import PolytopeContactLocation


def plan_triangle_flipup():
    TABLE_HEIGHT = 0.5
    TABLE_WIDTH = 2

    FINGER_HEIGHT = 0.1
    FINGER_WIDTH = 0.1

    TRIANGLE_MASS = 1
    FRICTION_COEFF = 0.7

    triangle = EquilateralPolytope2d(
        actuated=False,
        name="triangle",
        mass=TRIANGLE_MASS,
        vertex_distance=0.2,
        num_vertices=3,
    )
    table = Box2d(
        actuated=True,
        name="table",
        mass=None,
        width=TABLE_WIDTH,
        height=TABLE_HEIGHT,
    )
    finger = Box2d(
        actuated=True,
        name="finger",
        mass=None,
        width=FINGER_WIDTH,
        height=FINGER_HEIGHT,
    )



if __name__ == "__main__":
    plan_triangle_flipup()
