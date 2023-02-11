from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt


class Hyperplane(NamedTuple):
    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]


def get_angle_between_planes(plane_A: Hyperplane, plane_B: Hyperplane) -> float:
    theta = np.arccos(plane_B.a.T.dot(plane_A.a))[0, 0]
    return theta


def construct_2d_plane_from_points(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> Hyperplane:
    diff = p2 - p1
    normal_vec = np.array([-diff[1], diff[0]]).reshape((-1, 1))
    a = normal_vec / np.linalg.norm(normal_vec)
    b = a.T.dot(p1)
    return Hyperplane(a, b)


def calc_intersection_with_so_2(
    plane_2d: Hyperplane,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    a_x = plane_2d.a[0, 0]
    a_y = plane_2d.a[1, 0]

    if a_x == 0:
        c_th_val = np.sqrt(1 / (1 + (a_x / a_y) ** 2))
        s_th_val = -(a_x / a_y) * c_th_val
    else:
        s_th_val = np.sqrt(1 / (1 + (a_y / a_x) ** 2))
        c_th_val = -(a_y / a_x) * s_th_val

    p1 = np.array([c_th_val, s_th_val]).reshape((-1, 1))
    p2 = -p1
    return p1, p2


def calculate_convex_hull_cut_for_so_2(
    plane_1: Hyperplane, plane_2: Hyperplane
) -> Hyperplane:
    # All intersection points with SO(2)
    p1, p2 = calc_intersection_with_so_2(plane_1)
    p3, p4 = calc_intersection_with_so_2(plane_2)

    # From the four intersecting points, we pick the two that are on the positive side of the nonpenetration hyperplanes
    cut_point_1 = p1 if plane_2.a.T.dot(p1) >= 0 else p2
    cut_point_2 = p3 if plane_1.a.T.dot(p3) >= 0 else p4

    # Sort the points so that we always get the plane with a normal pointing outwards
    # (sort by angle in polar coordinates)
    cut_points = sorted(
        [cut_point_1, cut_point_2], key=lambda p: np.arctan2(p[0], p[1])
    )

    so_2_cut = construct_2d_plane_from_points(*cut_points)

    return so_2_cut
