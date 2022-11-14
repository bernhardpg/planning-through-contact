from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import VPolytope


def visualize_polytopes(polytopes: List[VPolytope]) -> None:
    temp = [p.vertices() for p in polytopes]
    vertices = [np.concatenate([t, t[0:1, :]]) for t in temp]
    for vs in vertices:
        plt.plot(vs[:, 0], vs[:, 1])
    plt.show()


def create_test_polytopes() -> List[VPolytope]:
    vertices = [
        np.array([[-1, 0], [0, 0], [0, 3], [-1, 3]]),
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.array([[0, 1], [1, 1], [1, 2], [0, 2]]),
        np.array([[0, 2], [1, 2], [1, 3], [0, 3]]),
        np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
        np.array([[1, 1], [2, 1], [2, 2], [1, 2]]),
        np.array([[1, 2], [2, 2], [2, 3], [1, 3]]),
        np.array([[2, 0], [3, 0], [3, 1], [2, 1]]),
        np.array([[2, 1], [3, 1], [3, 2], [2, 2]]),
        np.array([[2, 2], [3, 2], [3, 3], [2, 3]]),
        np.array([[3, 1], [4, 1], [4, 2], [3, 2]]),
    ]
    polytopes = [VPolytope(vs) for vs in vertices]
    return polytopes


def gcs_a_star():
    print("Running GCS A* demo")

    polytopes = create_test_polytopes()
    visualize_polytopes(polytopes)
    breakpoint()

    # Define source and target
    # Run hierarchical GCS
