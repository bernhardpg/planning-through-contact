"""
Script for obtaining lists of faces and vertices from a union of axis-aligned planar
boxes. Planar means that the boxes are aligned with the XY plane and can be treated as
rectangles along this plane. The edges of the boxe's shouldn't overlap exctly as this
wouldn't occur with a fitting method.
The idea is to generate files like `t_pusher_2d.py` for any geometry consisting
of union of boxes.
NOTE: Much of this code was generated with help of ChatGPT and thus isn't clean or
consistent. This should be considered as initial prototyping.
"""

import matplotlib.pyplot as plt
import numpy as np

from planning_through_contact.geometry.collision_geometry.helpers import *


def main():
    # T-shape
    # TODO: Check that this actually is a T-shape and ensure some box overlap
    loaded_boxes = [
        {
            "name": "box1",
            "size": [0.2, 0.05, 0.05],
            "transform": np.eye(4),
        },
        {
            "name": "box2",
            "size": [0.05, 0.15001, 0.05],  # Require a small overlap
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.1],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]

    loaded_boxes = [
        {
            "name": "box1",
            "size": [0.3, 0.1, 0.05],
            "transform": np.eye(4),
        },
        {
            "name": "box2",
            "size": [0.15, 0.15, 0.05],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, -0.1],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "box3",
            "size": [0.1, 0.3, 0.05],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.12],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "box3",
            "size": [0.05, 0.05, 0.05],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, -0.15],
                    [0.0, 1.0, 0.0, 0.0],  # More challenging: [0.0,0.0,-0.05]
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]

    width, height = compute_union_dimensions(loaded_boxes)
    print(f"Union dimensions: width={width}, height={height}")

    # TODO: Factor plotting logic into class whose constructor creates the figure.
    plt.ion()
    fig, ax = plt.subplots()
    # Optionally set the aspect ratio to 'equal' to ensure that the unit of x is same as y
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    # Set limits for x and y axis
    x_min, x_max, y_min, y_max = compute_union_bounds_world(loaded_boxes)
    x_margin = 0.2 * width
    y_margin = 0.2 * height
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    plot_boxes(ax, loaded_boxes)

    outer_vertices = compute_outer_vertices(loaded_boxes)
    print("Number of outer vertices:", len(outer_vertices))
    # plot_vertices(ax, outer_vertices)

    outer_edges = compute_outer_edges(outer_vertices, loaded_boxes)
    print("Number of outer edges:", len(outer_edges))
    # plot_lines(ax, outer_edges)

    ordered_edges = order_edges_by_connectivity(outer_edges, loaded_boxes)
    # plot_lines(ax, ordered_edges)
    ordered_vertices = extract_ordered_vertices(ordered_edges)
    # plot_vertices(ax, ordered_vertices)
    
    normal_vecs = compute_normal_vecs_from_edges(ordered_edges, loaded_boxes)
    # plot_lines(ax, normal_vecs, color="black")

    planes_per_region, faces_per_region = compute_collision_free_regions(
        loaded_boxes, ordered_edges
    )
    for planes, faces in zip(planes_per_region, faces_per_region):
        color = np.random.rand(3)
        plot_lines(ax, planes, color=color)
        plot_lines(ax, faces, color=color)
        plt.pause(2.0)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
