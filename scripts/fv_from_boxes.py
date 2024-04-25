"""
Script for obtaining lists of faces and vertices from a union of axis-aligned planar
boxes. Planar means that the boxes are aligned with the XY plane and can be treated as
rectangles along this plane. The edges of the boxe's shouldn't overlap exctly as this
wouldn't occur with a fitting method.
The idea is to generate files like `t_pusher_2d.py` for any geometry consisting
of union of boxes.
"""

import argparse
from itertools import combinations
from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_boxes(ax, loaded_boxes):
    # Create a rectangle for each box
    for box in loaded_boxes:
        # Extract size and transform
        size = box["size"]
        transform = box["transform"]

        # The bottom-left corner of the rectangle
        # transform[0, 3] and transform[1, 3] are the x and y offsets respectively
        bottom_left_x = transform[0, 3] - size[0] / 2
        bottom_left_y = transform[1, 3] - size[1] / 2

        # Create the rectangle
        rect = patches.Rectangle(
            (bottom_left_x, bottom_left_y),
            size[0],
            size[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the rectangle to the plot
        ax.add_patch(rect)


def plot_vertices(ax, vertices):
    # Unzip the list of vertices into separate x and y coordinate lists
    x_coords, y_coords = zip(*vertices)

    # Plot the vertices
    ax.plot(
        x_coords, y_coords, "o", markersize=5, linewidth=1
    )  # 'o-' creates a line with circle markers

    # Add labels (optional)
    for i, (x, y) in enumerate(vertices):
        ax.text(x, y, f" {i}", verticalalignment="bottom", horizontalalignment="right")


def plot_lines(ax, lines, color="green"):
    # Lines have format ((x1, y1), (x2, y2)).
    # Plot each edge and label it
    for index, edge in enumerate(lines):
        (start_x, start_y), (end_x, end_y) = edge
        # Plot the edge as a line
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            marker="o",
            label=f"Edge {index}",
            color=color,
        )

        # Calculate midpoint for the label placement
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(
            mid_x, mid_y, f"{index}", color=color, fontsize=12, ha="center", va="center"
        )


def compute_box_vertices(box):
    # Extract size and transformation matrix
    size = box["size"]
    transform = box["transform"]

    # Calculate the half-size for convenience
    half_size_x = size[0] / 2
    half_size_y = size[1] / 2

    # Define the relative corner points (vertices) of the box
    relative_vertices = np.array(
        [
            [-half_size_x, -half_size_y, 0, 1],  # Bottom-left
            [half_size_x, -half_size_y, 0, 1],  # Bottom-right
            [half_size_x, half_size_y, 0, 1],  # Top-right
            [-half_size_x, half_size_y, 0, 1],  # Top-left
        ]
    )

    # Apply the transformation matrix to each vertex
    vertices = [transform @ vertex for vertex in relative_vertices]

    # Extract only the x and y components to return 2D vertices
    vertices_2d = [(vertex[0], vertex[1]) for vertex in vertices]

    return vertices_2d


def compute_box_bounds(box):
    # Extract size and transformation matrix
    size = box["size"]
    transform = box["transform"]

    # Get the center position
    center_x = transform[0, 3]
    center_y = transform[1, 3]

    # Calculate the edges
    left = center_x - size[0] / 2
    right = center_x + size[0] / 2
    bottom = center_y - size[1] / 2
    top = center_y + size[1] / 2

    return left, right, bottom, top


def is_point_on_box_edge(point, box):
    """Check if the point is on the edge of the box."""
    x, y = point
    left, right, bottom, top = compute_box_bounds(box)
    return (
        (x == left or x == right)
        and bottom <= y <= top
        or (y == bottom or y == top)
        and left <= x <= right
    )


def compute_intersection_points(boxes):
    intersections = []

    # Assuming that boxes are axis-aligned and do not rotate
    for i, box1 in enumerate(boxes):
        # Compute the edges for box1
        left1, right1, bottom1, top1 = compute_box_bounds(box1)

        for j, box2 in enumerate(boxes):
            if i == j:  # Skip comparing the box with itself
                continue

            # Compute the edges for box2
            left2, right2, bottom2, top2 = compute_box_bounds(box2)

            # Check for horizontal overlap
            if left1 < right2 and right1 > left2:
                # Check for vertical overlap
                if bottom1 < top2 and top1 > bottom2:
                    # Calculate the intersection points
                    horizontal_overlap = [max(left1, left2), min(right1, right2)]
                    vertical_overlap = [max(bottom1, bottom2), min(top1, top2)]

                    # There are only 2 actual intersection points where the edges overlap
                    candidates = [
                        (horizontal_overlap[0], vertical_overlap[0]),
                        (horizontal_overlap[0], vertical_overlap[1]),
                        (horizontal_overlap[1], vertical_overlap[0]),
                        (horizontal_overlap[1], vertical_overlap[1]),
                    ]
                    for candidate in candidates:
                        if not is_inside_box(box1, candidate) and not is_inside_box(
                            box2, candidate
                        ):
                            intersections.append(candidate)

    # The intersection points calculated above include the corners of the overlapping area,
    # we need to filter out the points that are not actually intersection of the edges
    actual_intersections = []
    for point in intersections:
        if any(is_point_on_box_edge(point, box) for box in boxes):
            actual_intersections.append(point)

    return list(set(actual_intersections))  # Remove duplicates and return


def is_inside_box(box, point):
    """Check if a 2D point is inside the 2D rectangle on the XY plane defined by `box`.
    This returns False if the point is on the boundary of the box."""
    # Extract size and transformation matrix
    size = box["size"]
    transform = box["transform"]

    # Center of the box
    center_x = transform[0, 3]
    center_y = transform[1, 3]

    # Half-sizes of the box
    half_size_x = size[0] / 2
    half_size_y = size[1] / 2

    # Calculate the edges of the box
    left_edge = center_x - half_size_x
    right_edge = center_x + half_size_x
    bottom_edge = center_y - half_size_y
    top_edge = center_y + half_size_y

    # Point coordinates
    x, y = point

    # Check if the point is inside the box boundaries
    is_inside = left_edge < x < right_edge and bottom_edge < y < top_edge

    return is_inside


def is_inside_any_box(boxes, point):
    # Check if the vertex is inside any of the boxes
    for box in boxes:
        if is_inside_box(box, point):
            return True
    return False


def compute_box_bounds(box):
    # Extract size and transformation matrix
    size = box["size"]
    transform = box["transform"]

    # Get the center position
    center_x = transform[0, 3]
    center_y = transform[1, 3]

    # Calculate the edges
    left = center_x - size[0] / 2
    right = center_x + size[0] / 2
    bottom = center_y - size[1] / 2
    top = center_y + size[1] / 2

    return left, right, bottom, top


def compute_outer_vertices(boxes):
    # Compute all vertices and check if they are inside any other box
    box_vertices = [compute_box_vertices(box) for box in boxes]
    flattened_vertices = [vertex for vertices in box_vertices for vertex in vertices]
    intersection_points = compute_intersection_points(boxes)
    unique_points = list(set(flattened_vertices + intersection_points))
    outer_vertices = []
    for point in unique_points:
        if not is_inside_any_box(boxes, point):
            outer_vertices.append(point)
    return np.array(outer_vertices)


def connect_all_points(points):
    """Generate all possible edges between the given points."""
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            edges.append((points[i], points[j]))
    return edges


def filter_axis_aligned_edges(edges):
    """Filter out edges that are not perfectly horizontal or vertical."""
    axis_aligned_edges = []

    for edge in edges:
        (start_x, start_y), (end_x, end_y) = edge
        # Check if the edge is horizontal (same y) or vertical (same x)
        if start_x == end_x or start_y == end_y:
            axis_aligned_edges.append(edge)

    return axis_aligned_edges


def filter_edges_with_midpoint_collision(edges, boxes):
    """Filter out all edges whose midpoint is inside another box."""
    filtered_edges = []
    for edge in edges:
        midpoint = (edge[0] + edge[1]) / 2
        if not is_inside_any_box(boxes, midpoint):
            filtered_edges.append(edge)
    return filtered_edges


def line_intersects_box(line, box, num_samples=10):
    p1, p2 = line

    # Sample points along the line. Don't include the endpoints.
    x_values = np.linspace(p1[0], p2[0], num_samples + 2)[1:-1]
    y_values = np.linspace(p1[1], p2[1], num_samples + 2)[1:-1]
    points = np.column_stack((x_values, y_values))

    # Check if any of the points is inside the box
    for point in points:
        if is_inside_box(box, point):
            return True
    return False


def filter_edges_with_collision(edges, boxes):
    """Filter out all edges whose non-endpoints part is inside another box."""
    filtered_edges = []
    for edge in edges:
        intersects = False
        for box in boxes:
            if line_intersects_box(edge, box):
                intersects = True
                break
        if not intersects:
            filtered_edges.append(edge)
    return filtered_edges


def compute_outer_edges(vertices, boxes):
    edges = connect_all_points(vertices)
    axis_aligned_edges = filter_axis_aligned_edges(edges)
    outer_edges = filter_edges_with_collision(axis_aligned_edges, boxes)
    return outer_edges


def find_next_edge(edges, current_edge):
    """
    Find the next edge that connects to the current edge.
    """
    last_vertex = current_edge[1]  # The end vertex of the current edge
    for edge in edges:
        if np.array_equal(edge[0], last_vertex):
            return edge
        if np.array_equal(edge[1], last_vertex):
            return (edge[1], edge[0])  # Reverse the edge if needed
    return None


def order_edges_by_connectivity(edges):
    """
    Order edges by greedy connectivity starting from an arbitrary edge.
    """
    if not edges:
        return []

    # Convert edges to tuple format if they are numpy arrays
    edges = [
        (tuple(edge[0]), tuple(edge[1])) if isinstance(edge[0], np.ndarray) else edge
        for edge in edges
    ]

    ordered_edges = [edges[0]]  # Start with the first edge
    remaining_edges = list(edges[1:])

    while remaining_edges:
        next_edge = find_next_edge(remaining_edges, ordered_edges[-1])
        if next_edge:
            ordered_edges.append(next_edge)
            # Remove the edge from remaining_edges, considering potential tuple format
            remaining_edges.remove(
                next_edge
                if next_edge in remaining_edges
                else (tuple(next_edge[1]), tuple(next_edge[0]))
            )
        else:
            break  # No more connectable edges, stop here

    return ordered_edges


def extract_ordered_vertices(ordered_edges):
    """
    Extract the ordered vertices from the list of ordered edges.
    """
    ordered_vertices = [ordered_edges[0][0]]
    for edge in ordered_edges:
        ordered_vertices.append(edge[1])
    # The last vertex of the last edge is the same as the first vertex of the first
    # edge, so remove the last vertex to avoid duplication.
    ordered_vertices.pop()
    return ordered_vertices


def main():
    # T-shape
    # TODO: Check that this actually is a T-shape and ensure some box overlap
    loaded_boxes = [
        {
            "name": "box1",
            "size": [3.0, 1.0, 1.0],
            "transform": np.eye(4),
        },
        {
            "name": "box2",
            "size": [1.0, 3.0, 1.0],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]

    # loaded_boxes = [
    #     {
    #         "name": "box1",
    #         "size": [3.0, 1.0, 1.0],
    #         "transform": np.eye(4),
    #     },
    #     {
    #         "name": "box2",
    #         "size": [1.5, 1.5, 1.0],
    #         "transform": np.array(
    #             [
    #                 [1.0, 0.0, 0.0, 1.0],
    #                 [0.0, 1.0, 0.0, -1.0],
    #                 [0.0, 0.0, 1.0, 0.0],
    #                 [0.0, 0.0, 0.0, 1.0],
    #             ]
    #         ),
    #     },
    #     {
    #         "name": "box3",
    #         "size": [1.0, 3.0, 1.0],
    #         "transform": np.array(
    #             [
    #                 [1.0, 0.0, 0.0, 0.0],
    #                 [0.0, 1.0, 0.0, -1.2],
    #                 [0.0, 0.0, 1.0, 0.0],
    #                 [0.0, 0.0, 0.0, 1.0],
    #             ]
    #         ),
    #     },
    #     {
    #         "name": "box3",
    #         "size": [0.5, 0.5, 1.0],
    #         "transform": np.array(
    #             [
    #                 [1.0, 0.0, 0.0, -1.5],
    #                 [0.0, 1.0, 0.0, -0.5],
    #                 [0.0, 0.0, 1.0, 0.0],
    #                 [0.0, 0.0, 0.0, 1.0],
    #             ]
    #         ),
    #     },
    # ]

    # TODO: Factor plotting logic into class whose constructor creates the figure.
    fig, ax = plt.subplots()
    # Optionally set the aspect ratio to 'equal' to ensure that the unit of x is same as y
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    # Set limits for x and y axis
    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 1)
    plot_boxes(ax, loaded_boxes)

    outer_vertices = compute_outer_vertices(loaded_boxes)
    print("Number of outer vertices:", len(outer_vertices))
    # plot_vertices(ax, outer_vertices)

    outer_edges = compute_outer_edges(outer_vertices, loaded_boxes)
    print("Number of outer edges:", len(outer_edges))
    # plot_lines(ax, outer_edges)

    ordered_edges = order_edges_by_connectivity(outer_edges)
    plot_lines(ax, ordered_edges)
    ordered_vertices = extract_ordered_vertices(ordered_edges)
    plot_vertices(ax, ordered_vertices)

    plt.show()


if __name__ == "__main__":
    main()
