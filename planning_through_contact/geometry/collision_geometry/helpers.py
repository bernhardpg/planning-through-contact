import matplotlib.patches as patches
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


def compute_box_union_bounds(boxes):
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for box in boxes:
        left, right, bottom, top = compute_box_bounds(box)
        min_x = min(min_x, left)
        max_x = max(max_x, right)
        min_y = min(min_y, bottom)
        max_y = max(max_y, top)

    return min_x, max_x, min_y, max_y


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


def direct_edges_so_right_points_inside(edges, boxes):
    """
    Flip edges so that the right side next to the edge is inside a box.
    """
    width, height = compute_union_dimensions(boxes)
    scale = min(width, height) / 1000

    directed_edges = []
    for edge in edges:
        # Compute the midpoint of the edge.
        midpoint = (edge[0] + edge[1]) / 2

        # Compute point to the right of the edge.
        start, end = edge
        diff0 = end[0] - start[0]
        diff1 = end[1] - start[1]
        normal = np.array([diff1, -diff0])
        normal /= np.linalg.norm(normal)
        normal *= scale * 100  # Scale for visualization
        scaled_normal = normal * scale  # Scale for collision checking
        right_point = (midpoint[0] + scaled_normal[0], midpoint[1] + scaled_normal[1])

        # Check if the right point is inside a box.
        if is_inside_any_box(boxes, right_point):
            directed_edges.append(edge)
        else:
            directed_edges.append((edge[1], edge[0]))

    return directed_edges


def order_edges_by_connectivity(edges, boxes):
    """
    Order edges by greedy connectivity starting from an arbitrary edge.
    Ensures that the first vertex of the first edge is not an intersection vertex.
    """
    if not edges:
        return []

    # Convert edges to tuple format if they are numpy arrays
    edges = [
        (tuple(edge[0]), tuple(edge[1])) if isinstance(edge[0], np.ndarray) else edge
        for edge in edges
    ]

    # Start with edge that has highest y-coordinate of the first vertex.
    edges.sort(key=lambda edge: edge[0][1], reverse=True)

    intersection_vertices = compute_intersection_points(boxes)
    if edges[0][0] in intersection_vertices:
        # Use the 2nd edge as the starting edge.
        edges = edges[1:] + [edges[0]]

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


def compute_union_dimensions(boxes):
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for box in boxes:
        # Extract the position from the transformation matrix
        x_center, y_center, _ = box["transform"][:3, 3]

        # Half-sizes for x and y dimensions
        half_size_x, half_size_y = box["size"][0] / 2, box["size"][1] / 2

        # Calculate min and max coordinates for this box
        box_min_x = x_center - half_size_x
        box_max_x = x_center + half_size_x
        box_min_y = y_center - half_size_y
        box_max_y = y_center + half_size_y

        # Update the overall min and max
        min_x = min(min_x, box_min_x)
        max_x = max(max_x, box_max_x)
        min_y = min(min_y, box_min_y)
        max_y = max(max_y, box_max_y)

    # Compute the width and height of the union of the boxes
    width = max_x - min_x
    height = max_y - min_y

    return width, height


def compute_union_bounds_world(boxes):
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for box in boxes:
        # Extract the position from the transformation matrix
        x_center, y_center, _ = box["transform"][:3, 3]

        # Half-sizes for x and y dimensions
        half_size_x, half_size_y = box["size"][0] / 2, box["size"][1] / 2

        # Calculate min and max coordinates for this box
        box_min_x = x_center - half_size_x
        box_max_x = x_center + half_size_x
        box_min_y = y_center - half_size_y
        box_max_y = y_center + half_size_y

        # Update the overall min and max
        min_x = min(min_x, box_min_x)
        max_x = max(max_x, box_max_x)
        min_y = min(min_y, box_min_y)
        max_y = max(max_y, box_max_y)

    return min_x, max_x, min_y, max_y


def compute_collision_free_regions(boxes, faces):
    # This assumes that the first vertex of the first face is not an intersection vertex
    # Keeps the ordering of the faces.

    width, height = compute_union_dimensions(boxes)
    scale = min(width, height) / 1000

    intersection_vertices = compute_intersection_points(boxes)

    assert faces[0][0] not in intersection_vertices

    UL = np.array([-1, 1]) * scale
    UR = np.array([1, 1]) * scale
    DR = np.array([1, -1]) * scale
    DL = np.array([-1, -1]) * scale

    def get_offset(vertex):
        if is_inside_any_box(boxes, vertex + UL) and not is_inside_any_box(
            boxes, vertex + DR
        ):
            return DR * 100
        if is_inside_any_box(boxes, vertex + UR) and not is_inside_any_box(
            boxes, vertex + DL
        ):
            return DL * 100
        if is_inside_any_box(boxes, vertex + DR) and not is_inside_any_box(
            boxes, vertex + UL
        ):
            return UL * 100
        if is_inside_any_box(boxes, vertex + DL) and not is_inside_any_box(
            boxes, vertex + UR
        ):
            return UR * 100
        raise ValueError("No offset found")

    planes_per_region = []
    faces_per_region = []
    current_idx = 0
    while current_idx < len(faces):
        region_faces = []
        face = faces[current_idx]
        region_faces.append(face)
        incoming_plane = (face[0] + get_offset(face[0]), face[0])
        while face[1] in intersection_vertices:
            face = faces[current_idx + 1]
            region_faces.append(face)
            current_idx += 1
        outgoing_plane = (face[1], face[1] + get_offset(face[1]))
        planes_per_region.append([incoming_plane, outgoing_plane])
        faces_per_region.append(region_faces)
        current_idx += 1

    return planes_per_region, faces_per_region


def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def compute_normal_vecs_from_edges(edges, boxes):
    """
    Compute normal vectors for edges that start at the edge midpoint and point inside
    the box.
    """
    width, height = compute_union_dimensions(boxes)
    scale = min(width, height) / 1000

    normal_vecs = []
    for edge in edges:
        start, end = edge
        diff0 = end[0] - start[0]
        diff1 = end[1] - start[1]
        normal = np.array([diff1, -diff0])
        normal /= np.linalg.norm(normal)
        normal *= scale * 100  # Scale for visualization
        scaled_normal = normal * scale  # Scale for collision checking

        mid_point = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

        # Make sure that normal vector points inside the box
        if is_inside_any_box(
            boxes, (mid_point[0] + scaled_normal[0], mid_point[1] + scaled_normal[1])
        ):
            normal_vec = (
                (mid_point[0], mid_point[1]),
                (mid_point[0] + normal[0], mid_point[1] + normal[1]),
            )
        else:
            normal_vec = (
                (mid_point[0], mid_point[1]),
                (mid_point[0] - normal[0], mid_point[1] - normal[1]),
            )
        normal_vecs.append(normal_vec)
    return normal_vecs


def compute_normalized_normal_vector_points_from_edges(edges, boxes):
    """
    Computes a point that defines the normal vector with respect to the world origin.
    The normal vectors point inside the shape.
    """
    normal_vecs = compute_normal_vecs_from_edges(edges, boxes)
    normal_vec_points = [
        np.array([np.array(vec[1]) - np.array(vec[0])]).reshape((-1, 1))
        for vec in normal_vecs
    ]
    normalized_normal_vec_points = [
        vec / np.linalg.norm(vec) for vec in normal_vec_points
    ]
    return normalized_normal_vec_points


def compute_com_from_uniform_density(boxes):
    # Initialize variables to accumulate weighted centroids and total area
    sum_weighted_x = 0
    sum_weighted_y = 0
    total_area = 0

    # Iterate through each box
    for box in boxes:
        # Extract size and transformation
        size = box["size"]
        transform = box["transform"]

        # Calculate area (ignoring z-dimension)
        width, height = size[0], size[1]
        area = width * height

        # Calculate the centroid of the box
        x_centroid = transform[0, 3]
        y_centroid = transform[1, 3]

        # Accumulate weighted centroids and total area
        sum_weighted_x += x_centroid * area
        sum_weighted_y += y_centroid * area
        total_area += area

    # Calculate the center of mass
    center_of_mass_x = sum_weighted_x / total_area
    center_of_mass_y = sum_weighted_y / total_area

    return center_of_mass_x, center_of_mass_y


def offset_boxes(boxes, offset):
    for box in boxes:
        box["transform"][0, 3] += offset[0]
        box["transform"][1, 3] += offset[1]
    return boxes
