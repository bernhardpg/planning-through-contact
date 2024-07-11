import matplotlib.pyplot as plt
import numpy as np

make_sym = lambda a: np.array([[a[0], a[1]], [a[1], a[2]]])

A = make_sym([0, 1, -2])
B = make_sym([2, -2, 2])

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, -1], [4, 3]])


def generate_x_vectors(n, step_size, min_val, max_val):
    """
    Generate a list of x vectors in R^n within a specified range for each dimension with a given step size.

    Parameters:
    n (int): The dimension of the space R^n.
    step_size (float): The step size for each dimension.
    min_val (float): The minimum value for each dimension.
    max_val (float): The maximum value for each dimension.

    Returns:
    list: A list of np.array, each representing a point in R^n within the specified range.
    """
    # Use np.arange to create a range of values with the given step size for each dimension
    range_values = np.arange(min_val, max_val + step_size, step_size)

    # Use np.meshgrid to create a mesh grid of points in R^n
    grids = np.meshgrid(*([range_values] * n), indexing="ij")

    # Flatten the mesh grids and stack them to create the list of vectors
    x_vectors = np.stack(np.meshgrid(*[range_values] * n), -1).reshape(-1, n)

    return x_vectors


def sample_spherical(n_points, ndim=3):
    """
    Generate points uniformly distributed on the surface of a unit sphere in R^ndim.

    Parameters:
    n_points (int): Number of points to generate.
    ndim (int): Dimension of the space.

    Returns:
    np.array: Array of points on the unit sphere.
    """
    vec = np.random.randn(ndim, n_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


n = 2  # Dimension of space, R^2 for this example
step_size = 0.01  # Step size
min_val = -2  # Minimum value for each dimension
max_val = 2  # Maximum value for each dimension
x_values = generate_x_vectors(n, step_size, min_val, max_val)


n_points = 1000
sphere_points = sample_spherical(n_points, ndim=2)

# Calculate the corresponding y values for W(A, B)
F_values = np.array([(x.T @ A @ x, x.T @ B @ x) for x in sphere_points.T])
W_values = np.array([(x.T @ A @ x, x.T @ B @ x) for x in x_values])

axis_limit = np.max(np.abs(F_values)) * 1.2
plt.figure(figsize=(8, 8))
plt.scatter(W_values[:, 0], W_values[:, 1], alpha=0.3)
plt.scatter(F_values[:, 0], F_values[:, 1], alpha=0.3)
plt.title("Visualization of the sets")
plt.legend(["W(A,B)", "F(A,B)"])
plt.xlabel("x^T A x")
plt.ylabel("x^T B x")
plt.grid(True)
plt.axis("equal")  # This line sets equal scaling on the axes
# Set the same amount of negative and positive values on the axes
plt.xlim(-axis_limit, axis_limit)
plt.ylim(-axis_limit, axis_limit)
plt.show()
