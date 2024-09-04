import numpy as np
import numpy.typing as npt
from scipy.linalg import qr


def null_space_basis_svd(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    U, s, V_hermitian_transpose = np.linalg.svd(A)
    eps = 1e-6
    zero_idxs = np.where(np.abs(s) <= eps)[0].tolist()
    V = V_hermitian_transpose.T  # Real matrix, so conjugate transpose = transpose

    remaining_idxs = list(range(len(s), len(V)))

    V_zero = V[:, zero_idxs + remaining_idxs]

    nullspace_matrix = V_zero
    return nullspace_matrix


def permutation_matrix_from_vec(
    vec: npt.NDArray[np.int32] | list[int],
) -> npt.NDArray[np.float64]:
    """
    Efficiently create a permutation matrix from a given permutation vector.

    Parameters:
    perm_vector (numpy.ndarray or list): A permutation vector that represents the reordering of columns.

    Returns:
    numpy.ndarray: The permutation matrix corresponding to the given permutation vector.
    """
    n = len(vec)
    I = np.eye(n, dtype=int)  # Create an identity matrix of size n x n
    P = I[:, np.array(vec)]  # Reorder columns of the identity matrix
    return P


def null_space_basis_qr_pivot(A: npt.NDArray[np.float64], tol=1e-12):
    """
    Compute a basis for the null space of matrix A using QR decomposition with pivoting.

    Parameters:
    A (numpy.ndarray): The input matrix A (m × n)
    tol (float): Tolerance to determine the rank (for numerical stability)

    Returns:
    numpy.ndarray: A matrix whose columns form a basis for the null space of A
    """
    # Perform QR decomposition with column pivoting: AP = QR
    # A: n x m
    # Q: n x n (orthogonal)
    # R: n x m (upper triangular)
    # p: n x 1 (permutation vector)
    Q, R, p = qr(A, pivoting=True)

    P = permutation_matrix_from_vec(p)

    # Determine the rank based on the tolerance
    # (remember that R is upper triangular with nonzero diagonal entries)
    diag_R = np.abs(np.diag(R))
    rank = int(np.sum(diag_R > tol))

    # Partition R into R₁ and R₂
    R1 = R[:rank, :rank]  # Upper triangular part
    R2 = R[:rank, rank:]  # Remaining columns

    # Form the matrix: [-R₁⁻¹ R₂; I]
    R1_inv_R2 = -np.linalg.solve(R1, R2)  # Solving R₁ * X = R₂ gives X = R₁⁻¹R₂
    identity_block = np.eye(A.shape[1] - rank)

    # The null space basis matrix in terms of permutation P
    null_space_matrix = np.vstack([R1_inv_R2, identity_block])

    # Apply the permutation matrix P to get the null space basis in the correct order
    null_space_basis = P @ null_space_matrix

    return null_space_basis
