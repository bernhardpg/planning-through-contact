import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal

from planning_through_contact.tools.math import (
    null_space_basis_qr_pivot,
    permutation_matrix_from_vec,
)


def test_permutation_matrix_from_vec() -> None:
    # fmt: off
    perm_vector = [2, 1, 0]
    expected_output = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    # fmt: on
    result = permutation_matrix_from_vec(perm_vector)
    np.testing.assert_array_almost_equal(result, expected_output, decimal=7)

    # fmt: off
    perm_vector = [1, 2, 0]
    expected_output = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    # fmt: on
    result = permutation_matrix_from_vec(perm_vector)
    np.testing.assert_array_almost_equal(result, expected_output, decimal=7)

    # fmt: off
    perm_vector = [0, 1, 2]
    expected_output = np.eye(3)
    # fmt: on
    result = permutation_matrix_from_vec(perm_vector)
    np.testing.assert_array_almost_equal(result, expected_output, decimal=7)


def test_nullspace_basis_qr_pivot() -> None:

    def _assert_linear_independence(N: npt.NDArray[np.float64]) -> None:
        num_cols = N.shape[1]
        basis_rank = np.linalg.matrix_rank(N)
        assert basis_rank == num_cols

    # Rank 2 matrix
    # fmt: off
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [7, 8, 9]])
    # fmt: on

    # Null space dimension should be 1
    N = null_space_basis_qr_pivot(A)
    assert N.shape[1] == 1
    # Check that the nullspace matrix actually lies in the nullspace
    assert_array_almost_equal(A @ N, 0)
    _assert_linear_independence(N)

    # Full rank matrix
    I = np.eye(3)  # (full rank)
    N = null_space_basis_qr_pivot(I)
    assert N.shape[1] == 0

    # Rank 1 matrix
    # fmt: off
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    # fmt: on

    N = null_space_basis_qr_pivot(A)
    assert N.shape[1] == 2
    # Check that the nullspace matrix actually lies in the nullspace
    assert_array_almost_equal(A @ N, 0)
    _assert_linear_independence(N)
