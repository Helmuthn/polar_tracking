import numpy as np

from polar_tracking.brownian import _covariance_tensor_to_matrix
from polar_tracking.brownian import manifold_stiefel_projection
from polar_tracking.brownian import brownian_motion_matrix_step
from polar_tracking.brownian import brownian_motion_stiefel_step

class Testmanifold_stiefel_projection:
    def test_basic(self):
        """Spot check that the constraint is satisfied

        The tangent space of a the Stiefel manifold at U is
        the set of matrices V such that
        U.T @ V + V.T @ U = 0
        """
        U = np.zeros((2,2))
        U[0,0] = 1
        U[1,1] = 1
        candidate = np.array([[1,2],[3,4]])
        V = manifold_stiefel_projection(U, candidate)
        assert np.sum(np.abs(U.T @ V + V.T @ U)) < 0.001


class Test_covariance_tensor_to_matrix:
    def test_basic_rectangle_shape(self):
        """Tests that the shape of of the output matrix is correct
        for a rectangular input."""
        covariance_tensor = np.ones((2,3,2,3))
        out = _covariance_tensor_to_matrix(covariance_tensor)
        assert out.shape == (6,6)


class Testbrownian_motion_stiefel_step:
    def test_orthogonal_result(self):
        """Spot check that the output matrix has orthogonal rows"""
        covariance_tensor = np.zeros((2,2,2,2))
        covariance_tensor[0,0,0,0] = 1
        covariance_tensor[1,1,1,1] = 1
        covariance_tensor[0,0,1,1] = 1
        covariance_tensor[1,1,0,0] = 1

        current_state = np.ones((2,2))
        current_state[0,1] = 0
        current_state[1,0] = 0

        updated = brownian_motion_stiefel_step(current_state, covariance_tensor)

        assert np.abs(np.dot(updated[0,:], updated[1,:])) < 0.001

    def test_normalized_result(self):
        """Spot check that the output has unit norm rows"""
        covariance_tensor = np.zeros((2,2,2,2))
        covariance_tensor[0,0,0,0] = 1
        covariance_tensor[1,1,1,1] = 1
        covariance_tensor[0,0,1,1] = 1
        covariance_tensor[1,1,0,0] = 1

        current_state = np.ones((2,2))
        current_state[0,1] = 0
        current_state[1,0] = 0

        updated = brownian_motion_stiefel_step(current_state, covariance_tensor)

        assert np.abs(np.linalg.norm(updated[0,:]) - 1) < 0.001
        assert np.abs(np.linalg.norm(updated[1,:]) - 1) < 0.001
