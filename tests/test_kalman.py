import numpy as np
import pytest
from polar_tracking.kalman import _check_compatible_shape_linear, kalman_filter

class Test_check_compatibility_shape_linear:
    def test_pass(self):
        transition_matrix = np.zeros((2,2))
        observation_matrix = np.zeros((5, 2))
        observations = np.zeros((10,5))
        assert _check_compatible_shape_linear(transition_matrix,
                                              observation_matrix,
                                              observations)

    def test_fail_transition_square(self):
        transition_matrix = np.zeros((2,3))
        observation_matrix = np.zeros((5, 2))
        observations = np.zeros((10,5))
        assert not _check_compatible_shape_linear(transition_matrix,
                                                  observation_matrix,
                                                  observations)

    def test_fail_transition_observation(self):
        transition_matrix = np.zeros((2,2))
        observation_matrix = np.zeros((5, 3))
        observations = np.zeros((10,5))
        assert not _check_compatible_shape_linear(transition_matrix,
                                                  observation_matrix,
                                                  observations)

    def test_fail_data_observation(self):
        transition_matrix = np.zeros((2,2))
        observation_matrix = np.zeros((6, 2))
        observations = np.zeros((10,5))
        assert not _check_compatible_shape_linear(transition_matrix,
                                                  observation_matrix,
                                                  observations)

class Testkalman_filter:
    def test_ValueError(self):
        transition_matrix = np.zeros((2,3))
        observation_matrix = np.zeros((5, 2))
        observations = np.zeros((10,5))
        with pytest.raises(ValueError):
            kalman_filter(transition_matrix,
                          observation_matrix,
                          observations)
        
