"""Extended Kalman filters for tracking linear system dynamics.

This module contains implementation of extended Kalman filters applied
to two different types of dynamics.
 - As a baseline, it includes an implementation to track the linear dynamics
element-wise.
 - As an alternative, it includes an implementation tracking decompositions.

For convenience, it additionally includes a standard Kalman filter.
"""

import numpy as np
import numpy.typing as npt

def _check_compatible_shape_linear(transition_matrix: npt.NDArray,
                                   observation_matrix: npt.NDArray,
                                   observations: npt.NDArray
    ) -> bool:
    """Checks for compatible dimensions of a linear system

    Args:
        transition_matrix: State transition operator
        observation_matrix: Observation operator
        observations: 2D array of observations

    Returns:
        True if the shapes are compatible, otherwise false
    """

    out = True

    # Check transition matrix is square
    out &= transition_matrix.shape[0] == transition_matrix.shape[1]
    
    # Check if transition_matrix is compatible with observation matrix
    out &= transition_matrix.shape[0] == observation_matrix.shape[1]

    # Check if observations are compatible with observation matrix
    out &= observation_matrix.shape[0] == observations.shape[1]

    # Check if observation_matrix is compatible 
    return out


def kalman_filter_prediction(transition_matrix: npt.NDArray,
                             state_estimate: npt.NDArray,
                             state_covariance: npt.NDArray,
                             noise_covariance: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the Kalman Prediction Step.

    Given the current MMSE state estimate and the transition matrix,
    compute the new state estimate and covariance matrix.

    Args:
        transition_matrix: operator for advnacing the state
        state_estimate: Current MMSE state estimate
        state_covariance: Current estimate covariance
        noise_covariance: Covariance matrix of the system noise

    Returns:
        `(state_prediction, covariance_prediction)`
          where state_prediction is the updated state vector and 
          covariance_prediction is the covariance matrix of the prediction.
    """

    state_prediction = transition_matrix @ state_estimate

    covariance_prediction = transition_matrix \
                                @ state_covariance \
                                @ transition_matrix.T \
                                + noise_covariance

    return state_prediction, covariance_prediction


def kalman_gain(prediction_covariance: npt.NDArray,
                measurement_operator: npt.NDArray,
                measurement_covariance: npt.NDArray
    ) -> npt.NDArray:
    """Return the Kalman Gain for an Update

    Args:
        prediction_covariance: Covariance matrix of the prediction step
        measurement_operator: Matrix representing measurement operator
        measurement_covariance: Covarience of measurement noise

    Returns:
        The Kalman gain for the current timestep
    """

    innovation_covariance = measurement_operator \
                                @ prediction_covariance \
                                @ measurement_operator.T \
                                + measurement_covariance

    kalman_gain = prediction_covariance \
                    @ measurement_operator.T \
                    @ np.linalg.inv(innovation_covariance)

    return kalman_gain


def kalman_filter_update(measurement_matrix: npt.NDArray,
                         observation: npt.NDArray,
                         state_estimate: npt.NDArray,
                         state_covariance: npt.NDArray,
                         measurement_covariance: npt.NDArray 
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the Kalman Update Step

    Args:
        measurement_matrix: Matrix representing the measurement operator
        observation: Current observation vector
        state_estimate: State prediction for current timestep
        state_covariance: Covariance matrix for current state prediction
        measurement_covariance: Measurement covariance matrix

    Returns:
        Updated state estimate and covariance matrix of the estimate
    """

    gain = kalman_gain(state_covariance,
                       measurement_matrix,
                       measurement_covariance)

    state_estimate_update = state_estimate \
            - gain @ measurement_matrix @ state_estimate \
             + gain @ observation

    state_covariance_update = state_covariance \
            - gain @ measurement_matrix @ state_covariance

    return state_estimate_update, state_covariance_update


def kalman_filter(transition_matrix: npt.NDArray, 
                  observation_matrix: npt.NDArray, 
                  observations: npt.NDArray
    ) -> npt.NDArray:
    """Computes the Kalman filter estimates for a time series

    Args:
        transition_matrix: State transition operator, M by M matrix
        observation_matrix: Measurement operator, K by M matrix 
        observations: Array of measurements, timesteps by K matrix

    Returns:
        The filtered time series.

    Raises:
        ValueError: If the inputs are incompatible shapes

    """
    if not _check_compatible_shape_linear(transition_matrix, 
                                          observation_matrix, 
                                          observations):
        raise ValueError("Incompatible array data shapes in kalman_filter")

    return np.zeros(1)

