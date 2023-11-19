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

class ExtendedKalmanFilter:
    """Extended Kalman Filter for application to nonlinear systems

    Represents and extended kalman filter constructed through a linearization
    of a nonlinear system.
    """

    def __init__(self,
                 transition_operator: npt.NDArray,
                 measurement_operator: npt.NDArray,
                 noise_covariance_system: npt.NDArray,
                 noise_covariance_measurement: npt.NDArray,
                 state_mean: npt.NDArray,
                 state_covariance: npt.NDArray):
        """Initialize a Kalman Filter given system dynamics and initialization

        Args:
            transition_operator: State Transition Operator, d by d 
            measurement_operator: Measurement Operator, k by d 
            noise_covariance_system: System noise covariance, d by d 
            noise_covariance_measurement: Measurement noise covariance, k by k
            state_mean: Initial state estimate, length d
            state_covariance: Initial estimate covariance, d by d
            
        Raises:
            ValueError: If the inputs are not compatible sizes
        """
        if self._check_valid_initialization_sizes(transition_operator,
                                                  measurement_operator,
                                                  noise_covariance_system,
                                                  noise_covariance_measurement,
                                                  state_mean,
                                                  state_covariance):
            raise ValueError("Incompatible System Component Dimensionalities")
            

        self.transition_operator = transition_operator
        self.measurement_operator = measurement_operator
        self.noise_covariance_system = noise_covariance_system
        self.noise_covariance_measurement = noise_covariance_measurement
        self.state_estimate = state_mean
        self.state_covariance = state_covariance


class KalmanFilter:
    """Kalman Filter for linear systems

    Representation of a Kalman filter for linear systems.
    Initialized with the system dynamics and initial estimate
    statistics. The function `KalmanFilter.update` accepts
    a new measurement and computes the updated state estimate
    and covariance matrix. 
    The internal state of the Kalman filter is updated as well.
    """

    def __init__(self,
                 transition_operator: npt.NDArray,
                 measurement_operator: npt.NDArray,
                 noise_covariance_system: npt.NDArray,
                 noise_covariance_measurement: npt.NDArray,
                 state_mean: npt.NDArray,
                 state_covariance: npt.NDArray):
        """Initialize a Kalman Filter given system dynamics and initialization

        Args:
            transition_operator: State Transition Operator, d by d 
            measurement_operator: Measurement Operator, k by d 
            noise_covariance_system: System noise covariance, d by d 
            noise_covariance_measurement: Measurement noise covariance, k by k
            state_mean: Initial state estimate, length d
            state_covariance: Initial estimate covariance, d by d
            
        Raises:
            ValueError: If the inputs are not compatible sizes
        """
        if self._check_valid_initialization_sizes(transition_operator,
                                                  measurement_operator,
                                                  noise_covariance_system,
                                                  noise_covariance_measurement,
                                                  state_mean,
                                                  state_covariance):
            raise ValueError("Incompatible System Component Dimensionalities")
            

        self.transition_operator = transition_operator
        self.measurement_operator = measurement_operator
        self.noise_covariance_system = noise_covariance_system
        self.noise_covariance_measurement = noise_covariance_measurement
        self.state_estimate = state_mean
        self.state_covariance = state_covariance


    def _check_valid_initialization_sizes(self,
                 transition_operator: npt.NDArray,
                 measurement_operator: npt.NDArray,
                 noise_covariance_system: npt.NDArray,
                 noise_covariance_measurement: npt.NDArray,
                 state_mean: npt.NDArray,
                 state_covariance: npt.NDArray) -> bool:
        """Checks for compatible sizes in initialization

        See `__init__` for more details on requirements.
        """

        if (len(transition_operator.shape) != 2 
            or len(measurement_operator.shape) != 2
            or len(noise_covariance_system.shape) != 2
            or len(noise_covariance_measurement.shape) != 2
            or len(state_mean.shape) != 1
            or len(state_covariance.shape) != 2):
            return False

        out = transition_operator.shape[0] == transition_operator.shape[1]
        out &= transition_operator.shape[0] == measurement_operator.shape[1]
        out &= transition_operator.shape[0] == noise_covariance_system.shape[0]
        out &= transition_operator.shape[0] == state_mean.shape[0]
        out &= transition_operator.shape[0] == state_covariance.shape[0]

        out &= measurement_operator.shape[0] == noise_covariance_measurement.shape[0]
        
        out &= noise_covariance_system.shape[0] == noise_covariance_system.shape[1]

        out &= noise_covariance_measurement.shape[0] == noise_covariance_measurement.shape[0]
        out &= state_covariance.shape[0] == state_covariance.shape[1]

        return out



    def _prediction(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Computes the MMSE prediction and associated covariance

        Compute the prediction step of the Kalman filter algorithm.
        Does not update the internal state.

        Returns:
            `(state_prediction, covariance_prediction)`
              where state_prediction is the updated state vector and 
              covariance_prediction is the covariance matrix of the prediction.
            """
        state_prediction = self.transition_operator @ self.state_estimate

        covariance_prediction = self.transition_operator \
                                    @ self.state_covariance \
                                    @ self.transition_operator.T \
                                    + self.noise_covariance_system

        return state_prediction, covariance_prediction


    def _update_prediction(self, 
                           observation: npt.NDArray, 
                           state_prediction: npt.NDArray, 
                           prediction_covariance: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray]:
        """Computes the Kalman Update Step

        Given an observation vector and previous 

        Args:
            observation: Current observation vector
            state_prediction: State prediction for current timestep
            prediction_covariance: Covariance matrix for current state prediction

        Returns:
            Updated state estimate and covariance matrix of the estimate
        """
        gain = self._kalman_gain(prediction_covariance)

        state_estimate_update = state_prediction \
                - gain @ self.measurement_operator @ state_prediction \
                 + gain @ observation

        state_covariance_update = prediction_covariance \
                - gain @ self.measurement_operator @ prediction_covariance

        return state_estimate_update, state_covariance_update


    def _kalman_gain(self, prediction_covariance) -> npt.NDArray:
        """Computes the kalman gain given the prediction covariance

        Args:
            prediction_covariance: Covariance matrix for current stat prediction

        Returns:
            The computed Kalman gain
        """

        innovation_covariance = self.measurement_operator \
                                    @ prediction_covariance \
                                    @ self.measurement_operator.T \
                                    + self.noise_covariance_measurement

        kalman_gain = prediction_covariance \
                        @ self.measurement_operator.T \
                        @ np.linalg.inv(innovation_covariance)

        return kalman_gain


    def update(self, observation: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray]:
        """Update the Kalman Filter given a new observation

        Computes a full update step to the Kalman filter given a new
        observation. Updates the internal state estimate and covariance
        and returns the new values.

        Args:
            observation: New observation vector

        Returns:
            `(state_estimate, state_covariance)` 
              where `state_estimate` is the updated state estimate and 
              `state_covariance` is the covariance matrix

        Raises:
            ValueError: If the observation is not a compatible shape.

        """
        if (len(observation.shape) != 1 
            or observation.shape[0] != self.measurement_operator.shape[0]):
            raise ValueError(f"Invalid Measurement Shape, expected vector of length {self.measurement_operator.shape[0]}")

        prediction, prediction_covariance = self._prediction()
        estimate, covariance = self._update_prediction(observation,
                                                       prediction,
                                                       prediction_covariance)

        self.state_estimate = estimate 
        self.state_covariance = covariance

        return self.state_estimate, self.state_covariance


    def get_state_estimate(self) -> npt.NDArray:
        return self.state_estimate

    def get_state_covariance(self) -> npt.NDArray:
        return self.state_covariance

    def get_state_prediction(self) -> npt.NDArray:
        return self.transition_operator @ self.state_estimate

    def get_state_prediction_covariance(self) -> npt.NDArray:
        _, covariance = self._prediction()
        return covariance


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

    Warning:
        Not Yet Implemented
    """
    if not _check_compatible_shape_linear(transition_matrix, 
                                          observation_matrix, 
                                          observations):
        raise ValueError("Incompatible array data shapes in kalman_filter")

    return np.zeros(1)

