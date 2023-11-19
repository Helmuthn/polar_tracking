"""Simulate Brownian Motion in Matrix Spaces

Warning:
    Currently only works for square matrices.
"""
import numpy.typing as npt
import numpy as np


def _covariance_tensor_to_matrix(covariance_tensor: npt.NDArray
    ) -> npt.NDArray:
    """Reshape the covariance tensor into a covariance matrix,

    Args:
        covariance_tensor: m by n by m by n covariance tensor

    Returns:
        An mn by mn covariance matrix

    Raises:
        ValueError: If the covariance_tensor shape is incorrect
    """
    if (len(covariance_tensor.shape) != 4 
            or covariance_tensor.shape[0]  
            != covariance_tensor.shape[2]  
            or covariance_tensor.shape[1] 
            != covariance_tensor.shape[3]):
                raise ValueError("Invalid shape for covariance_tensor")

    d1, d2 = covariance_tensor.shape[0:2]
    return covariance_tensor.reshape((d1*d2, d1*d2))


def brownian_motion_matrix_step(current_state: npt.NDArray,
                                covariance_tensor: npt.NDArray
    ) -> npt.NDArray:
    """Compute a step of Brownian motion in matrix-element space.

    Args:
        current_state: Current state vector, d by d array
        covariance_tensor: Covariance tensor for d by d array steps

    Returns:
        The updated state
    """
    d = current_state.shape[0] * current_state.shape[1]
    covariance_matrix = _covariance_tensor_to_matrix(covariance_tensor)
    step = np.random.multivariate_normal(np.zeros(d), covariance_matrix)
    step = step.reshape(current_state.shape)

    return current_state + step


def manifold_stiefel_projection(current_state: npt.NDArray,
                                proposed_step: npt.NDArray
    ) -> npt.NDArray:
    """Projects onto the tangent space of Stiefel Manifold

    Args:
        current_state: Current point on the stiefel manifold
        proposed_step: Matrix in the ambient space

    Returns:
        Projection of the proposed step
    """
    tmp = proposed_step.T @ current_state;
    tmp = (tmp + tmp.T)/2
    return proposed_step - current_state @ tmp


def brownian_motion_stiefel_step(current_state: npt.NDArray, 
                                 covariance_tensor: npt.NDArray
    ) -> npt.NDArray:
    """Brownian motion on the Stiefel manifold
    
    Given the current state and a covariance tensor in the
    ambient space, return an updated state from an approximation
    of Brownian motion on the Stiefel manifold.

    The step is computed by first sampling in the ambient matrix
    space, projecting the step onto the tangent space, then
    finally projecting the new state onto the manifold through
    QR factorization.

    Args:
        current_state: The current orthonormal matrix state
        covariance_tensor: Covariance tensor for the orthonormal matrix

    Returns:
        The updated state matrix.
    """
    proposed_step = brownian_motion_matrix_step(current_state,
                                                covariance_tensor)

    tangent_update = manifold_stiefel_projection(current_state,
                                                 proposed_step)

    new_state, _ = np.linalg.qr(tangent_update)
    return new_state

