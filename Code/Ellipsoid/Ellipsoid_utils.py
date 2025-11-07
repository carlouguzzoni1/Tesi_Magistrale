# ======================================================================
#  Utility functions for Ellipsoid.
# ======================================================================

import numpy as np


def gram_schmidt_basis(u_b):
    """
    Original implementation of GS to obtain T.
    
    Args:
        u_b: The first vector of the basis.
        
    Returns:
        T: The change of basis matrix.
    """
    
    # Prepare the change of basis matrix from the standard basis to the
    # one with first vector u_b.
    n = len(u_b)
    e = np.eye(n)
    T = np.zeros((n, n))
    T[:, 0] = u_b

    # Gram-Schmidt orthogonalization.
    for k in range(1, n):
        z_k = np.zeros(n)
        
        for i in range(k):
            z_k += ((e[:, k] @ T[:, i]) / (T[:, i] @ T[:, i])) * T[:, i]
        
        T[:, k] = e[:, k] - z_k
    
    # Normalize the resulting basis.
    for k in range(n):
        T[:, k] /= np.linalg.norm(T[:, k])
    
    return T


def qr_basis(u_b):
    """
    New method based on QR decomposition.
    
    Args:
        u_b: The first vector of the basis.
        
    Returns:
        T: The change of basis matrix.
    """
    
    # Build the initial matrix: u_b + identity as remaining basis.
    n = len(u_b)
    E = np.eye(n)
    M = np.column_stack([u_b, E[:, 1:]])

    # QR decomposition: Q gives orthonormal basis.
    Q, _ = np.linalg.qr(M)
    
    return Q