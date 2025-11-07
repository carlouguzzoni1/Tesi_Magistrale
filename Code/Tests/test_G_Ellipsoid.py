# Test script used to:
# - Compare the original orthogonalization (Gram-Schmidt) with the
# new optimized one (QR decomposition).
# - Verify whether the original formula for G (T.T @ A @ T) is consistent
# with the paper.
# Overall, just a bunch of prints for visual inspection.

import numpy as np



# --- Dummy data ---

w       = np.array([0.2, 0.2, 0.6])
w_b     = 1 / w
w_b     = w_b / np.linalg.norm(w_b)
epsilon = 2
A       = np.eye(len(w_b))
A[0, 0] = 1 / epsilon**2
e_1     = np.array([1.0, 0.0, 0.0])


# --- Helpers ---

def print_frame():
    print()
    print("-" * 50)
    print()

def print_space():
    print()

# --- Test functions ---

def gram_schmidt_manual(w_b):
    """
    Previous implementation.
    """
    
    # Prepare the change of basis matrix from the standard basis to the
    # one with first vector w_b.
    e = np.eye(len(w_b))
    T = np.zeros((len(w_b), len(w_b)))
    T[:, 0] = w_b

    # Gram-Schmidt orthogonalization.
    for k in range(1, len(w_b)):
        z_k = np.zeros(len(w_b))
        
        for i in range(k):
            z_k += ((e[:, k] @ T[:, i]) / (T[:, i] @ T[:, i])) * T[:, i]
        
        T[:, k] = e[:, k] - z_k
    
    # Normalize the resulting basis.
    for k in range(len(w_b)):
        T[:, k] /= np.linalg.norm(T[:, k])
    
    return T


def gram_schmidt_basis(w_b):
    """
    New optimized implementation.
    """
    
    # Build the initial matrix: w_b + identity as remaining basis.
    n = len(w_b)
    E = np.eye(n)
    M = np.column_stack([w_b, E[:, 1:]])

    # QR decomposition: Q gives orthonormal basis.
    Q, _ = np.linalg.qr(M)
    
    if np.dot(Q[:, 0], w_b) < 0:
        Q[:, 0] *= -1
    
    return Q


# --- Tests ---

# Algorithms run.
T1 = gram_schmidt_manual(w_b)
T2 = gram_schmidt_basis(w_b)

print_frame()

# Print results.
print("Test 1: print orthogonal bases.")
print_space()

print("w:")
print(w)

print_space()

print("w_b:")
print(w_b)

print_space()

print("A:")
print(A)

print_space()

print("Gram-Schmidt original implementation basis:")
print(T1)
print_space()
print("Gram-Schmidt optimized implementation basis:")
print(T2)

print_frame()

# Verify orthogonality (ie. T.T @ T = T^{-1} @ T = I).
print("Test 2: verify orthogonality.")
print_space()

print("Original implementation:", np.allclose(T1.T @ T1, np.eye(T1.shape[0])))
print("Optimized implementation:", np.allclose(T2.T @ T2, np.eye(T2.shape[0])))

print_frame()

# Verify normalization.
print("Test 3: verify normalization.")
print_space()

print("Original implementation:", np.allclose(np.linalg.norm(T1, axis=0), np.ones(T1.shape[1])))
print("Optimized implementation:", np.allclose(np.linalg.norm(T2, axis=0), np.ones(T2.shape[1])))

print_frame()

# Verify whether T.T @ A @ T gives the same result as T @ A @ T.T.
print("Test 4: verify T.T @ A @ T = T @ A @ T.T.")
print_space()

F1 = T1.T @ A @ T1
F2 = T1 @ A @ T1.T
G1 = T2.T @ A @ T2
G2 = T2 @ A @ T2.T

print("Original implementation - T1.T @ A @ T1:")
print(F1)
print("Original implementation - T1 @ A @ T1.T:")
print(F2)
print_space()
print("Optimized implementation - T2.T @ A @ T2:")
print(G1)
print("Optimized implementation - T2 @ A @ T2.T:")
print(G2)
print_space()

print("Equality check for original implementation:")
print(np.allclose(F1, F2))
print("Equality check for optimized implementation:")
print(np.allclose(G1, G2))

print_frame()

# Verify whether T @ e_1 = w_b.
print("Test 5: verify T @ e_1 = w_b (ie. w_b is the first column of T).")
print("If False, the basis has to be transposed and G = T.T @ A @ T is correct.")
print_space()

print("Original implementation:", np.allclose(T1 @ e_1, w_b))
print("Optimized implementation:", np.allclose(T2 @ e_1, w_b))

print_space()