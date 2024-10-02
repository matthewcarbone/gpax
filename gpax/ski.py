import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist


def compute_cubic_interpolation_weights(X, U, kernel):
    """
    Thanks, Chat GPT!

    Compute the weight matrix W such that K_XU ≈ W K_UU.

    Parameters:
    - X: np.ndarray, shape (N, d), data points.
    - U: np.ndarray, shape (M, d), inducing points.
    - kernel: callable, the kernel function k(x_i, x_j).

    Returns:
    - W: np.ndarray, shape (N, M), sparse weight matrix with 4 non-zero entries per row.
    """

    N = X.shape[0]  # Number of data points
    M = U.shape[0]  # Number of inducing points

    # Compute the kernel matrix K_UU (M x M)
    K_UU = kernel(U, U)

    # Initialize the weight matrix W (N x M) with all zeros
    W = np.zeros((N, M))

    # Compute the pairwise distances between X and U
    distances = cdist(X, U, metric="euclidean")

    for i in range(N):
        # Find the indices of the 4 closest inducing points for each data point in X
        closest_inds = np.argsort(distances[i])[:4]

        # Extract the corresponding rows of K_UU (local 4x4 kernel submatrix)
        K_UU_local = K_UU[np.ix_(closest_inds, closest_inds)]

        # Compute the kernel values between the data point X[i] and the 4 inducing points
        K_XU_local = kernel(X[i : i + 1], U[closest_inds]).reshape(-1)

        # Solve the linear system to find the weights W_i
        W_i = np.linalg.solve(K_UU_local, K_XU_local)

        # Place the weights into the correct positions in the row of W
        W[i, closest_inds] = W_i

    return W


def compute_cubic_interpolation_sparse_weights(X, U, kernel):
    """
    Compute the sparse weight matrix W such that K_XU ≈ W K_UU.

    Parameters:
    - X: np.ndarray, shape (N, d), data points.
    - U: np.ndarray, shape (M, d), inducing points.
    - kernel: callable, the kernel function k(x_i, x_j).

    Returns:
    - W: scipy.sparse.csr_matrix, shape (N, M), sparse weight matrix with 4 non-zero entries per row.
    """
    N = X.shape[0]  # Number of data points
    M = U.shape[0]  # Number of inducing points

    # Compute the kernel matrix K_UU (M x M)
    K_UU = kernel(U, U)

    # Initialize the weight matrix W as a list-of-lists (LIL) sparse matrix
    W_sparse = lil_matrix((N, M))

    # Compute the pairwise distances between X and U
    distances = cdist(X, U, metric="euclidean")

    for i in range(N):
        # Find the indices of the 4 closest inducing points for each data point in X
        closest_inds = np.argsort(distances[i])[:4]

        # Extract the corresponding rows of K_UU (local 4x4 kernel submatrix)
        K_UU_local = K_UU[np.ix_(closest_inds, closest_inds)]

        # Compute the kernel values between the data point X[i] and the 4 inducing points
        K_XU_local = kernel(X[i : i + 1], U[closest_inds]).reshape(-1)

        # Solve the linear system to find the weights W_i
        W_i = np.linalg.solve(K_UU_local, K_XU_local)

        # Place the weights into the sparse matrix
        W_sparse[i, closest_inds] = W_i

    # Convert the matrix to Compressed Sparse Row (CSR) format for efficiency
    return W_sparse.tocsr()


def reorthogonalize(v, Q, j):
    """
    Reorthogonalizes the vector v against the first j columns of Q.

    Args:
        v (ndarray): Vector to reorthogonalize.
        Q (ndarray): Orthonormal matrix whose columns are the Lanczos vectors.
        j (int): Current iteration step (number of columns in Q to reorthogonalize against).

    Returns:
        v (ndarray): Reorthogonalized vector.
    """
    for i in range(j):
        v = v - np.dot(Q[:, i], v) * Q[:, i]
    return v


def lanczos(A, k, v0=None):
    """
    Perform k iterations of the Lanczos algorithm with full reorthogonalization
    to factorize matrix A into QTQ^T.

    Args:
        A (ndarray): Input matrix (n x n) to decompose.
        k (int): Number of iterations (k <= n).
        v0 (ndarray): Initial vector (optional).

    Returns:
        Q (ndarray): Orthonormal matrix (n x k).
        T (ndarray): Tridiagonal matrix (k x k).
        r (ndarray): Residual vector after k iterations (n,).
    """
    n = A.shape[0]

    if v0 is None:
        v0 = np.random.rand(n)

    # Normalize the initial vector
    v0 = v0 / np.linalg.norm(v0)

    Q = np.zeros((n, k))
    T = np.zeros((k, k))

    beta = 0
    v_prev = np.zeros_like(v0)
    v = v0

    for j in range(k):
        if beta < 1e-12:
            # Generate a new random vector orthogonal to the existing Q columns if beta is too small
            v = np.random.rand(n)

        # Reorthogonalize v against the previous Lanczos vectors
        v = reorthogonalize(v, Q, j)

        # Normalize the vector
        v = v / np.linalg.norm(v)

        Q[:, j] = v

        # Matrix-vector product
        w = A @ v

        # Compute alpha
        alpha = np.dot(v, w)
        T[j, j] = alpha

        # Subtract the projection onto the previous vector
        w = w - alpha * v - beta * v_prev

        # Compute new beta
        beta = np.linalg.norm(w)

        if j < k - 1:
            T[j, j + 1] = beta
            T[j + 1, j] = beta

        # Prepare for the next iteration
        v_prev = v
        if beta >= 1e-12:
            v = w / beta

    # Compute the residual r (the portion of A not captured by the k iterations)
    r = A @ Q[:, -1] - T[-1, -1] * Q[:, -1]

    return Q, T, np.linalg.norm(r)
