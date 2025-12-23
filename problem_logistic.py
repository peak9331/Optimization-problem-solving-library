import numpy as np

def loss(A, b, x):
    """Logistic Loss"""
    m = A.shape[0]
    z = -b * (A @ x)
    return np.sum(np.log(1 + np.exp(z))) / m

def gradient(A, b, x):
    """Logistic Gradient"""
    m = A.shape[0]
    z = b * (A @ x)
    # q = 1 / (1 + exp(z))
    q = 1 / (1 + np.exp(z))
    grad = -(1/m) * (A.T @ (b * q))
    return grad

def calc_lipschitz(A):
    """L = ||A||^2 / 4m"""
    m = A.shape[0]
    return (np.linalg.norm(A, ord=2)**2) / (4 * m)