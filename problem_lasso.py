import numpy as np

def loss(A, b, x):
    """0.5 * ||Ax - b||^2"""
    return 0.5 * np.linalg.norm(A @ x - b)**2

def gradient(A, b, x):
    """A.T * (Ax - b)"""
    return A.T @ (A @ x - b)

def calc_lipschitz(A):
    """L = ||A||_2^2 (最大特征值)"""
    return np.linalg.norm(A, ord=2)**2