import numpy as np
import time
from utils import soft_thresholding


def solve(A, b, grad_func, loss_func, lam, alpha, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(n)
    t = 1.0
    x_path = []

    start_time = time.time()
    for k in range(max_iter):
        x_path.append(x.copy())
        x_old = x.copy()

        # y 是动量点
        grad = grad_func(A, b, y)
        u = y - alpha * grad
        x = soft_thresholding(u, alpha * lam)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new

        if np.linalg.norm(x - x_old) < tol:
            x_path.append(x.copy())
            break

    return x, x_path, time.time() - start_time