import numpy as np
import time
from utils import soft_thresholding


def solve(A, b, grad_func, loss_func, lam, alpha, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    x_path = []  # 记录轨迹

    start_time = time.time()
    for k in range(max_iter):
        x_path.append(x.copy())  # 必须 copy，否则存的是引用

        grad = grad_func(A, b, x)
        u = x - alpha * grad
        x_new = soft_thresholding(u, alpha * lam)

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            x_path.append(x.copy())
            break
        x = x_new

    return x, x_path, time.time() - start_time