import numpy as np
import time
from utils import soft_thresholding


def solve(A, b, lam, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    x_path = []

    # 预计算每个特征列的 Lipschitz 常数 (L_j = ||A_j||^2)
    z = np.sum(A ** 2, axis=0)

    start_time = time.time()
    for k in range(max_iter):
        x_path.append(x.copy())
        x_old_iter = x.copy()

        for j in range(n):
            # 简化计算 rho 的逻辑
            prediction = A @ x
            residual = b - prediction
            rho = A[:, j].T @ residual + z[j] * x[j]

            # 更新 x[j]
            if z[j] > 1e-12:  # 防止除以0
                x[j] = soft_thresholding(rho, lam) / z[j]

        if np.linalg.norm(x - x_old_iter) < tol:
            x_path.append(x.copy())
            break

    return x, x_path, time.time() - start_time