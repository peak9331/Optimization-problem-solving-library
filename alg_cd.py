import numpy as np
import time
from utils import soft_thresholding

"""
===========================================
坐标下降法 (Coordinate Descent) - LASSO 求解器
===========================================

【文件作用】
本文件实现了坐标下降法（CD），专门用于求解LASSO问题。
CD每次只更新一个坐标，特别适合高维稀疏问题，无需计算全局Lipschitz常数。

【主要功能】
- solve(): 运行坐标下降算法求解LASSO问题
- 返回最优解、迭代轨迹和运行时间
- 仅适用于LASSO问题（不支持Logistic回归）

【算法原理】
坐标下降法是一种优化算法，每次迭代只更新一个坐标（特征维度），固定其他坐标。
通过循环更新所有坐标，最终收敛到最优解。

【适用场景】
- 高维稀疏问题（特征数很大但非零元素少）
- LASSO 问题（L1 正则化线性回归）
- 不需要计算全局 Lipschitz 常数（只需每个坐标的常数）

【算法步骤】
1. 初始化 x = 0
2. 对于每次迭代 k：
   - 对于每个坐标 j = 1, 2, ..., n：
     * 计算该坐标的梯度
     * 使用软阈值更新 x[j]
   - 检查收敛条件
3. 返回最优解

【与其他算法对比】
- PGD: 每次更新所有坐标，需要全局 Lipschitz 常数
- FISTA: PGD 的加速版本，收敛更快
- CD: 每次只更新一个坐标，适合超高维问题
"""


def solve(A, b, lam, max_iter=1000, tol=1e-12):
    """
    坐标下降法求解 LASSO 问题

    【优化目标】
    minimize F(x) = (1/2)||Ax - b||₂² + λ||x||₁

    【参数说明】
    Args:
        A (ndarray): 特征矩阵，形状 (m, n)
                     m = 样本数，n = 特征数
        b (ndarray): 标签向量，形状 (m,)
        lam (float): L1 正则化参数 λ
                     λ 越大，解越稀疏（更多零元素）
        max_iter (int): 最大迭代次数，默认 1000
        tol (float): 收敛容忍度，默认 1e-6
                     当 ||x_new - x_old|| < tol 时停止

    【返回值】
    Returns:
        x (ndarray): 最优解向量，形状 (n,)
        x_path (list): 迭代轨迹，用于绘制收敛曲线
                       x_path[k] 是第 k 次迭代的解
        elapsed_time (float): 算法运行时间（秒）

    【算法特点】
    - 优点：不需要全局 Lipschitz 常数，适合超高维问题
    - 缺点：收敛速度通常比 FISTA 慢
    - 复杂度：每次迭代 O(mn)，其中 m 是样本数，n 是特征数
    """
    m, n = A.shape
    x = np.zeros(n)  # 初始化解向量为零
    x_path = []  # 记录每次迭代的解，用于后续分析和可视化

    # 预计算每个特征列的 Lipschitz 常数 (L_j = ||A_j||^2)
    # z[j] = sum(A[:, j]^2) = A_j 的平方和
    # 这样可以避免在内层循环中重复计算，大幅提高效率
    z = np.sum(A ** 2, axis=0)

    start_time = time.time()  # 记录开始时间

    # 主迭代循环
    for k in range(max_iter):
        x_path.append(x.copy())  # 保存当前解（必须 copy，否则会被后续修改）
        x_old_iter = x.copy()  # 保存旧解，用于判断收敛

        # 内层循环：逐个更新每个坐标
        for j in range(n):
            # 【坐标下降的核心步骤】
            # 目标：在固定其他坐标的情况下，找到 x[j] 的最优值

            # 步骤1：计算当前的预测值和残差
            prediction = A @ x  # Ax，当前模型的预测值
            residual = b - prediction  # b - Ax，预测误差（残差）

            # 步骤2：计算坐标 j 的梯度方向投影
            # rho 的数学含义：如果忽略 L1 正则项，x[j] 的最优更新方向
            # 推导：对 F(x) 关于 x[j] 求偏导，得到：
            # ∂F/∂x[j] = A_j^T(Ax - b) + λ·sign(x[j])
            # 忽略正则项，令偏导为 0：A_j^T(Ax - b) = 0
            # 移项得：A_j^T·b = A_j^T·A·x = A_j^T·(A_{-j}x_{-j} + A_j·x[j])
            # 进一步得：rho = A_j^T·residual + z[j]·x[j]
            rho = A[:, j].T @ residual + z[j] * x[j]

            # 步骤3：使用软阈值算子更新 x[j]
            # 软阈值处理 L1 正则项，公式：sign(rho) * max(|rho| - λ, 0) / z[j]
            if z[j] > 1e-12:  # 防止除以0（z[j] 可能为 0 如果该特征列全为 0）
                x[j] = soft_thresholding(rho, lam) / z[j]

        # 检查收敛条件：解的变化量小于容忍度
        if np.linalg.norm(x - x_old_iter) < tol:
            x_path.append(x.copy())  # 保存最终解
            break

    return x, x_path, time.time() - start_time