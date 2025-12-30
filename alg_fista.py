import numpy as np
import time
from utils import soft_thresholding

"""
===========================================
FISTA 算法 (Fast Iterative Shrinkage-Thresholding Algorithm)
===========================================

【文件作用】
本文件实现了FISTA算法，是PGD的加速版本，具有更快的收敛速度O(1/k²)。
通过Nesterov动量机制，FISTA在保持PGD简单性的同时，显著提升了收敛速度。

【主要功能】
- solve(): 运行FISTA算法求解优化问题
- 返回最优解、迭代轨迹和运行时间
- 支持自定义梯度函数、损失函数和正则化参数

【算法全称】
快速迭代收缩阈值算法

【算法作者】
Beck 和 Teboulle (2009)

【核心思想】
FISTA 是近端梯度下降法（PGD）的加速版本，通过引入动量项（momentum）实现加速。
关键创新：在"外推点"y 处计算梯度，而不是当前点 x

【加速原理】
- PGD: 收敛速率 O(1/k)
- FISTA: 收敛速率 O(1/k²) ← 平方级加速！

【Nesterov 动量机制】
y_k = x_k + β_k(x_k - x_{k-1})  ← 外推到"未来"位置
x_{k+1} = prox(y_k - α∇f(y_k))  ← 在外推点计算梯度

【适用场景】
- LASSO 问题（L1 正则化）
- 总变分去噪（TV denoising）
- 压缩感知（Compressed Sensing）
- 任何 f(x) + g(x) 形式的凸优化问题

【与 PGD 的对比】
┌────────────┬──────────────┬──────────────┐
│   特性     │     PGD      │    FISTA     │
├────────────┼──────────────┼──────────────┤
│ 收敛速率   │   O(1/k)     │   O(1/k²)    │
│ 计算复杂度 │   低         │   略高       │
│ 实现难度   │   简单       │   中等       │
│ 推荐使用   │   快速原型   │   生产环境   │
└────────────┴──────────────┴──────────────┘
"""


def solve(A, b, grad_func, loss_func, lam, alpha, max_iter=1000, tol=1e-12):
    """
    FISTA 算法求解 LASSO 问题（PGD 的加速版本）

    【优化目标】
    minimize F(x) = f(x) + g(x)
    其中:
      f(x) = (1/2)||Ax - b||₂²  ← 光滑项（可微）
      g(x) = λ||x||₁            ← 非光滑项（L1 正则）

    【算法流程】
    1. 初始化 x₀ = 0, y₀ = 0, t₀ = 1
    2. 对于 k = 0, 1, 2, ...:
       - 计算梯度：∇f(y_k)
       - 近端步：x_{k+1} = prox_{αλ}(y_k - α∇f(y_k))
       - 更新动量系数：t_{k+1} = (1 + √(1 + 4t_k²)) / 2
       - 外推：y_{k+1} = x_{k+1} + ((t_k - 1) / t_{k+1})(x_{k+1} - x_k)
    3. 返回 x_k

    【参数说明】
    Args:
        A (ndarray): 特征矩阵，形状 (m, n)
        b (ndarray): 标签向量，形状 (m,)
        grad_func (callable): 梯度函数，计算 ∇f(x)
                              签名：grad_func(A, b, x) -> ndarray
        loss_func (callable): 损失函数，计算 f(x)
                              签名：loss_func(A, b, x) -> float
        lam (float): L1 正则化参数 λ
        alpha (float): 学习率（步长），通常设为 1/L
                       其中 L 是 f(x) 的 Lipschitz 常数
        max_iter (int): 最大迭代次数，默认 1000
        tol (float): 收敛容忍度，默认 1e-6

    【返回值】
    Returns:
        x (ndarray): 最优解向量，形状 (n,)
        x_path (list): 迭代轨迹，x_path[k] 是第 k 次迭代的解
        elapsed_time (float): 算法运行时间（秒）

    【收敛性保证】
    若 f(x) 是凸且可微的，梯度 Lipschitz 连续（常数 L），
    且步长 α ≤ 1/L，则：
      F(x_k) - F(x*) ≤ 2L||x₀ - x*||² / (k+1)²

    【实际应用提示】
    - 步长选择：α = 1/L 是最优步长（L 可用 calc_lipschitz 计算）
    - 稀疏性：增大 λ 会得到更稀疏的解（更多零元素）
    - 调试：若不收敛，尝试减小 α 或增大 max_iter
    """
    m, n = A.shape
    x = np.zeros(n)  # 当前迭代点
    y = np.zeros(n)  # 动量点（外推点）
    t = 1.0  # 动量系数
    x_path = []  # 记录迭代轨迹

    start_time = time.time()  # 记录开始时间

    # 主迭代循环
    for k in range(max_iter):
        x_path.append(x.copy())  # 保存当前解
        x_old = x.copy()  # 保存旧解，用于动量计算和收敛判断

        # 【FISTA 的核心步骤】

        # 步骤1：在动量点 y 处计算梯度（而非当前点 x）
        # 这是 FISTA 的关键创新：在"外推"位置计算梯度
        grad = grad_func(A, b, y)

        # 步骤2：梯度下降步
        # u = y_k - α∇f(y_k)
        u = y - alpha * grad

        # 步骤3：应用软阈值算子（近端算子）
        # x_{k+1} = prox_{αλ||·||₁}(u) = soft_threshold(u, αλ)
        x = soft_thresholding(u, alpha * lam)

        # 步骤4：更新动量系数（Nesterov 加速方案）
        # t_{k+1} = (1 + √(1 + 4t_k²)) / 2
        # 这个公式来自 Beck & Teboulle (2009) 的理论分析
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2

        # 步骤5：更新动量点 y（外推到"未来"位置）
        # y_{k+1} = x_{k+1} + β_k(x_{k+1} - x_k)
        # 其中 β_k = (t_k - 1) / t_{k+1}
        # 物理意义：在当前梯度方向上"超前"一步
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new

        # 检查收敛条件：解的变化量小于容忍度
        if np.linalg.norm(x - x_old) < tol:
            x_path.append(x.copy())  # 保存最终解
            break

    return x, x_path, time.time() - start_time