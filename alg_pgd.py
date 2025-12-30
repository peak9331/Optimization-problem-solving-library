import numpy as np
import time
from utils import soft_thresholding

"""
===========================================
近端梯度下降法 (Proximal Gradient Descent, PGD)
===========================================

【文件作用】
本文件实现了近端梯度下降法（PGD），用于求解带有L1正则化的优化问题。
PGD是处理"光滑项+非光滑项"复合优化问题的经典算法，特别适合LASSO和Logistic回归问题。

【主要功能】
- solve(): 运行PGD算法求解优化问题
- 返回最优解、迭代轨迹和运行时间
- 支持自定义梯度函数、损失函数和正则化参数

【算法原理】
近端梯度下降法是求解"光滑项 + 非光滑项"组合优化问题的经典方法。
核心思想：将问题分解为两部分分别处理
- 光滑部分（可微）：使用梯度下降
- 非光滑部分（不可微）：使用近端算子

【优化问题形式】
minimize F(x) = f(x) + g(x)
其中:
  f(x): 光滑凸函数（可微），如 (1/2)||Ax - b||₂²
  g(x): 非光滑凸函数（可能不可微），如 λ||x||₁

【算法迭代公式】
x_{k+1} = prox_{αg}(x_k - α∇f(x_k))

分解为两步：
1. 梯度下降步：u = x_k - α∇f(x_k)
2. 近端算子步：x_{k+1} = argmin_x { (1/(2α))||x - u||² + g(x) }

【LASSO 问题的具体实现】
f(x) = (1/2)||Ax - b||₂²  → ∇f(x) = A^T(Ax - b)
g(x) = λ||x||₁            → prox = soft_thresholding

【收敛速率】
O(1/k) - 次线性收敛（比 FISTA 的 O(1/k²) 慢）

【适用场景】
- LASSO 回归（L1 正则化）
- 图像去噪
- 压缩感知
- 稀疏信号恢复

【与其他算法的关系】
- 梯度下降：PGD 当 g(x) = 0 时退化为梯度下降
- FISTA：FISTA 是 PGD 的加速版本
- 次梯度法：PGD 比次梯度法收敛更快更稳定
"""


def solve(A, b, grad_func, loss_func, lam, alpha, max_iter=1000, tol=1e-12):
    """
    近端梯度下降法（Proximal Gradient Descent）求解 LASSO 问题

    【优化目标】
    minimize F(x) = f(x) + g(x)
    其中:
      f(x) = (1/2)||Ax - b||₂²  ← 光滑项（用梯度下降处理）
      g(x) = λ||x||₁            ← 非光滑项（用近端算子处理）

    【算法步骤】
    1. 初始化 x₀ = 0
    2. 对于 k = 0, 1, 2, ...:
       - 计算梯度：grad = ∇f(x_k)
       - 梯度下降：u = x_k - α·grad
       - 近端算子：x_{k+1} = soft_threshold(u, αλ)
       - 检查收敛：||x_{k+1} - x_k|| < tol
    3. 返回 x_k

    【参数说明】
    Args:
        A (ndarray): 特征矩阵，形状 (m, n)
                     m = 样本数，n = 特征数
        b (ndarray): 标签向量，形状 (m,)
        grad_func (callable): 梯度函数，计算 ∇f(x)
                              签名：grad_func(A, b, x) -> ndarray
        loss_func (callable): 损失函数，计算 f(x)
                              签名：loss_func(A, b, x) -> float
                              （本函数未使用，但保留接口一致性）
        lam (float): L1 正则化参数 λ
                     λ 越大，解越稀疏
        alpha (float): 学习率（步长），通常设为 1/L
                       其中 L 是 f(x) 的 Lipschitz 常数
                       L = ||A||₂²（A 的谱范数的平方）
        max_iter (int): 最大迭代次数，默认 1000
        tol (float): 收敛容忍度，默认 1e-6

    【返回值】
    Returns:
        x (ndarray): 最优解向量，形状 (n,)
        x_path (list): 迭代轨迹，x_path[k] 是第 k 次迭代的解
                       用于绘制收敛曲线和分析算法性能
        elapsed_time (float): 算法运行时间（秒）

    【收敛性保证】
    若 f(x) 是凸且可微的，梯度 Lipschitz 连续（常数 L），
    且步长满足 α ≤ 1/L，则算法保证收敛到全局最优解。
    收敛速率：F(x_k) - F(x*) = O(1/k)

    【步长选择建议】
    - 最优步长：α = 1/L（收敛最快）
    - 保守步长：α = 0.9/L（更稳定）
    - 过大步长：α > 1/L（可能不收敛或震荡）
    - 过小步长：α << 1/L（收敛太慢）

    【使用示例】
    >>> from problem_lasso import gradient, loss, calc_lipschitz
    >>> L = calc_lipschitz(A)
    >>> alpha = 1.0 / L
    >>> x, path, time = solve(A, b, gradient, loss, lam=0.1, alpha=alpha)
    """
    m, n = A.shape
    x = np.zeros(n)  # 初始化解向量为零
    x_path = []  # 记录迭代轨迹，用于后续分析和可视化

    start_time = time.time()  # 记录开始时间

    # 主迭代循环
    for k in range(max_iter):
        x_path.append(x.copy())  # 保存当前解（必须 copy，否则存的是引用）

        # 【PGD 的两步迭代】

        # 步骤1：计算光滑部分的梯度
        # grad = ∇f(x) = A^T(Ax - b)
        grad = grad_func(A, b, x)

        # 步骤2：梯度下降步
        # u = x_k - α∇f(x_k)
        # 这是标准梯度下降的更新，但还未处理 L1 正则项
        u = x - alpha * grad

        # 步骤3：应用近端算子处理 L1 正则项
        # x_{k+1} = prox_{αλ||·||₁}(u)
        # 对于 L1 范数，近端算子就是软阈值算子
        # soft_threshold(u, αλ) = sign(u) * max(|u| - αλ, 0)
        x_new = soft_thresholding(u, alpha * lam)

        # 检查收敛条件：解的变化量小于容忍度
        # ||x_{k+1} - x_k||₂ < tol
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            x_path.append(x.copy())  # 保存最终解
            break

        x = x_new  # 更新解

    return x, x_path, time.time() - start_time