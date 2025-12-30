import alg_fista
import numpy as np
"""
===========================================
Ground Truth 求解器 - 计算真实最优解
===========================================

【文件作用】
本文件使用高精度FISTA算法计算优化问题的近似最优解（Ground Truth）。
这个最优解用作评估其他算法性能的基准，通过计算目标函数值与最优值的差距来衡量收敛速度。

【主要功能】
- get_optimal_x(): 计算给定问题的近似最优解
- 使用FISTA算法以极高精度（max_iter=10000, tol=1e-12）求解
- 返回最优解x*，用于后续性能评估

【模块作用】
为算法性能评估提供基准（Ground Truth），即真实的最优解 x*

【为什么需要 Ground Truth】
在优化算法研究中，我们需要评估算法的收敛速度和精度：
- 收敛速度：||x_k - x*|| 随迭代次数 k 的下降速度
- 最终精度：算法停止时与最优解的距离

但问题是：我们通常不知道真实的最优解 x* 是多少！

【解决方案】
使用高精度的优化算法（FISTA）跑很多轮，得到近似最优解：
- max_iter = 10000（远超正常迭代次数）
- tol = 1e-12（远低于正常收敛容忍度）
得到的解可以认为"足够接近真实最优解"，作为评估基准

【类比】
就像测量仪器需要"标准件"来校准，算法评估需要"标准解"来对比

【使用场景】
1. 算法对比：比较 PGD、FISTA、CD 哪个收敛更快
2. 参数调优：测试不同步长 α 对收敛速度的影响
3. 理论验证：验证收敛速率是否符合理论分析（如 O(1/k²)）

【注意事项】
- Ground Truth 只是"近似最优解"，不是严格意义的最优解
- 对于高精度要求的问题，可能需要更多迭代次数
- 计算 Ground Truth 本身比较耗时，但只需计算一次
"""


def get_optimal_x(A, b, grad_func, loss_func, lam, alpha):
    """
    计算 LASSO 问题的近似最优解（Ground Truth）

    【方法】
    使用 FISTA 算法以极高精度求解，得到的结果作为"真实最优解"

    【参数说明】
    Args:
        A (ndarray): 特征矩阵，形状 (m, n)
        b (ndarray): 标签向量，形状 (m,)
        grad_func (callable): 梯度函数
        loss_func (callable): 损失函数
        lam (float): L1 正则化参数 λ
        alpha (float): 学习率（步长）

    【返回值】
    Returns:
        x_star (ndarray): 近似最优解，形状 (n,)

    【超参数设置】
    - max_iter = 10000: 远超正常使用的迭代次数（通常 100-1000）
    - tol = 1e-12: 远低于正常容忍度（通常 1e-6 ~ 1e-8）

    【为什么选择 FISTA】
    - 收敛速度快：O(1/k²) 收敛率，比 PGD 的 O(1/k) 快
    - 稳定性好：不会震荡，单调收敛
    - 精度高：能在有限迭代内达到极高精度

    【计算成本】
    Ground Truth 计算通常需要几秒到几分钟，但只需计算一次，
    之后可以用来评估多个算法的性能

    【使用示例】
    >>> x_star = get_optimal_x(A, b, gradient, loss, lam=0.1, alpha=1/L)
    >>> # 计算某算法的误差
    >>> error_k = np.linalg.norm(x_k - x_star)
    """
    print(f"[Ground Truth] 计算最优解 (Lambda={lam})...")
    print(f"[Ground Truth] 使用 FISTA 高精度求解：max_iter=20000, tol=1e-15")

    # 使用 FISTA 算法求解
    # max_iter=20000: 允许足够多的迭代次数
    # tol=1e-15: 设置极严格的收敛条件
    x_star, _, _ = alg_fista.solve(
        A, b, grad_func, loss_func,
        lam, alpha,
        max_iter=20000,
        tol=1e-15
    )

    print(f"[Ground Truth] 最优解计算完成！")
    print(f"[Ground Truth] 非零元素数量: {np.sum(np.abs(x_star) > 1e-10)} / {len(x_star)}")
    print(f"[Ground Truth] 解的范数: ||x*||₂ = {np.linalg.norm(x_star):.6f}")

    return x_star