import numpy as np
import time

"""
===========================================
ADMM算法 (Alternating Direction Method of Multipliers)
===========================================

【文件作用】
本文件实现了交替方向乘子法（ADMM），这是一种高效求解带约束优化问题的算法。
ADMM特别适合求解LASSO问题，收敛速度快，迭代次数少，精度高。

【主要功能】
- solve(): 使用ADMM算法求解LASSO问题
- 返回最优解、迭代轨迹和运行时间
- 相比PGD/FISTA，在相同精度下需要的迭代次数更少

【算法原理】
ADMM是求解带约束优化问题的强大方法，通过变量分裂和对偶上升实现高效求解。

【优化问题形式】
原问题: minimize (1/2)||Ax - b||₂² + λ||x||₁

ADMM重构为:
minimize (1/2)||Ax - b||₂² + λ||z||₁
subject to x = z

【增广拉格朗日函数】
L_ρ(x, z, u) = (1/2)||Ax - b||₂² + λ||z||₁ + (ρ/2)||x - z + u||₂²

【算法迭代公式】
1. x-update: x^{k+1} = argmin_x L_ρ(x, z^k, u^k)
             = (A^T A + ρI)^{-1}(A^T b + ρ(z^k - u^k))
2. z-update: z^{k+1} = argmin_z L_ρ(x^{k+1}, z, u^k)
             = soft_threshold(x^{k+1} + u^k, λ/ρ)
3. u-update: u^{k+1} = u^k + x^{k+1} - z^{k+1}

【ADMM的优势】
1. 收敛速度快：通常在几百次迭代内达到高精度
2. 数值稳定：不需要精确的步长选择
3. 可并行化：适合大规模问题
4. 理论保证：凸问题保证收敛

【与其他算法对比】
- PGD: O(1/k) 收敛，需要调步长
- FISTA: O(1/k²) 收敛，但常数较大
- ADMM: 线性收敛，实际表现优异

【参数选择】
- ρ: 惩罚参数，影响收敛速度
  * 太小：收敛慢
  * 太大：数值不稳定
  * 推荐：ρ ≈ 1.0 或自适应调整
"""


def soft_thresholding(x, threshold):
    """
    软阈值算子（Soft-thresholding operator）

    【数学定义】
    soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)

    【分段函数形式】
           ⎧ x - λ,  if x > λ
    S(x) = ⎨ 0,      if |x| ≤ λ
           ⎩ x + λ,  if x < -λ

    【物理意义】
    对向量的每个元素施加"软收缩"：
    - 大于阈值的部分：向零收缩 λ
    - 小于阈值的部分：直接变为零（产生稀疏性）

    参数:
        x (ndarray): 输入向量
        threshold (float): 阈值参数 λ

    返回:
        ndarray: 软阈值后的向量
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def solve(A, b, lam, rho=1.0, max_iter=1000, tol=1e-12):
    """
    ADMM算法求解LASSO问题

    【优化目标】
    minimize F(x) = (1/2)||Ax - b||₂² + λ||x||₁

    【ADMM重构】
    minimize (1/2)||Ax - b||₂² + λ||z||₁
    subject to x = z

    【算法步骤】
    1. 初始化: x⁰ = 0, z⁰ = 0, u⁰ = 0
    2. 迭代更新:
       - x^{k+1} = (A^T A + ρI)^{-1}(A^T b + ρ(z^k - u^k))
       - z^{k+1} = soft_threshold(x^{k+1} + u^k, λ/ρ)
       - u^{k+1} = u^k + x^{k+1} - z^{k+1}
    3. 收敛判断: ||x^{k+1} - z^{k+1}|| < tol

    【参数说明】
    Args:
        A (ndarray): 特征矩阵，形状 (m, n)
        b (ndarray): 标签向量，形状 (m,)
        lam (float): L1正则化系数 λ > 0
        rho (float): ADMM惩罚参数 ρ > 0，默认1.0
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差

    【返回值】
    Returns:
        x (ndarray): 最优解，形状 (n,)
        x_path (list): 迭代轨迹，存储每次迭代的x值
        elapsed_time (float): 算法运行时间（秒）

    【收敛条件】
    Primal residual: r = ||x - z|| < tol
    Dual residual: s = ||ρ * (z^{k+1} - z^k)|| < tol

    【计算复杂度】
    每次迭代: O(n³) 用于求解线性方程组（预计算逆矩阵可优化）
    总复杂度: O(K * n³)，但K通常远小于PGD/FISTA的迭代次数
    """
    m, n = A.shape

    # 预计算 A^T A + ρI 的逆矩阵（Cholesky分解加速）
    # 这是ADMM的关键优化：只需计算一次
    ATA = A.T @ A
    ATA_rho_I = ATA + rho * np.eye(n)
    ATb = A.T @ b

    # Cholesky分解: ATA_rho_I = L * L^T
    # 使用Cholesky分解比直接求逆更快更稳定
    L = None
    ATA_rho_I_inv = None
    use_cholesky = False

    try:
        L = np.linalg.cholesky(ATA_rho_I)
        use_cholesky = True
    except np.linalg.LinAlgError:
        # 如果Cholesky失败，使用直接求逆
        ATA_rho_I_inv = np.linalg.inv(ATA_rho_I)
        use_cholesky = False

    # 初始化变量
    x = np.zeros(n)  # 原始变量
    z = np.zeros(n)  # 辅助变量
    u = np.zeros(n)  # 对偶变量（scaled dual variable）

    # 存储迭代轨迹
    x_path = [x.copy()]

    # 开始计时
    start_time = time.time()

    # ADMM主循环
    for k in range(max_iter):
        # 保存旧的z值（用于计算对偶残差）
        z_old = z.copy()

        # ===== Step 1: x-update =====
        # x^{k+1} = (A^T A + ρI)^{-1}(A^T b + ρ(z^k - u^k))
        rhs = ATb + rho * (z - u)

        if use_cholesky:
            # 使用Cholesky分解求解: L * L^T * x = rhs
            # 先解 L * y = rhs
            y = np.linalg.solve(L, rhs)
            # 再解 L^T * x = y
            x = np.linalg.solve(L.T, y)
        else:
            x = ATA_rho_I_inv @ rhs

        # ===== Step 2: z-update =====
        # z^{k+1} = soft_threshold(x^{k+1} + u^k, λ/ρ)
        z = soft_thresholding(x + u, lam / rho)

        # ===== Step 3: u-update =====
        # u^{k+1} = u^k + x^{k+1} - z^{k+1}
        u = u + x - z

        # 存储当前x
        x_path.append(x.copy())

        # ===== 收敛性检查 =====
        # Primal residual: r = x - z
        primal_residual = np.linalg.norm(x - z)

        # Dual residual: s = ρ * (z - z_old)
        dual_residual = rho * np.linalg.norm(z - z_old)

        # 如果原始残差和对偶残差都很小，则收敛
        if primal_residual < tol and dual_residual < tol:
            break

    # 结束计时
    elapsed_time = time.time() - start_time

    return x, x_path, elapsed_time

