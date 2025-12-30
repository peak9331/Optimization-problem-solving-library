import numpy as np

"""
===========================================
Lasso 优化问题的数学函数模块
===========================================

【文件作用】
本文件定义了LASSO问题的数学函数，包括损失函数、梯度函数和Lipschitz常数计算。
这些函数被优化算法（PGD、FISTA、CD）调用，用于求解带L1正则化的线性回归问题。

【主要功能】
- loss(): 计算LASSO问题的光滑部分损失 (1/2)||Ax-b||²
- gradient(): 计算损失函数的梯度 A^T(Ax-b)
- calc_lipschitz(): 计算Lipschitz常数，用于确定最优步长

【问题背景】
Lasso (Least Absolute Shrinkage and Selection Operator) 是一种带 L1 正则化的线性回归方法

【优化问题形式】
minimize F(x) = f(x) + g(x)
其中:
  f(x) = (1/2)||Ax - b||₂²  ← 光滑部分（本文件实现）
  g(x) = λ||x||₁            ← 非光滑部分（优化算法处理）

【问题分解】
- 光滑项 f(x): 最小二乘损失，可求导，由本文件提供梯度计算
- 非光滑项 g(x): L1 正则化，不可导，由优化算法通过近端算子处理

【本文件职责】
提供三个核心函数，支持优化算法（如 ISTA、FISTA、近端梯度下降等）
1. loss(): 计算目标函数值（监控优化进度）
2. gradient(): 计算梯度（确定下降方向）
3. calc_lipschitz(): 计算 Lipschitz 常数（确定最优步长）
"""

def loss(A, b, x):
    """
    【函数名称】损失函数（Loss Function）

    【数学公式】
    f(x) = (1/2)||Ax - b||₂²
         = (1/2) Σᵢ(aᵢᵀx - bᵢ)²

    【物理意义】
    - Ax: 线性模型的预测值向量
    - b: 真实观测值向量
    - Ax - b: 预测误差（残差向量）
    - ||·||₂²: 欧几里得范数的平方，即所有元素平方和
    - 系数 1/2: 技巧性设置，使求导后消除常数 2

    【为什么用平方】
    1. 数学性质好: 可微、凸函数
    2. 惩罚大误差: 平方使大误差贡献更大
    3. 统计意义: 最大似然估计（假设误差服从高斯分布）

    【使用场景】
    - 训练过程: 每次迭代后评估当前解的质量
    - 收敛判断: loss 变化小于阈值时停止
    - 可视化: 绘制 loss 曲线观察收敛过程

    【计算复杂度】
    O(n·d + n) = O(nd)
    - A @ x: O(n·d) 其中 n=样本数, d=特征数
    - 范数计算: O(n)

    Args:
        A: 特征矩阵 (n_samples, n_features)
        b: 标签向量 (n_samples,)
        x: 当前权重向量 (n_features,)

    Returns:
        float: 损失函数值（非负标量）

    【示例】
    >>> A = np.array([[1, 2], [3, 4]])
    >>> b = np.array([1, 2])
    >>> x = np.array([0.5, 0.5])
    >>> loss(A, b, x)
    0.125  # (1/2) * ||(1.5, 3.5) - (1, 2)||² = (1/2) * (0.25 + 2.25)
    """
    return 0.5 * np.linalg.norm(A @ x - b)**2

def gradient(A, b, x):
    """
    【函数名称】梯度函数（Gradient）

    【数学公式】
    ∇f(x) = Aᵀ(Ax - b)

    【详细推导】
    设 r = Ax - b (残差向量)
    f(x) = (1/2)||r||₂² = (1/2)rᵀr

    求导:
    df/dx = (1/2) · d(rᵀr)/dx
          = (1/2) · 2rᵀ · dr/dx      [链式法则]
          = rᵀ · A                   [因为 r = Ax - b, 所以 dr/dx = A]
          = Aᵀr                       [转置]
          = Aᵀ(Ax - b)

    【几何意义】
    - 梯度方向: 函数值增长最快的方向
    - 负梯度方向: 函数值下降最快的方向（最速下降法）
    - 梯度大小: 表示下降速度

    【向量形式解释】
    ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₐ]ᵀ
    每个分量 ∂f/∂xⱼ 表示沿第 j 个特征方向的变化率

    【在优化中的作用】
    1. 梯度下降: xₖ₊₁ = xₖ - α∇f(xₖ)
    2. 近端梯度法: 先梯度下降，再应用近端算子
    3. 加速方法: FISTA 使用梯度和动量

    【计算复杂度】
    O(nd + d²) ≈ O(nd) (当 n >> d 时)
    - A @ x: O(nd)
    - A.T @ (...): O(nd)

    Args:
        A: 特征矩阵 (n_samples, n_features)
        b: 标签向量 (n_samples,)
        x: 当前权重向量 (n_features,)

    Returns:
        ndarray: 梯度向量 (n_features,)，每个元素是对应特征的偏导数

    【示例】
    >>> A = np.array([[1, 0], [0, 1]])
    >>> b = np.array([1, 1])
    >>> x = np.array([2, 3])
    >>> gradient(A, b, x)
    array([1, 2])  # Aᵀ([2,3] - [1,1]) = [1, 2]
    """
    return A.T @ (A @ x - b)

def calc_lipschitz(A):
    """
    【函数名称】Lipschitz 常数计算

    【数学公式】
    L = ||A||₂² = λₘₐₓ(AᵀA)

    【Lipschitz 连续性定义】
    函数 f 的梯度满足 Lipschitz 连续，如果存在常数 L 使得:
    ||∇f(x) - ∇f(y)||₂ ≤ L||x - y||₂  对所有 x, y 成立

    【为什么是 ||A||₂²】
    对于 f(x) = (1/2)||Ax - b||₂²:
    - ∇f(x) = Aᵀ(Ax - b)
    - ∇f(x) - ∇f(y) = AᵀA(x - y)
    - ||∇f(x) - ∇f(y)||₂ = ||AᵀA(x - y)||₂
                          ≤ ||AᵀA||₂ · ||x - y||₂
                          = ||A||₂² · ||x - y||₂
    因此 L = ||A||₂²

    【矩阵范数补充】
    - ||A||₂: 谱范数（2-范数），等于 A 的最大奇异值
    - ||A||₂² = σₘₐₓ²(A) = λₘₐₓ(AᵀA)
    - 表示矩阵对向量的最大"拉伸"能力

    【在优化中的关键作用】
    1. 确定步长上界: α ≤ 1/L 保证收敛
    2. 最优步长: α = 1/L 收敛最快
    3. 收敛速率: O(L/k) 或 O(√(L/k))

    【步长选择的影响】
    - α > 1/L: 可能发散（震荡或爆炸）
    - α = 1/L: 最优收敛速度
    - α < 1/L: 可以收敛但速度慢

    【计算方法】
    使用 numpy.linalg.norm(A, ord=2):
    - 内部使用 SVD 或幂迭代法
    - 复杂度: O(min(nd², n²d)) ≈ O(nd²) 当 n > d
    - 只需计算一次（A 不变）

    Args:
        A: 特征矩阵 (n_samples, n_features)

    Returns:
        float: Lipschitz 常数 L（正标量）

    【示例】
    >>> A = np.array([[1, 0], [0, 2]])  # 对角矩阵
    >>> calc_lipschitz(A)
    4.0  # max(1², 2²) = 4

    >>> A = np.eye(100)  # 单位矩阵
    >>> calc_lipschitz(A)
    1.0  # 单位矩阵的谱范数为 1
    """
    return np.linalg.norm(A, ord=2)**2

"""
===========================================
三个函数的协同工作流程
===========================================

【典型优化算法伪代码】
```
# 初始化
L = calc_lipschitz(A)      # 计算步长 (一次性)
alpha = 1 / L              # 最优步长
x = zeros(n_features)      # 初始解

# 迭代优化
for k in range(max_iter):
    grad = gradient(A, b, x)           # 计算梯度
    x = x - alpha * grad                # 梯度下降
    x = prox_L1(x, lambda * alpha)     # L1 近端算子（软阈值）

    current_loss = loss(A, b, x)       # 监控进度
    if converged: break
```

【数值稳定性注意事项】
1. 矩阵条件数: 如果 A 病态（条件数很大），L 会很大，步长会很小
2. 数据归一化: 建议对 A 进行列归一化，改善条件数
3. 精度问题: 使用 float64 避免数值误差累积
"""

