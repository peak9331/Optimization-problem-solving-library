"""
===========================================
可视化工具模块 - 算法收敛曲线绘制
===========================================

【文件总体作用】
本文件负责绘制优化算法的收敛曲线图，包括：
1. 计算当前解向量与最优解的2范数距离
2. 使用对数坐标展示收敛过程
3. 支持多算法对比、不同参数对比等多种可视化需求
4. 解决中文显示和负号显示问题

【核心功能】
- calculate_solution_distances: 计算解向量与最优解的2范数距离
- plot_single_convergence: 绘制单个算法的收敛曲线
- plot_error_convergence: 绘制多算法对比图
- plot_lambda_comparison: 绘制不同正则化系数对比图
- plot_pgd_convergence: 绘制PGD算法专用图
===========================================
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# 解决负号显示问题：设置为False使用ASCII的减号，避免unicode负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 设置数学文本字体，确保指数等数学符号正常显示
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.default'] = 'regular'

"""
===========================================
可视化工具模块 - 算法收敛曲线绘制（续）
===========================================

【模块作用】
为优化算法提供可视化工具，帮助分析和比较算法的收敛性能

【核心功能】
1. calculate_errors: 计算迭代过程中与最优解的距离
2. plot_single_convergence: 绘制单个算法的收敛曲线
3. plot_error_convergence: 绘制多个算法的对比曲线

【为什么使用对数坐标】
优化算法通常呈现"指数收敛"特性：
- 线性坐标：误差下降曲线难以观察细节（前期变化大，后期变化小）
- 对数坐标：将指数曲线变为直线，便于观察收敛速率

例如：
- O(1/k) 收敛：对数图上呈现为斜率 -1 的直线
- O(1/k²) 收敛：对数图上呈现为斜率 -2 的直线

【典型使用流程】
1. 运行算法得到迭代轨迹 x_path = [x₀, x₁, ..., xₖ]
2. 计算 Ground Truth 得到 x*
3. 计算误差序列 errors = [||x₀ - x*||, ||x₁ - x*||, ...]
4. 绘制 log(errors) vs k 的曲线图

【应用场景】
- 算法调试：观察算法是否正常收敛
- 性能比较：对比不同算法的收敛速度
- 参数调优：评估不同参数设置的影响
- 论文/报告：生成专业的算法性能图表
"""


def calculate_solution_distances(x_path, x_optimal):
    """
    计算每次迭代的解向量与最优解的2范数距离

    【数学公式】
    distance_k = ||x* - x_k||_2

    【物理意义】
    - distance_k 表示第 k 次迭代的解向量与最优解向量在欧氏空间中的距离
    - distance_k → 0 表示算法收敛到最优解
    - distance_k 下降速度反映算法的收敛速率

    【参数说明】
    Args:
        x_path (list of ndarray): 每次迭代的解向量列表
                                  x_path[k] 是第 k 次迭代得到的解向量
        x_optimal (ndarray): 最优解向量 x*

    【返回值】
    Returns:
        distances (list of float): 距离列表
                                   distances[k] = ||x* - x_k||_2

    【使用示例】
    >>> x_path = [x0, x1, x2, ...]  # 算法迭代轨迹
    >>> x_optimal = x_star           # 最优解
    >>> distances = calculate_solution_distances(x_path, x_optimal)
    >>> # distances = [||x*-x0||, ||x*-x1||, ||x*-x2||, ...]
    """
    distances = []
    for x_k in x_path:
        # 计算2范数距离: ||x* - x_k||_2
        distance = np.linalg.norm(x_optimal - x_k)
        # 确保距离非负（理论上总是非负，但防止数值误差）
        distance = max(distance, 1e-16)
        distances.append(distance)
    return distances


def plot_single_convergence(x_path, x_optimal, algorithm_name, filename="pgd_convergence.png"):
    """
    绘制单个算法的收敛速度图

    【功能】
    生成一张图表，展示算法的解向量如何逐步逼近最优解

    【图表特点】
    - 横坐标：迭代次数 k (0, 1, 2, ...)
    - 纵坐标：||x* - x_k||_2 （对数尺度）
    - 曲线形状：
      * 陡峭下降：收敛快
      * 平缓下降：收敛慢
      * 水平段：已达到收敛

    【参数说明】
    Args:
        x_path (list of ndarray): 每次迭代的解向量
        x_optimal (ndarray): 最优解向量
        algorithm_name (str): 算法名称（用于图例和标题）
        filename (str): 保存的文件名，默认 "pgd_convergence.png"

    【输出】
    - 显示图表窗口
    - 保存图片到当前目录
    - 打印保存路径
    """
    # 1. 计算解向量距离
    distances = calculate_solution_distances(x_path, x_optimal)

    # 2. 创建图表
    plt.figure(figsize=(10, 6))

    # 使用对数尺度绘制距离（纵坐标为对数尺度）
    plt.semilogy(distances, linewidth=2, color='blue', marker='o',
                markersize=4, label=algorithm_name)

    # 3. 设置标签和标题
    plt.title(f"{algorithm_name} 收敛曲线", fontsize=14, fontweight='bold')
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel(r"$||x^* - x^{(k)}||_2$ (对数尺度)", fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 4. 保存并显示图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[Plot] 单算法收敛图已保存为 {filename}")
    plt.show()
    plt.close()


def plot_error_convergence(results_dict, title, filename):
    """
    绘制多算法对比图

    【功能】
    在同一张图上展示多个算法的收敛速度，便于性能比较

    【典型应用】
    比较 PGD、FISTA、Coordinate Descent 三种算法：
    - 哪个算法收敛最快？
    - 哪个算法最终精度最高？
    - 哪个算法最稳定？

    【参数说明】
    Args:
        results_dict (dict): 字典，格式为 {算法名称: 解向量距离列表}
                            例如：{
                                'PGD': [1.5, 0.8, 0.3, ...],
                                'FISTA': [1.5, 0.5, 0.1, ...],
                                'CD': [1.5, 0.9, 0.4, ...]
                            }
        title (str): 图表标题
        filename (str): 保存的文件名
    """
    plt.figure(figsize=(10, 6))

    # 定义不同的标记样式，便于区分不同算法
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 遍历每个算法的结果
    for idx, (name, distances) in enumerate(results_dict.items()):
        marker = markers[idx % len(markers)]

        # 使用对数尺度绘制
        plt.semilogy(distances, label=name, linewidth=2,
                    marker=marker, markersize=5, markevery=max(1, len(distances)//20))

    # 设置图表属性
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel(r"$||x^* - x^{(k)}||_2$ (对数尺度)", fontsize=12)

    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存并显示
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[Plot] 对比图已保存为 {filename}")
    plt.show()
    plt.close()


def plot_lambda_comparison(lambda_results_dict, algorithm_name, title, filename):
    """
    绘制不同正则化系数的对比图

    【功能】
    在同一张图上展示同一算法在不同λ值下的收敛表现

    【参数说明】
    Args:
        lambda_results_dict (dict): 字典，格式为 {λ值: 解向量距离列表}
        algorithm_name (str): 算法名称（如 "PGD"）
        title (str): 图表标题
        filename (str): 保存的文件名
    """
    plt.figure(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 按λ值排序以便图例更清晰
    sorted_items = sorted(lambda_results_dict.items(), key=lambda x: x[0])

    for idx, (lam, distances) in enumerate(sorted_items):
        marker = markers[idx % len(markers)]
        label = f"λ = {lam}"

        # 使用对数尺度
        plt.semilogy(distances, label=label, linewidth=2,
                    marker=marker, markersize=5, markevery=max(1, len(distances)//20))

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel(r"$||x^* - x^{(k)}||_2$ (对数尺度)", fontsize=12)

    plt.legend(fontsize=11, loc='best', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[Plot] λ对比图已保存为 {filename}")
    plt.show()
    plt.close()


def plot_pgd_convergence(x_path, x_optimal, filename="pgd_convergence.png"):
    """
    专门绘制PGD（近端梯度下降法）的收敛曲线

    【功能】
    为PGD算法提供专用的可视化函数

    【参数说明】
    Args:
        x_path (list of ndarray): PGD算法每次迭代的解向量
        x_optimal (ndarray): 最优解向量
        filename (str): 保存的文件名，默认 "pgd_convergence.png"
    """
    plot_single_convergence(
        x_path,
        x_optimal,
        algorithm_name="近端梯度下降法 (PGD)",
        filename=filename
    )

