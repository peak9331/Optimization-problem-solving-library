import matplotlib.pyplot as plt
import numpy as np


def calculate_errors(x_path, x_star):
    """
    计算轨迹中每一步与最优解的距离: ||x_k - x*||_2
    """
    errors = []
    for x_k in x_path:
        # 计算 L2 范数距离
        diff = np.linalg.norm(x_k - x_star)
        errors.append(diff)
    return errors


def plot_single_convergence(x_path, x_star, algorithm_name, filename="pgd_convergence.png"):
    """
    【新增功能】绘制单个算法的收敛速度图
    横坐标：迭代次数
    纵坐标：最优值减去当前值的模长 (对数坐标)
    """
    # 1. 计算误差
    errors = calculate_errors(x_path, x_star)

    # 2. 绘图
    plt.figure(figsize=(8, 6))

    # 使用 semilogy 实现纵坐标取对数的效果
    # 为了防止 log(0) 的情况，加一个极小值 eps
    safe_errors = np.array(errors) + 1e-16

    plt.semilogy(safe_errors, linewidth=2, color='blue', label=algorithm_name)

    # 3. 设置标签和标题 (支持 LaTeX 公式显示)
    plt.title(f"{algorithm_name} Convergence Speed")
    plt.xlabel("Iteration Number (k)")
    # 纵坐标标签：Log of Error Norm
    plt.ylabel(r"Log Error: $\log(\|x^* - x^{(k)}\|_2)$")

    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 4. 保存
    plt.savefig(filename)
    plt.show()
    print(f"[Plot] 单算法收敛图已保存为 {filename}")


def plot_error_convergence(results_dict, title, filename):
    """
    (原有功能) 绘制多算法对比图
    """
    plt.figure(figsize=(10, 6))

    for name, x_path in results_dict.items():
        # 注意：这里传入的若是 x_path 列表，我们需要 x_star 才能计算。
        # 为了兼容性，假设这里传入的已经是计算好的 errors 列表 (如 main.py 中逻辑)
        # 或者我们统一在 main.py 里算好 errors 再传进来。
        # 这里假设传入的是 errors 列表
        safe_errors = np.array(x_path) + 1e-16
        plt.semilogy(safe_errors, label=name, linewidth=2)

    plt.title(title)
    plt.xlabel("Iterations (k)")
    plt.ylabel(r"$\log(\|x^* - x^{(k)}\|_2)$")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"[Plot] 对比图已保存为 {filename}")