import numpy as np
import data_loader
import problem_lasso
import problem_logistic
import alg_pgd
import alg_fista
import alg_cd
import solver_ground_truth
import plotter


def run_experiment():
    # 1. 加载数据
    A, b = data_loader.load_data("a1a")

    # 2. 参数设置
    L = problem_lasso.calc_lipschitz(A)
    alpha = 1.0 / L
    max_iter = 1000
    base_lambda = 0.1

    print(f"\n=== 计算 Ground Truth (Lambda = {base_lambda}) ===")
    x_star = solver_ground_truth.get_optimal_x(
        A, b, problem_lasso.gradient, problem_lasso.loss, base_lambda, alpha
    )

    # ==========================================
    # 【新增】 单独绘制 PGD 近端梯度算法收敛图
    # ==========================================
    print("\n=== 单独测试 PGD 并绘图 ===")
    _, path_pgd, _ = alg_pgd.solve(
        A, b, problem_lasso.gradient, problem_lasso.loss,
        base_lambda, alpha, max_iter
    )

    # 调用刚刚在 plotter.py 里写的新函数
    plotter.plot_single_convergence(
        path_pgd,
        x_star,
        algorithm_name="Proximal Gradient Descent (PGD)",
        filename="pgd_convergence_only.png"
    )

    # ==========================================
    # 下面是原有的对比实验代码 (保持不变即可)
    # ==========================================
    print("\n=== 多算法对比 ===")
    # 运行 FISTA
    _, path_fista, _ = alg_fista.solve(A, b, problem_lasso.gradient, problem_lasso.loss, base_lambda, alpha, max_iter)
    # 运行 CD
    _, path_cd, _ = alg_cd.solve(A, b, base_lambda, max_iter)

    # 准备对比数据 (计算 Errors)
    algo_errors = {
        'PGD': plotter.calculate_errors(path_pgd, x_star),
        'FISTA': plotter.calculate_errors(path_fista, x_star),
        'Coordinate Descent': plotter.calculate_errors(path_cd, x_star)
    }

    plotter.plot_error_convergence(
        algo_errors,
        title="Algorithm Comparison",
        filename="compare_algorithms.png"
    )

    # ... (后续的任务2代码保持不变)


if __name__ == "__main__":
    run_experiment()