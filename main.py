"""
===========================================
主实验脚本 - LASSO 和逻辑回归算法性能对比与可视化
===========================================

【文件总体作用】
本文件是整个优化算法对比实验的主控程序，负责：
1. 协调各个模块（数据加载、问题定义、算法求解、结果可视化）
2. 运行完整的实验流程，比较不同算法的性能
3. 生成所有的收敛曲线图表（共6张图）
4. 输出实验结果和性能统计

【实验内容】
- 两个优化问题：LASSO回归、Logistic回归
- 三种算法：PGD（近端梯度下降）、FISTA（快速迭代收缩阈值）、CD（坐标下降，仅LASSO）
- 每个问题生成3张图：
  1. 不同算法对比图
  2. 不同正则化系数对比图（PGD算法）
  3. 近端梯度下降法单独图

【输出图表】（共6张）
LASSO问题：
  - lasso_1_compare_algorithms.png: 不同算法LASSO问题对比图
  - lasso_2_lambda_comparison.png: 不同正则化系数LASSO问题图
  - lasso_3_pgd_convergence.png: 近端梯度下降法LASSO问题图

Logistic问题：
  - logistic_1_compare_algorithms.png: 不同算法逻辑问题对比图
  - logistic_2_lambda_comparison.png: 不同正则化系数逻辑问题图
  - logistic_3_pgd_convergence.png: 近端梯度下降法逻辑问题图

【关键特性】
- 纵坐标为解向量距离 ||x* - x_k||_2（最优解与当前解的2范数距离）
- 纵坐标尺度为对数尺度（semilogy）
- 最优值通过solver_ground_truth.py高精度求解获得
- 支持中文图表显示和负号正常显示
===========================================
"""

import data_loader
import problem_lasso
import problem_logistic
import alg_pgd
import alg_fista
import alg_cd
import alg_admm
import solver_ground_truth
import plotter

"""
===========================================
主实验脚本（续）
===========================================

【实验目的】
比较三种优化算法在两种不同问题上的性能：
问题类型：
1. LASSO 问题（线性回归 + L1 正则化）
2. Logistic Regression 问题（分类问题 + 对数损失）

优化算法：
1. PGD (Proximal Gradient Descent) - 近端梯度下降
2. FISTA (Fast ISTA) - 快速迭代收缩阈值算法
3. CD (Coordinate Descent) - 坐标下降法

【实验流程】
对于每个问题（LASSO 和 Logistic Regression）：
1. 数据加载：从 LIBSVM 数据集加载特征矩阵 A 和标签 b
2. 参数设置：计算 Lipschitz 常数，确定最优步长
3. Ground Truth：使用高精度算法计算真实最优解
4. 算法对比：运行三种算法并绘制对比图
5. 正则化对比：测试不同λ值对收敛的影响
6. PGD详细分析：绘制近端梯度下降法的单独收敛曲线

【评估指标】
- 收敛速度：达到相同精度所需的迭代次数
- 最终精度：算法停止时与最优解的距离
- 运行时间：算法执行的总时间

【输出结果】
1. 控制台输出：算法运行信息、收敛状态
2. 图表文件（每个问题4张图，共8张）：
   LASSO问题：
   - lasso_1_compare_algorithms.png: 三种算法对比图（对数坐标）
   - lasso_2_lambda_comparison.png: 不同正则化系数(λ)对比图（对数坐标）
   - lasso_3_pgd_convergence.png: 近端梯度下降法单独收敛曲线（对数坐标）
   - lasso_4_algorithms_comparison_log.png: 算法对比图（对数坐标，备用）
   
   Logistic Regression问题：
   - logistic_1_compare_algorithms.png: 三种算法对比图（对数坐标）
   - logistic_2_lambda_comparison.png: 不同正则化系数(λ)对比图（对数坐标）
   - logistic_3_pgd_convergence.png: 近端梯度下降法单独收敛曲线（对数坐标）
   - logistic_4_algorithms_comparison_log.png: 算法对比图（对数坐标，备用）

【图表说明】
所有图表的横坐标：迭代次数 k (0, 1, 2, ...)
所有图表的纵坐标：最优解与当前解的2范数距离 ||x* - x^(k)||₂（对数尺度）

【典型结论】
- FISTA 收敛最快（O(1/k²)）
- PGD 中等速度（O(1/k)）
- CD 适合超高维问题但可能较慢
- λ越大，收敛越快但解越稀疏
- Logistic Regression 由于 Lipschitz 常数较小，可能收敛更快
"""


def compute_loss_values(x_path, A, b, loss_func, lam):
    """
    计算迭代路径中每一步的目标函数值

    【功能】
    给定算法的迭代轨迹，计算每次迭代的目标函数值 F(x_k)

    【参数说明】
    Args:
        x_path (list of ndarray): 迭代轨迹，x_path[k]是第k次迭代的解
        A (ndarray): 特征矩阵
        b (ndarray): 标签向量
        loss_func (callable): 损失函数（不包含正则项）
        lam (float): 正则化系数

    【返回值】
    Returns:
        loss_values (list of float): 每次迭代的目标函数值
                                     loss_values[k] = f(x_k) + λ*||x_k||_1
    """
    loss_values = []
    for x_k in x_path:
        # 计算光滑部分损失
        smooth_loss = loss_func(A, b, x_k)
        # 添加L1正则项
        total_loss = smooth_loss + lam * sum(abs(x_k))
        loss_values.append(total_loss)
    return loss_values


def run_experiment_for_problem(problem_name, A, b, gradient_func, loss_func, L,
                                base_lambda=0.1, max_iter=1000, use_cd=True):
    """
    运行单个问题的完整实验

    【参数说明】
    Args:
        problem_name (str): 问题名称，如 "LASSO" 或 "Logistic"
        A (ndarray): 特征矩阵
        b (ndarray): 标签向量
        gradient_func: 梯度函数
        loss_func: 损失函数
        L (float): Lipschitz 常数
        base_lambda (float): 基准正则化参数
        max_iter (int): 最大迭代次数
        use_cd (bool): 是否使用坐标下降法（仅适用于LASSO）

    【生成的图表】
    1. {problem_name}_1_compare_algorithms.png: 所有算法对比
    2. {problem_name}_2_lambda_comparison.png: 不同λ值对比
    3. {problem_name}_3_pgd_convergence.png: PGD单独收敛曲线
    4. {problem_name}_4_algorithms_comparison_log.png: 算法对比（对数坐标）
    """
    print("\n" + "="*70)
    print(f" {problem_name} 问题实验".center(70))
    print("="*70)

    # 计算最优步长
    alpha = 1.0 / L
    print(f"[参数] Lipschitz 常数 L = {L:.6f}")
    print(f"[参数] 学习率 α = 1/L = {alpha:.6e}")
    print(f"[参数] 基准正则化参数 λ = {base_lambda}")
    print(f"[参数] 最大迭代次数 = {max_iter}")

    # ==========================================
    # 步骤1: 计算 Ground Truth（基准λ）
    # ==========================================
    print("\n" + "-"*70)
    print(f"步骤 1/4: 计算 Ground Truth (λ = {base_lambda})")
    print("-"*70)
    x_star = solver_ground_truth.get_optimal_x(
        A, b, gradient_func, loss_func, base_lambda, alpha
    )
    # 计算最优目标函数值
    optimal_loss = loss_func(A, b, x_star) + base_lambda * sum(abs(x_star))
    print(f"[完成] Ground Truth 计算完成")
    print(f"  - 解的维度: {x_star.shape}")
    print(f"  - 最优目标函数值: F(x*) = {optimal_loss:.6e}")

    # ==========================================
    # 步骤2: 图1 - 所有算法对比图
    # ==========================================
    print("\n" + "-"*70)
    print("步骤 2/4: 绘制所有算法对比图")
    print("-"*70)

    # 运行 PGD
    print("[PGD] 运行近端梯度下降算法...")
    _, path_pgd, time_pgd = alg_pgd.solve(
        A, b, gradient_func, loss_func,
        base_lambda, alpha, max_iter
    )
    loss_pgd = compute_loss_values(path_pgd, A, b, loss_func, base_lambda)
    print(f"[PGD] 完成！迭代次数: {len(path_pgd)}, 运行时间: {time_pgd:.3f} 秒")

    # 运行 FISTA
    print("[FISTA] 运行快速迭代收缩阈值算法...")
    _, path_fista, time_fista = alg_fista.solve(
        A, b, gradient_func, loss_func,
        base_lambda, alpha, max_iter
    )
    loss_fista = compute_loss_values(path_fista, A, b, loss_func, base_lambda)
    print(f"[FISTA] 完成！迭代次数: {len(path_fista)}, 运行时间: {time_fista:.3f} 秒")

    # 准备绘图数据（计算解向量与最优解的2范数距离）
    results_dict = {
        'PGD (近端梯度下降)': plotter.calculate_solution_distances(path_pgd, x_star),
        'FISTA (快速迭代收缩阈值)': plotter.calculate_solution_distances(path_fista, x_star)
    }

    # 运行 ADMM（仅对LASSO问题，因为它不需要梯度函数）
    path_admm = None
    time_admm = None
    if use_cd:  # LASSO问题才运行ADMM
        print("[ADMM] 运行交替方向乘子法...")
        _, path_admm, time_admm = alg_admm.solve(A, b, base_lambda, rho=1.0, max_iter=max_iter)
        loss_admm = compute_loss_values(path_admm, A, b, loss_func, base_lambda)
        print(f"[ADMM] 完成！迭代次数: {len(path_admm)}, 运行时间: {time_admm:.3f} 秒")
        results_dict['ADMM (交替方向乘子法)'] = plotter.calculate_solution_distances(path_admm, x_star)

    # 运行 CD（仅对LASSO问题）
    path_cd = None
    time_cd = None
    if use_cd:
        print("[CD] 运行坐标下降算法...")
        _, path_cd, time_cd = alg_cd.solve(A, b, base_lambda, max_iter)
        loss_cd = compute_loss_values(path_cd, A, b, loss_func, base_lambda)
        print(f"[CD] 完成！迭代次数: {len(path_cd)}, 运行时间: {time_cd:.3f} 秒")
        results_dict['CD (坐标下降)'] = plotter.calculate_solution_distances(path_cd, x_star)

    # 绘制对比图（对数坐标）
    print(f"\n[绘图] 生成图1: {problem_name} 所有算法对比图（对数坐标）...")
    plotter.plot_error_convergence(
        results_dict=results_dict,
        title=f"{problem_name} 所有算法收敛对比 (λ={base_lambda})",
        filename=f"{problem_name.lower()}_1_compare_algorithms.png"
    )

    # 绘制对比图（对数坐标，备用）
    print(f"[绘图] 生成图4: {problem_name} 所有算法对比图（对数坐标，备用）...")
    plotter.plot_error_convergence(
        results_dict=results_dict,
        title=f"{problem_name} 所有算法收敛对比 (λ={base_lambda}, 对数坐标)",
        filename=f"{problem_name.lower()}_4_algorithms_comparison_log.png"
    )

    # ==========================================
    # 步骤3: 图2 - 不同正则化系数对比图
    # ==========================================
    print("\n" + "-"*70)
    print("步骤 3/4: 绘制不同正则化系数对比图")
    print("-"*70)

    # 测试不同的λ值
    lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    lambda_results = {}

    print(f"[λ对比] 将测试 {len(lambda_values)} 个不同的正则化系数...")
    for idx, lam in enumerate(lambda_values, 1):
        print(f"\n[{idx}/{len(lambda_values)}] λ = {lam}")

        # 计算该λ的最优解和最优函数值
        print(f"  - 计算 Ground Truth...")
        x_star_lam = solver_ground_truth.get_optimal_x(
            A, b, gradient_func, loss_func, lam, alpha
        )
        optimal_loss_lam = loss_func(A, b, x_star_lam) + lam * sum(abs(x_star_lam))

        # 运行PGD
        print(f"  - 运行 PGD 算法...")
        _, path_pgd_lam, time_lam = alg_pgd.solve(
            A, b, gradient_func, loss_func,
            lam, alpha, max_iter
        )
        print(f"  - 完成！迭代次数: {len(path_pgd_lam)}, 时间: {time_lam:.3f}秒")

        # 计算解向量与最优解的2范数距离
        distances_lam = plotter.calculate_solution_distances(path_pgd_lam, x_star_lam)
        lambda_results[lam] = distances_lam

    # 绘制不同λ的对比图
    print(f"\n[绘图] 生成图2: {problem_name} 不同正则化系数对比图...")
    plotter.plot_lambda_comparison(
        lambda_results_dict=lambda_results,
        algorithm_name="PGD",
        title=f"{problem_name} 不同正则化系数下的 PGD 收敛对比",
        filename=f"{problem_name.lower()}_2_lambda_comparison.png"
    )

    # ==========================================
    # 步骤4: 图3 - 近端梯度下降法单独图
    # ==========================================
    print("\n" + "-"*70)
    print("步骤 4/4: 绘制近端梯度下降法单独收敛曲线")
    print("-"*70)

    print(f"[绘图] 生成图3: {problem_name} PGD 单独收敛曲线...")
    plotter.plot_pgd_convergence(
        x_path=path_pgd,
        x_optimal=x_star,
        filename=f"{problem_name.lower()}_3_pgd_convergence.png"
    )

    # ==========================================
    # 实验总结报告
    # ==========================================
    print("\n" + "-"*70)
    print(f"{problem_name} 问题实验完成")
    print("-"*70)

    print(f"\n【{problem_name} 算法性能对比】")
    print("-"*70)
    print(f"{'算法':<20} {'迭代次数':<12} {'运行时间(秒)':<15} {'最终误差'}")
    print("-"*70)
    errors_pgd = results_dict['PGD (近端梯度下降)']
    errors_fista = results_dict['FISTA (快速迭代收缩阈值)']
    print(f"{'PGD':<20} {len(path_pgd):<12} {time_pgd:<15.3f} {errors_pgd[-1]:.6e}")
    print(f"{'FISTA':<20} {len(path_fista):<12} {time_fista:<15.3f} {errors_fista[-1]:.6e}")
    if use_cd:
        errors_cd = results_dict['CD (坐标下降)']
        print(f"{'CD':<20} {len(path_cd):<12} {time_cd:<15.3f} {errors_cd[-1]:.6e}")
    print("-"*70)

    print(f"\n【生成的 {problem_name} 图表文件】")
    print("-"*70)
    print(f"1. {problem_name.lower()}_1_compare_algorithms.png      - 所有算法对比（对数坐标）")
    print(f"2. {problem_name.lower()}_2_lambda_comparison.png       - 不同正则化系数对比（对数坐标）")
    print(f"3. {problem_name.lower()}_3_pgd_convergence.png         - 近端梯度下降法收敛曲线（对数坐标）")
    print(f"4. {problem_name.lower()}_4_algorithms_comparison_log.png - 所有算法对比（对数坐标，备用）")
    print("-"*70)


def run_experiment():
    """
    运行完整的 LASSO 和 Logistic Regression 算法性能对比实验

    【实验设置】
    - 数据集: a1a (1605 样本, 123 特征)
    - 基准正则化参数: λ = 0.1
    - 最大迭代: 1000 次
    - 收敛容忍度: 1e-6
    - 纵坐标：所有图表使用对数尺度

    【实验步骤】
    第一部分：LASSO 问题实验
    1. 加载数据并计算 Lipschitz 常数
    2. 计算基准λ的 Ground Truth（真实最优解）
    3. 绘制所有算法对比图（PGD, FISTA, CD）
    4. 绘制不同正则化系数对比图（λ = 0.01, 0.05, 0.1, 0.5, 1.0）
    5. 绘制近端梯度下降法单独收敛曲线

    第二部分：Logistic Regression 问题实验
    1-5. 与 LASSO 相同的步骤（不包含CD算法）

    【生成的图表】
    共8张图表（每个问题4张）
    LASSO: lasso_1~4.png
    Logistic: logistic_1~4.png

    所有图表纵坐标均为对数尺度
    """
    print("="*70)
    print(" "*10 + "LASSO 和 Logistic Regression 算法性能对比实验")
    print("="*70)

    # ==========================================
    # 加载数据
    # ==========================================
    print("\n" + "="*70)
    print("数据加载")
    print("="*70)
    A, b = data_loader.load_data("a1a")
    print(f"[完成] 数据加载成功")
    print(f"  - 特征矩阵 A: {A.shape} (样本数 × 特征数)")
    print(f"  - 标签向量 b: {b.shape}")

    # ==========================================
    # 第一部分：LASSO 问题实验
    # ==========================================
    print("\n" + "="*70)
    print("第一部分：LASSO 问题实验")
    print("="*70)

    L_lasso = problem_lasso.calc_lipschitz(A)
    run_experiment_for_problem(
        problem_name="LASSO",
        A=A,
        b=b,
        gradient_func=problem_lasso.gradient,
        loss_func=problem_lasso.loss,
        L=L_lasso,
        base_lambda=0.1,
        max_iter=1000,
        use_cd=True  # LASSO 问题可以使用坐标下降法
    )

    # ==========================================
    # 第二部分：Logistic Regression 问题实验
    # ==========================================
    print("\n" + "="*70)
    print("第二部分：Logistic Regression 问题实验")
    print("="*70)

    L_logistic = problem_logistic.calc_lipschitz(A)
    run_experiment_for_problem(
        problem_name="Logistic",
        A=A,
        b=b,
        gradient_func=problem_logistic.gradient,
        loss_func=problem_logistic.loss,
        L=L_logistic,
        base_lambda=0.1,
        max_iter=1000,
        use_cd=False  # Logistic Regression 不使用坐标下降法
    )

    # ==========================================
    # 总体实验总结
    # ==========================================
    print("\n" + "="*70)
    print(" "*25 + "实验全部完成!")
    print("="*70)

    print("\n【实验总结】")
    print("-"*70)
    print("✓ 完成 LASSO 问题的所有实验和图表生成")
    print("✓ 完成 Logistic Regression 问题的所有实验和图表生成")
    print("✓ 所有图表纵坐标均使用对数尺度")
    print("-"*70)

    print("\n【生成的图表文件（共8张）】")
    print("-"*70)
    print("LASSO 问题（4张图）：")
    print("  1. lasso_1_compare_algorithms.png      - 所有算法对比（对数坐标）")
    print("  2. lasso_2_lambda_comparison.png       - 不同正则化系数对比（对数坐标）")
    print("  3. lasso_3_pgd_convergence.png         - 近端梯度下降法收敛曲线（对数坐标）")
    print("  4. lasso_4_algorithms_comparison_log.png - 所有算法对比（对数坐标，备用）")
    print()
    print("Logistic Regression 问题（4张图）：")
    print("  5. logistic_1_compare_algorithms.png   - 所有算法对比（对数坐标）")
    print("  6. logistic_2_lambda_comparison.png    - 不同正则化系数对比（对数坐标）")
    print("  7. logistic_3_pgd_convergence.png      - 近端梯度下降法收敛曲线（对数坐标）")
    print("  8. logistic_4_algorithms_comparison_log.png - 所有算法对比（对数坐标，备用）")
    print("-"*70)

    print("\n【图表说明】")
    print("- 横坐标: 迭代次数 k")
    print("- 纵坐标: 最优解与当前解的2范数距离 ||x* - x^(k)||_2 (对数尺度)")
    print("- 对数坐标下曲线越陡峭，收敛越快")
    print("- 对数坐标下直线斜率反映收敛速率")

    print("\n【关键结论】")
    print("- FISTA 收敛最快，适合对速度要求高的场景")
    print("- PGD 实现简单，收敛稳定，适合入门学习")
    print("- CD 适合超高维稀疏问题（仅用于LASSO）")
    print("- λ 越大，收敛越快，但解越稀疏（更多零元素）")
    print("- Logistic Regression 由于 Lipschitz 常数较小，可能比 LASSO 收敛更快")
    print("="*70)


if __name__ == "__main__":
    run_experiment()
if __name__ == "__main__":
    run_experiment()
