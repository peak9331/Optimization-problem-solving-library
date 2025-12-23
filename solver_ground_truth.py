import alg_fista

def get_optimal_x(A, b, grad_func, loss_func, lam, alpha):
    """
    使用 FISTA 跑 10000 轮，tol=1e-12，以此结果近似作为 x*
    """
    print(f"[Ground Truth] 计算最优解 (Lambda={lam})...")
    x_star, _, _ = alg_fista.solve(
        A, b, grad_func, loss_func,
        lam, alpha,
        max_iter=10000,
        tol=1e-12
    )
    return x_star