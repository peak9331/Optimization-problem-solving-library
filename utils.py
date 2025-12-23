import numpy as np

def soft_thresholding(u, threshold):
    """
    L1 正则化的近端算子 (Soft Thresholding Operator)
    prox(u) = sign(u) * max(|u| - threshold, 0)
    """
    return np.sign(u) * np.maximum(np.abs(u) - threshold, 0)