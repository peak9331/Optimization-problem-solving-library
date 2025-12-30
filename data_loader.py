import os
import requests
import numpy as np
from sklearn.datasets import load_svmlight_file

"""
===========================================
数据加载模块 - LIBSVM 格式数据处理
===========================================

【文件作用】
本文件负责加载和处理LIBSVM格式的数据集，为优化算法提供特征矩阵A和标签向量b。
支持从网络自动下载数据集，并转换为NumPy数组格式。

【主要功能】
- load_data(): 加载指定的LIBSVM数据集
- download_data(): 从网络下载数据集（如果本地不存在）
- 自动转换稀疏矩阵为密集矩阵
- 返回特征矩阵A和标签向量b

【模块作用】
从网络或本地加载 LIBSVM 格式的数据集，并转换为优化问题所需的矩阵形式

【LIBSVM 格式简介】
LIBSVM 是支持向量机（SVM）库使用的标准数据格式：
- 每行代表一个样本
- 格式：<label> <index1>:<value1> <index2>:<value2> ...
- 稀疏存储：只记录非零特征

例如：
+1 1:0.5 3:0.8 10:1.2
-1 2:0.3 5:0.9

【数据来源】
台湾大学 LIBSVM 数据集仓库：
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

常用数据集：
- a1a: 二分类，1605 样本，123 特征
- a2a: 二分类，2265 样本，123 特征
- w1a: 二分类，2477 样本，300 特征

【转换关系】
LIBSVM 数据 → (A, b)
- A: 特征矩阵 (n_samples × n_features)
- b: 标签向量 (n_samples,)，值域 {-1, +1}

【优化问题映射】
加载后的 (A, b) 可用于：
- LASSO: min (1/2)||Ax - b||₂² + λ||x||₁
- Logistic: min Σ log(1 + exp(-bᵢ·aᵢᵀx)) + λ||x||₂²
"""


def load_data(filename="a1a"):
    """
    加载 LIBSVM 格式数据集

    【功能】
    1. 检查本地是否存在数据文件
    2. 如果不存在，从网络下载
    3. 解析 LIBSVM 格式，转换为 NumPy 数组
    4. 标准化标签为 {-1, +1}

    【参数说明】
    Args:
        filename (str): 数据文件名（不含路径），默认 "a1a"
                       例如："a1a", "a2a", "w1a"

    【返回值】
    Returns:
        A (ndarray): 特征矩阵，形状 (n_samples, n_features)
                     每一行是一个样本的特征向量
        b (ndarray): 标签向量，形状 (n_samples,)
                     每个元素是 +1 或 -1

    【数据处理细节】
    1. 稀疏→密集转换：
       - LIBSVM 存储为稀疏矩阵（scipy.sparse）
       - 转换为密集矩阵（numpy.ndarray）便于矩阵运算

    2. 标签标准化：
       - 某些数据集标签是 {0, 1}
       - 统一转换为 {-1, +1}（SVM 和逻辑回归的标准形式）

    【使用示例】
    >>> A, b = load_data("a1a")
    [Data] 数据集加载完成: 样本数=1605, 特征数=123
    >>> print(A.shape)  # (1605, 123)
    >>> print(np.unique(b))  # [-1  1]
    """
    # 构建数据集 URL
    url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{filename}"

    # 检查本地文件是否存在
    if not os.path.exists(filename):
        print(f"[Data] 正在下载 {filename} ...")
        print(f"[Data] 下载地址: {url}")

        # 从网络下载数据
        r = requests.get(url)

        # 保存到本地
        with open(filename, 'wb') as f:
            f.write(r.content)

        print(f"[Data] 下载完成，已保存到 {filename}")
    else:
        print(f"[Data] 发现本地文件 {filename}，直接加载")

    # 加载 LIBSVM 格式数据
    # data: 稀疏特征矩阵 (scipy.sparse)
    # target: 标签数组
    data, target = load_svmlight_file(filename)

    # 转换为密集矩阵，这就是优化问题中的 A
    A = data.toarray()  # 形状: (n_samples, n_features)

    # 这就是优化问题中的 b
    b = target  # 形状: (n_samples,)

    # 标签修正：确保是 {-1, 1}
    # 因为某些数据集标签可能是 {0, 1}，需要统一为 {-1, 1}
    b = np.where(b <= 0, -1, 1)

    # 打印数据集信息
    print(f"[Data] 数据集加载完成: 样本数={A.shape[0]}, 特征数={A.shape[1]}")
    print(f"[Data] 标签分布: {np.sum(b == 1)} 个正样本, {np.sum(b == -1)} 个负样本")
    print(f"[Data] 这对应于优化问题: min (1/2)||Ax - b||₂² + λ||x||₁")
    print(f"[Data] 其中 A.shape={A.shape}, b.shape={b.shape}, x 待求解")

    return A, b