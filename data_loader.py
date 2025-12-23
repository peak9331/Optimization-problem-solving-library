import os
import requests
import numpy as np
from sklearn.datasets import load_svmlight_file


def load_data(filename="a1a"):
    """
    下载并加载 a1a 数据集，确保标签为 {-1, 1}
    """
    url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{filename}"

    if not os.path.exists(filename):
        print(f"[Data] 正在下载 {filename} ...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

    data, target = load_svmlight_file(filename)
    A = data.toarray()
    b = target

    # 标签修正：确保是 {-1, 1}
    b = np.where(b <= 0, -1, 1)

    print(f"[Data] 数据集加载完成: 样本数={A.shape[0]}, 特征数={A.shape[1]}")
    return A, b