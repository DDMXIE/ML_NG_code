import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    # 这个部分计算J(Ѳ)，X是矩阵
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def computeCost2(X, y, theta):
    # 不看上面cell，尝试自己实现一下
    arr = np.power((X * theta.T - y), 2)
    return np.sum(arr) / (2 * len(X))

if __name__ == '__main__':

    # 输出一个5*5的单位矩阵
    A = np.eye(5)

    # 读取数据
    path = '../dataset/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data.head())

    data.insert(0, 'Ones', 1)
    print(data)

    # 初始化X和y
    cols = data.shape[1]
    X = data.iloc[:, :-1]  # X是data里的除最后列
    y = data.iloc[:, cols - 1:cols]  # y是data最后一列
    print(X)
    print(y)

    # 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))
    print(X.shape, theta.shape, y.shape)

    # 计算代价函数
    res = computeCost2(X, y, theta)
    print(res)

    # 数据显示
    plt.scatter(x=data['Population'], y=data['Profit'], s=5)
    plt.show()
