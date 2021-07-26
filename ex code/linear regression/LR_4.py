# @Title    : 单变量线性回归 正规方程优化算法
# @Author   : tony
# @Date     : 2021/7/25
# @Dec      : h(x) = Ѳ0 + Ѳ1*x
# @Dataset  : ../dataset/ex1data1.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalEqn(X,y):
    """
    正规方程 求使J(Ѳ)最小的Ѳ
    :param X:特征矩阵
    :param y:向量
    :return:优化后的theta列表
    """
    # X.T@X等价于X.T.dot(X)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

if __name__ == '__main__':

    # 读取数据
    path = '../dataset/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data)

    # 学习的数据
    X = pd.DataFrame(data['Population'])
    y = np.array(data['Profit'])
    X.insert(0, 'Ones', 1)
    print(X)
    print(y)

    # 优化算法 - 正规方程
    theta_list = normalEqn(X, y)    # 正规方程方法无需迭代
    print(theta_list)

    # 找到最优化后的参数值
    theta_0_optimize = theta_list[0]
    theta_optimize = theta_list[1]
    print(theta_0_optimize)
    print(theta_optimize)

    # 数据显示
    plt.plot(np.array(data['Population']), theta_0_optimize + theta_optimize * np.array(data['Population']), color='r')
    plt.scatter(x=data['Population'], y=data['Profit'], s=5)
    plt.show()