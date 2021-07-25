# @Title    : 单变量线性回归 单变量梯度下降优化算法
# @Author   : tony
# @Date     : 2021/7/25
# @Dec      : h(x) = Ѳ0 + Ѳ1*x

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, theta_0, alpha, iters):
    """
    梯度下降求使J(Ѳ)最小的Ѳ
    :param X:Population向量
    :param y:Profit向量
    :param theta:h(x)中的参数值
    :param alpha:学习速率，梯度下降时的步长
    :param iters:迭代次数
    :return:theta_list 梯度下降的theta值
    """
    theta_list = [theta]
    theta_0_list = [theta_0]
    for it in range(0, iters):
        theta_j = theta_list[-1:][0]
        theta_0 = theta_0_list[-1:][0]
        theta_j_new = theta_j - ((alpha / len(X)) * np.sum((theta_0 + theta_j * X - y) * X))
        theta_0_new = theta_0 - ((alpha / len(X)) * np.sum((theta_0 + theta_j * X - y)))
        theta_list.append(theta_j_new)
        theta_0_list.append(theta_0_new)
    return theta_0_list, theta_list

if __name__ == '__main__':

    # 读取数据
    path = '../dataset/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data)

    # 学习的数据
    X = np.array(data['Population'])
    y = np.array(data['Profit'])
    print(X)
    print(y)

    # 梯度下降优化算法
    alpha = 0.01
    iters = 1500
    theta = 0
    theta_0 = 0
    theta_0_list, theta_list = gradientDescent(X, y, theta, theta_0, alpha, iters)
    print(theta_list)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_optimize = theta_list[-1:][0]
    print(theta_0_optimize)
    print(theta_optimize)

    # 数据显示
    plt.plot(np.array(data['Population']), theta_0_optimize + theta_optimize * np.array(data['Population']), color='r')
    plt.scatter(x=data['Population'], y=data['Profit'], s=5)
    plt.show()