# @Title    : 多项式回归 梯度下降优化算法
# @Author   : tony
# @Date     : 2021/7/26
# @Dec      : h(x) = Ѳ0 + Ѳ1*x + Ѳ2*sqrt(x)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(X1, X2, y, theta_0, theta_1, theta_2 ,alpha, iters):
    """
    梯度下降求使J(Ѳ)最小的Ѳ
    :param X:Population向量
    :param y:Profit向量
    :param theta:h(x)中的参数值
    :param alpha:学习速率，梯度下降时的步长
    :param iters:迭代次数
    :return:theta_list 梯度下降的theta值
    """
    theta_0_list = [theta_0]
    theta_1_list = [theta_1]
    theta_2_list = [theta_2]
    for it in range(0, iters):
        theta_0 = theta_0_list[-1:][0]
        theta_1 = theta_1_list[-1:][0]
        theta_2 = theta_2_list[-1:][0]

        theta_0_new = theta_0 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 - y))
        theta_1_new = theta_1 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 - y) * X1)
        theta_2_new = theta_2 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 - y) * X2)
        theta_0_list.append(theta_0_new)
        theta_1_list.append(theta_1_new)
        theta_2_list.append(theta_2_new)
    return theta_0_list, theta_1_list, theta_2_list

if __name__ == '__main__':

    # 学习的数据
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([2, 7, 12, 15, 17, 18, 18.5, 18.3, 18.2])

    # 将多次项令成1次项的不同特征，再利用线性回归求解
    X1 = X
    X2 = X ** 0.5

    # 梯度下降优化算法
    alpha = 0.01
    iters = 50000
    theta = 0
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    theta_0_list, theta_1_list, theta_2_list = gradientDescent(X1, X2, y, theta_0, theta_1, theta_2 ,alpha, iters)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_1_optimize = theta_1_list[-1:][0]
    theta_2_optimize = theta_2_list[-1:][0]

    # 数据显示
    plt.plot(X, theta_0_optimize + theta_1_optimize * np.array(X)
             + theta_2_optimize * (X ** 0.5), color='r')
    plt.scatter(x=X, y=y, s=5)
    plt.show()