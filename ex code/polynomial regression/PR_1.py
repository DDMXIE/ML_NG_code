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
def costFunction(X1,X2,y,theta_0_list,theta_1_list,theta_2_list,iters):
    """
    计算不同迭代次数下的代价
    :param X1:特征1
    :param X2:特征2
    :param y:标签向量
    :param theta_0_list:梯度下降的Ѳ0列表
    :param theta_1_list:梯度下降的Ѳ1列表
    :param theta_2_list:梯度下降的Ѳ2列表
    :param iters:迭代次数
    :return:cost_list 不同迭代次数下的代价列表
    """
    cost_list = []
    for it in range(0, iters):
        h_x = theta_0_list[it] + theta_1_list[it] * X1 + theta_2_list[it] * X2
        A = (h_x - y) * (h_x - y)
        cost_list.append(np.sum(A) / (2 * len(X1)))
    return cost_list

if __name__ == '__main__':

    # 学习的数据
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([2, 7, 12, 15, 17, 18, 18.5, 18.3, 18.2])

    # 将多次项令成1次项的不同特征，再利用线性回归求解
    X1 = X
    X2 = X ** 0.5

    # 梯度下降优化算法
    alpha = 0.01
    iters = 100000
    theta = 0
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    theta_0_list, theta_1_list, theta_2_list = gradientDescent(X1, X2, y, theta_0, theta_1, theta_2 ,alpha, iters)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_1_optimize = theta_1_list[-1:][0]
    theta_2_optimize = theta_2_list[-1:][0]

    # 梯度下降的代价与迭代次数的图像
    cost_list = costFunction(X1, X2, y, theta_0_list, theta_1_list, theta_2_list, iters)
    print(cost_list)
    iter_vec = np.arange(iters)
    cost_vec = cost_list
    plt.ylabel("J(Ѳ)")
    plt.xlabel("iter")
    plt.ylim(0, 10)
    plt.plot(iter_vec, cost_vec)
    plt.show()

    # 数据显示
    x = np.linspace(1, 10, num=100)
    plt.plot(x, theta_0_optimize + theta_1_optimize * np.array(x)
             + theta_2_optimize * (x ** 0.5), color='r')
    plt.scatter(x=X, y=y, s=5)
    plt.show()