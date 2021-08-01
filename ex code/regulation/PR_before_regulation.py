# @Title    : 多项式回归 梯度下降优化算法 正则化
# @Author   : tony
# @Date     : 2021/8/1
# @Dec      : h(x) = Ѳ0 + Ѳ1*x + Ѳ2*x^2 + Ѳ3*x^3 + Ѳ4*x^4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(X1, X2, X3, X4, y, theta_0, theta_1, theta_2, theta_3, theta_4 ,alpha, iters):
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
    theta_3_list = [theta_3]
    theta_4_list = [theta_4]
    for it in range(0, iters):
        theta_0 = theta_0_list[-1:][0]
        theta_1 = theta_1_list[-1:][0]
        theta_2 = theta_2_list[-1:][0]
        theta_3 = theta_3_list[-1:][0]
        theta_4 = theta_4_list[-1:][0]

        theta_0_new = theta_0 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y))
        # print(theta_0,theta_1,theta_2,theta_3,theta_4)
        # print(alpha / len(X1), np.sum(theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y))
        theta_1_new = theta_1 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y)* X1)
        theta_2_new = theta_2 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y)* X2)
        theta_3_new = theta_3 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y)* X3)
        theta_4_new = theta_4 - (alpha / len(X1)) * np.sum((theta_0 + theta_1 * X1 + theta_2 * X2 + theta_3 * X3 + theta_4 * X4 - y)* X4)
        theta_0_list.append(theta_0_new)
        theta_1_list.append(theta_1_new)
        theta_2_list.append(theta_2_new)
        theta_3_list.append(theta_3_new)
        theta_4_list.append(theta_4_new)
    return theta_0_list, theta_1_list, theta_2_list, theta_3_list, theta_4_list

def costFunction(X1,X2,X3,X4,y,theta_0_list,theta_1_list,theta_2_list, theta_3_list, theta_4_list,iters):
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
        h_x = theta_0_list[it] + theta_1_list[it] * X1 + theta_2_list[it] * X2 + theta_3_list[it] * X3 \
              + theta_4_list[it] * X4
        A = (h_x - y) * (h_x - y)
        cost_list.append(np.sum(A) / (2 * len(X1)))
    return cost_list

if __name__ == '__main__':

    # 学习的数据
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 7, 12, 10, 8, 7, 11, 13, 12, 14])

    # 将多次项令成1次项的不同特征，再利用线性回归求解
    # X1 = X
    # X2 = X * X
    # X3 = X * X * X
    # X4 = X * X * X * X

    # 因特征的取值范围相差很大，故需要对其进行均值归一化
    mean_X1 = np.mean(np.array(X))
    mean_X2 = np.mean(np.array(X * X))
    mean_X3 = np.mean(np.array(X * X * X))
    mean_X4 = np.mean(np.array(X * X * X * X))
    X1 = (np.array(X) - mean_X1) / np.std(X)
    X2 = (np.array(X * X) - mean_X2) / np.std(X * X)
    X3 = (np.array(X * X * X) - mean_X3) / np.std(X * X * X)
    X4 = (np.array(X * X * X * X) - mean_X4) / np.std(X * X * X * X)
    y = (np.array(y) - np.mean(np.array(y))) / np.std(y)

    # 梯度下降优化算法
    alpha = 0.3
    iters = 1000000
    theta = 0
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    theta_3 = 0
    theta_4 = 0
    theta_0_list, theta_1_list, theta_2_list, theta_3_list, theta_4_list = gradientDescent(X1, X2, X3, X4, y, theta_0,
                                                                                           theta_1, theta_2, theta_3,
                                                                                           theta_4, alpha, iters)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_1_optimize = theta_1_list[-1:][0]
    theta_2_optimize = theta_2_list[-1:][0]
    theta_3_optimize = theta_3_list[-1:][0]
    theta_4_optimize = theta_4_list[-1:][0]
    print("优化参数结果：", theta_0_optimize, theta_1_optimize, theta_2_optimize, theta_3_optimize, theta_4_optimize)

    # 梯度下降的代价与迭代次数的图像
    cost_list = costFunction(X1, X2, X3, X4, y, theta_0_list, theta_1_list, theta_2_list, theta_3_list, theta_4_list, iters)
    iter_vec = np.arange(iters)
    cost_vec = cost_list
    plt.ylabel("J(Ѳ)")
    plt.xlabel("iter")
    # plt.ylim(0, 10)
    plt.plot(iter_vec, cost_vec)
    plt.show()

    # 数据显示
    x = np.linspace(1, 10, num=100)
    x1 = x
    x2 = x * x
    x3 = x * x * x
    x4 = x * x * x * x
    mean_x1 = np.mean(np.array(x1))
    mean_x2 = np.mean(np.array(x2))
    mean_x3 = np.mean(np.array(x3))
    mean_x4 = np.mean(np.array(x4))
    x_1 = (np.array(x1) - mean_x1) / np.std(x1)
    x_2 = (np.array(x2) - mean_x2) / np.std(x2)
    x_3 = (np.array(x3) - mean_x3) / np.std(x3)
    x_4 = (np.array(x4) - mean_x4) / np.std(x4)
    plt.plot(x, theta_0_optimize + theta_1_optimize * np.array(x_1)
             + theta_2_optimize * np.array(x_2) + theta_3_optimize * np.array(x_3) + theta_4_optimize * np.array(x_4)
             , color='r')
    plt.scatter(x=X, y=y, s=5)
    plt.show()