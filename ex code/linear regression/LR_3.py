# @Title    : 多变量线性回归 多变量梯度下降优化算法
# @Author   : tony
# @Date     : 2021/7/25
# @Dec      : h(x) = Ѳ0 + Ѳ1*x1 + Ѳ2*x2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gradientDescent(X1, X2, y, theta_0, theta_1, theta_2 ,alpha, iters):
    """
    梯度下降求使J(Ѳ)最小的Ѳ
    :param X1: Size特征
    :param X2: Bedrooms特征
    :param y: Price向量
    :param theta_0: h(x) = Ѳ0 + Ѳ1*x1 + Ѳ2*x2 中的参数值 Ѳ0
    :param theta_1: h(x) = Ѳ0 + Ѳ1*x1 + Ѳ2*x2 中的参数值 Ѳ1
    :param theta_2: h(x) = Ѳ0 + Ѳ1*x1 + Ѳ2*x2 中的参数值 Ѳ2
    :param alpha: 迭代次数
    :param iters: 梯度下降的theta值
    :return: theta_0_list, theta_1_list, theta_2_list 梯度下降的theta值
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

    # 读取数据
    path = '../dataset/ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data)

    # 学习的数据
    Size = np.array(data['Size'])
    Bedrooms = np.array(data['Bedrooms'])
    Price = np.array(data['Price'])

    # 因特征的取值范围相差很大，故需要对其进行均值归一化
    len_X1 = len(Size)
    mean_X1 = np.mean(np.array(Size))
    len_X2 = len(Bedrooms)
    mean_X2 = np.mean(np.array(Bedrooms))
    X1 = (np.array(Size) - mean_X1) / np.std(Size)
    X2 = (np.array(Bedrooms) - mean_X2) / np.std(Bedrooms)
    y  = (np.array(Price) - np.mean(np.array(Price))) / np.std(Price)
    print(X1)
    print(X2)

    # 多元梯度下降
    alpha = 0.01
    iters = 1500
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    theta_0_list, theta_1_list, theta_2_list = gradientDescent(X1, X2, y, theta_0, theta_1, theta_2, alpha, iters)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_1_optimize = theta_1_list[-1:][0]
    theta_2_optimize = theta_2_list[-1:][0]
    print(theta_0_optimize, theta_1_optimize, theta_2_optimize)
    print(theta_0_optimize + theta_1_optimize * Size + theta_2_optimize * Bedrooms)

    # 数据显示
    x = np.linspace(-3, 3, num=100)
    x2 = np.linspace(-3, 3, num=100)
    plt.plot(x, (theta_0_optimize + theta_1_optimize * x + theta_2_optimize * x2), color='r')
    # plt.plot(X1, (theta_0_optimize + theta_1_optimize * X1 + theta_2_optimize * X2), color='r')
    plt.scatter(x=X1, y=y, s=5)
    plt.scatter(x=X2, y=y, s=5)
    plt.show()



