# @Title    : 多变量逻辑回归 梯度下降算法
# @Author   : tony
# @Date     : 2021/7/26
# @Dec      : h(x) = sigmoid(z)
# @Dataset  : ../dataset/ex2data1.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    模块化sigmoid函数
    :param z: 函数变量
    :return: sigmoid函数值
    """
    return 1 / (1 + np.exp(-z))

def gradientDescent(X1, X2, y, theta_0, theta_1, theta_2, alpha, iters):
    """

    :param X1:特征1
    :param X2:特征2
    :param y:标签向量
    :param theta_0:h(x)中的参数值
    :param theta_1:h(x)中的参数值
    :param theta_2:h(x)中的参数值
    :param alpha:学习率
    :param iters:迭代次数
    :return: theta_0_list, theta_1_list, theta_2_list
    """
    theta_0_list = [theta_0]
    theta_1_list = [theta_1]
    theta_2_list = [theta_2]
    for it in range(0, iters):
        theta_0 = theta_0_list[-1:][0]
        theta_1 = theta_1_list[-1:][0]
        theta_2 = theta_2_list[-1:][0]
        # theta_vec = np.array([theta_0,theta_1,theta_2])
        z = theta_0 + theta_1 * X1 + theta_2 * X2
        h_x = sigmoid(z)
        theta_0_new = theta_0 - (alpha / len(X1)) * np.sum(h_x - y)
        theta_1_new = theta_1 - (alpha / len(X1)) * np.sum((h_x - y) * X1)
        theta_2_new = theta_2 - (alpha / len(X1)) * np.sum((h_x - y) * X2)
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
        z = theta_0_list[it] + theta_1_list[it] * X1 + theta_2_list[it] * X2
        h_x = sigmoid(z)
        A = np.dot(y, np.log(h_x)) + np.dot((1 - y), np.log(1 - h_x))
        cost_list.append(- np.sum(A) / len(X1))
    return cost_list

if __name__ == '__main__':

    # 读取数据
    path = '../dataset/ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    print(data)

    # 处理待学习的样本数据
    X1 = np.array(data['Exam 1'])
    X2 = np.array(data['Exam 2'])
    y = np.array(data['Admitted'])

    # 梯度下降优化算法
    alpha = 0.001
    iters = 1000000
    theta = 0
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    theta_0_list, theta_1_list, theta_2_list = gradientDescent(X1, X2, y, theta_0, theta_1, theta_2, alpha, iters)

    # 找到最优化后的参数值
    theta_0_optimize = theta_0_list[-1:][0]
    theta_1_optimize = theta_1_list[-1:][0]
    theta_2_optimize = theta_2_list[-1:][0]
    print(theta_0_optimize, theta_1_optimize, theta_2_optimize)

    # 梯度下降的代价与迭代次数的图像
    cost_list = costFunction(X1, X2, y, theta_0_list, theta_1_list, theta_2_list, iters)
    print(cost_list)
    iter_vec = np.arange(iters)
    cost_vec = cost_list
    plt.ylabel("J(Ѳ)")
    plt.xlabel("iter")
    plt.plot(iter_vec, cost_vec)
    plt.show()

    # 数据显示
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    plotting_x1 = np.linspace(30, 100, 100)
    ax.plot(plotting_x1, (- theta_0_optimize - theta_1_optimize * plotting_x1) / theta_2_optimize, color='y',label='Prediction')
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()