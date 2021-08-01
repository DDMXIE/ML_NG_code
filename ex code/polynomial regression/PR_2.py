# @Title    : 针对真实数据 单变量多项式回归 梯度下降优化算法
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
    X = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
         59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
         87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
    y = np.array(
        [-0.0919716355524517, -0.11732556117751325, -0.12179304966906895, -0.09921602825626914, -0.05646850980952155,
         -0.03207869704894413, -0.041649975887450216, -0.07124004937223312, -0.09999177011664989, -0.1005357396178278,
         -0.07294345892618317, -0.024412921341205162, 0.004146011931592987, 0.002033120095929036, -0.026700071633855117,
         -0.055339469092555715, -0.0575429221992409, -0.03007792163463321, 0.017585780884976112, 0.042867065124581424,
         0.037855279874906916, 0.00900568846957811, -0.022483406968751573, -0.028663842977373188, -0.004254784694354014,
         0.042715936957082715, 0.07813573818683564, 0.07183513919919697, 0.04109302644048567, 0.01006112334781734,
         0.004963838004086351, 0.03228573315176301, 0.08452746539080996, 0.1253491057964536, 0.12045204292146364,
         0.09240034955647779, 0.07161067951068027, 0.09935901854714468, 0.20958151584004592, 0.3219905834124316,
         0.37600871735247454, 0.3727151806738381, 0.3250367652604191, 0.27794869430867836, 0.3089916504901006,
         0.35081315248768913, 0.42320410021323907, 0.4687612959924745, 0.45919404533763686, 0.40399672209624915,
         0.3455435285063131, 0.3319597196244849, 0.37219516962411103, 0.44818841751017724, 0.49165097080967296,
         0.470463844503421, 0.4103182247893988, 0.35610095408970627, 0.3435278935943688, 0.3931133683736135,
         0.4653681725828682, 0.5079418527345779, 0.48858579220477855, 0.4235310787340462, 0.3636615287550784,
         0.3541477091488158, 0.40045793149377173, 0.48194684053167286, 0.5299259317311155, 0.5051287026741929,
         0.43905692215456044, 0.38382091348240777, 0.3701550157219481, 0.42080582305205777, 0.5084556531776729,
         0.5499795696578689, 0.5180214994467611, 0.45073199743841796, 0.39850317667816837, 0.3854379225086843,
         0.4312291019484215, 0.4982439451730351, 0.5328372281777763, 0.5096947264474463, 0.44737345765508596,
         0.3988094279252865, 0.3890482969663854, 0.43935865923599204, 0.5168307654293036, 0.5557541843139254,
         0.5335968393583446, 0.4708434728300072, 0.41783761369704947, 0.40589464821492377, 0.4463360604411747,
         0.5272393435928036, 0.5685939843815256, 0.5424753084022352, 0.4758646471005004])

    # 将多次项令成1次项的不同特征，再利用线性回归求解
    X_1 = X
    X_2 = X * X
    X_3 = X * X * X
    X_4 = X * X * X * X

    # 因特征的取值范围相差很大，故需要对其进行均值归一化
    mean_X1 = np.mean(X_1)
    mean_X2 = np.mean(X_2)
    mean_X3 = np.mean(X_3)
    mean_X4 = np.mean(X_4)
    X1 = (np.array(X_1) - mean_X1) / np.std(X_1)
    X2 = (np.array(X_2) - mean_X2) / np.std(X_2)
    X3 = (np.array(X_3) - mean_X3) / np.std(X_3)
    X4 = (np.array(X_4) - mean_X4) / np.std(X_4)
    y = (np.array(y) - np.mean(np.array(y))) / np.std(y)

    # 梯度下降优化算法
    alpha = 0.3
    iters = 100000
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
    plt.plot(iter_vec, cost_vec)
    plt.show()

    # 数据显示
    x = np.linspace(1, 100, num=100)
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