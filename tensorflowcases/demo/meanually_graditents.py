#梯度下降预测房价 demo

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

n_epochs = 136500
learning_rate = 0.001

housing = fetch_california_housing(data_home='C:/Users/Shinelon/scikit_learn_data',download_if_missing=True)
m, n = housing.data.shape
print(m, n)
print(housing.target.shape)
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler().fit(housing_data_plus_bias)
scaler_horsing_data_plus_bias = scaler.transform(housing_data_plus_bias)
# 首先, tf.constant()定义了一个常量tensor,
X = tf.constant(scaler_horsing_data_plus_bias, dtype=tf.float32, name='X')
#定义真实值的标签 ，因为target原类型是(20640,) 是一位数组 现在需要转换为m行一列的二维数组(20640, 1)
Y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='Y')

#初始化9个  theta 并非是满足标准正太分布  为什么在-1 到 1 之间  因为需要保证梯度下降是  g梯度的调整大小一致
theta = tf.Variable(tf.random_uniform(shape=([n+1, 1]), minval=-1, maxval=1), name='theta')
#计算y的预测值， 此处x和theta不能调换位置  因为x是 m行9列  theta是9行1列 所以相乘结果是m行1列
Y_pred = tf.matmul(X,theta,name='predictions')
error = Y_pred - Y
# 计算评估指标
rmse = tf.sqrt(tf.reduce_mean(tf.square(error), name='rmse'))
# 梯度的公式 （y-pred - y ） * xj
#tf.transpose(X) 是矩阵转置 为什么需要转置  因为error是m行1列  x是m行9列 只有x变成9行一列 列才能得到一个1个值
gradients = 2/m * tf.matmul(tf.transpose(X), error)
# theta 赋值为theta - learning_rate * gradients  对BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.assign(theta, theta - learning_rate * gradients)

#验证w随机 对结果的影响
with tf.Session() as sess:
    for i in range(10):
        # 真正的开始初始化变量
        theta = tf.Variable(tf.random_uniform(shape=([n + 1, 1]), minval=-1, maxval=1), name='theta')
        # 初始化所有的变量
        init = tf.global_variables_initializer()

        sess.run(init)
        print("第",i,"次开始")
        # 定义上一次的rmse的值
        upRmse = 0.0
        # 定义上次mse相同的次数
        minMSECount = 0.0

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                lastRmse = rmse.eval()
                # print('epoch',epoch,'rmse=',lastRmse)
                #验证两次mse是否相等
                if lastRmse == upRmse:
                    minMSECount = minMSECount + 1
                #判断相同的次数是否等于10
                if minMSECount == 10:
                    print('已找到最优解最优解的mse为',lastRmse,"第多少", epoch,'次')
                    break
                upRmse = lastRmse
            #更新w参数
            sess.run(training_op)
        best_theta = theta.eval()
        print(upRmse)
        # print(best_theta)