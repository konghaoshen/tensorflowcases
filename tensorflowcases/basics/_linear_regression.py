import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

#立刻下载数据集
housing = fetch_california_housing(data_home="C:/Users/Shinelon/scikit_learn_data", download_if_missing=True)
#获取x数据行数和列数
m, n = housing.data.shape
print(m, n)
print(housing.data, housing.target)
print(housing.feature_names)

#这里添加一个额外的bias，输入特征（x0= 1）到所有的训练数据上面，因为使用numpy所以会立即执行
# housing_data_plus_bias = np.concatenate[np.ones((m, 1)), housing.data]
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#创建两个tensorflow常量节点x和y，去持有数据和标签
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
#使用一些tensorflow框架提供的矩阵操作去求theta
XT = tf.transpose(X)
# 解析解一步计算出最优解
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)
print(theta)

with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)