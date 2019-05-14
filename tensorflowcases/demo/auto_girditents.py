import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 10000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
# 可以使用TensorFlow或者Numpy或者sklearn的StandardScaler去进行归一化
scaler = StandardScaler().fit(housing.data)
scaled_housing_data_plus_bias = scaler.transform(housing.data)
#进行截距项增加
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data_plus_bias]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
Y = tf.constant(housing.target, dtype=tf.float32, name='Y')

#随机创建一组theta
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - Y
mse = tf.reduce_mean(tf.square(error), name='mse')
# 梯度的公式 （y_pred - y）* xj
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
#此处可以增加L1 和 L2正则项
L1 = 0.3 * tf.reduce_sum(tf.abs(theta))
gradients = tf.gradients(mse + L1, [theta])[0]

# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        training_op.eval()
    best_theta = theta.eval()
    print(theta)
