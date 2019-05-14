import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Tensorflow求梯度，同时也提供了更方便的求解方式
#它提供给我们与重不同的，有创意的一些优化器，包括梯度下降优化器

#设置超参数，Grid Search进行栅格搜索，说白了就是排列组合找到loss function最小的参数
n_epochs = 1000
learning_rate = 0.001
batch_size = 2000

#BGD brach gradient decrease，如果数据集比较大的时候，我们更倾向于mini gd
housing = fetch_california_housing()
m, n = housing.data.shape
#进行测试集和验证集 切分
X_train, X_test, y_train, y_test = train_test_split(housing.data,housing.target)

#使用sklearn的StantardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_test = scaler.transform(X_test)
X_test = np.c_[np.ones((len(X_test), 1)), X_test]

#构建计算图 使用placeholder可以使使用mini gd
X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')

#计算正向传播 求解mse
theta = tf.Variable(tf.random_uniform((n + 1, 1), -1, 1), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - Y
mse = tf.reduce_mean(tf.square(error), name='mse')

#使用tensorflow自定的梯度优化器  自动求解梯度   对什么求梯度  对所有的variable求梯度
training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    sess.run(init)

    #计算需要批次
    n_batch = int(len(X_test) / batch_size)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            temp_theta = theta.eval()
            print(temp_theta)

            print('Epoch', epoch, 'mse = ', sess.run(mse, feed_dict={
                X: X_train,
                Y:y_train
            }))

            print('Epoch', epoch, 'mse = ', sess.run(mse, feed_dict={
                X: X_test,
                Y: y_test
            }))
        arr = np.arange(len(X_train))
        np.random.shuffle(arr)
        X_train = X_train[arr]
        y_train = y_train[arr]

        for i in range(n_batch):
            sess.run(training_op, feed_dict={
                X : X_train[i*batch_size: i*batch_size + batch_size],
                Y : y_train[i*batch_size: i*batch_size + batch_size]
            })
    best_theta = theta.eval()
    print(best_theta)


