from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

my_mnist = input_data.read_data_sets("../MNIST_data_bak/", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=(None, 784))

#因为是神经网络  所以输入层相当于784个神经元  10个是分类的结果  全连接总量为7840个
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

#计算y_pred
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

#计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

#使用梯度下降的优化器进行计算最优解
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.global_variables_initializer()

#创建Saver() 节点
saver = tf.train.Saver()

n_epoch = 100000
with tf.Session() as sess:
    # init.eval()

    ckpt = tf.train.get_checkpoint_state('../ckpt/')
    if ckpt and ckpt.model_checkpoint_path:
        print('hello')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('hello bias %s'% sess.run(b))
    else:
        print('world')
        sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 1000 == 0:
            print('%s bias %s' %(epoch, sess.run(b)))
            save_path = saver.save(sess, "./ckpt/my_model.ckpt", global_step=epoch)

        batch_xs, batch_ys = my_mnist.train.next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    # best_theta = W.eval()
    save_path = saver.save(sess, "./ckpt/my_model_final.ckpt")

