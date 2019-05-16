import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import numpy as np
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

#构建图阶段
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 50
n_output = 10

mnist = input_data.read_data_sets('../MNIST_data_bak/')
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None,), name='y')
"""
# 构建神经网络层，我们这里两个隐藏层，基本一样，除了输入inputs到每个神经元的连接不同
# 和神经元个数不同
# 输出层也非常相似，只是激活函数从ReLU变成了Softmax而已
def neuron_layer(X, n_neurons, name, activation=None):
    # 包含所有计算节点对于这一层，name_scope可写可不写
    with tf.name_scope(name):
        # 取输入矩阵的维度作为层的输入连接个数
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
        # truncated normal distribution 比 regular normal distribution的值小
        # 不会出现任何大的权重值，确保慢慢的稳健的训练
        # 使用这种标准方差会让收敛快
        # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        # 向量表达的使用比一条一条加和要高效
        z = tf.matmul(X, w) + b

        reg_loss = tf.reduce_sum(tf.abs(w))
        reg_loss = tf.reduce_sum(tf.square(w))
        tf.add_to_collection("reg_losses", reg_loss)

        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    hidden3 = neuron_layer(hidden2, n_hidden3, "hidden2", activation="relu")
    # 进入到softmax之前的结果
    logits = neuron_layer(hidden3, n_outputs, "outputs")

"""
with tf.name_scope('dnn'):
    #同意为with下所有的全连接层增加 L1或者L2正则项
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)):
        hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
        hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
        hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
        #此处因为是用的是softmax的公式 ezi/ez1+ez2+ez3+ez4  因为分母一致，故此处不需要进行 激活函数变化
        logits = fully_connected(hidden3, n_output, scope='outputs', activation_fn=None)
with tf.name_scope("loss"):
    #定义交叉熵损失函数， 并且求个样本平均
    #函数等价于先使用softmax损失函数，在接着计算交叉熵，并且更有效率
    #类似的softmax_corss_entorpy_with_logits 只会给one-hot编码，我们使用的会给0-9分类号
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    #将全连接层增加的 l2正则项 使用， 正则项系数放置在常量池中
    reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.reduce_mean(xentropy, name='loss')
    total_loss = tf.add(loss, reg_losses)
learning_rate = 0.01

with tf.name_scope('train'):
    training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 计算图阶段
n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op], feed_dict={X : X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, 'train accuracy:', acc_train,'test accuracy:', acc_test)
    save_path = saver.save(sess, '../my_dnn_model_final.ckpt')
