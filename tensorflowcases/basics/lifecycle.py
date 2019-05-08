import tensorflow as tf

#当去计算一个节点的时候，Tensorflow自动计算它依赖的一组节点，并且首先计算依赖的节点
#计算节点和 spark运行机制类型， 属于懒加载模式，
#variable 会存储在内存中
#contant 创建完成后需要每次重新赋值
# w = tf.constant(3)
w = tf.Variable(3)
x = w + 2
y = x + 5
z = x * 3

#此种计算方式   x会被计算两次  因为第一次初始化w  然后计算x的值  得到y的值
#第二次  w如果是constant 会被初始化两次  如果是Variable 不需要在初始化  x需要再一次计算
with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(y))
    #这里为了去计算z ，又重新计算了x和w，除了variable值，tf是不会缓存其它比如contant等的值
    #一个variable的生命周期是当它的initializer运行的时候开始，到回话session close的时候结束
    print(sess.run(z))

with tf.Session() as sess:
    sess.run(w.initializer)
    y_val,z_val = sess.run([y,z])
    print(y_val)
    print(z_val)