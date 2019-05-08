import tensorflow as tf
#tf.variable 生成的变量，每次迭代都会变化
#这个变量也就是我们要去计算的结果，所以说你要计算什么，你需要将它定义为变量variable

with tf.device('/cpu:0'):
    x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y+y+2

#创建一个计算图的一个上下文环境
#配置里面是吧据图运行过程在哪里执行都给打印出来
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()