import tensorflow as tf
a = tf.constant(32)
b = tf.constant(10)
c = tf.add(a, b)

sess = tf.Session()
py_a = sess.run(a)
print(type(py_a))
print(py_a)
# 我们可以将执行图的结果保存到正常的变量中, tensorflow称这个过程为fetch.
# 运行下面的命令, 相信你会对fetch有一个初步的了解
py_r = sess.run([a,b,c])
print(type(py_r))
print(py_r[0],py_r[1],py_r[2])

#tensor可以有很多形式
hello = tf.constant('hello, tensorflow!')
boolean = tf.constant(True)
int_array = tf.constant([1,2], dtype=tf.int32)
float_array = tf.constant([1,2], dtype=tf.float32)

print(sess.run(hello))
print(sess.run(boolean))
print(sess.run(int_array))
print(sess.run(float_array))
sess.close()