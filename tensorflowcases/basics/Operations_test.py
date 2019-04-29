import tensorflow as tf

# tensorflow原生支持很多的operation, 以后我们将用op来简称operation.
#  注意, op也可以有名字用来标识

a = tf.constant(31)
b = tf.constant(10)
c = tf.add(a, b)

sess = tf.Session()
#累加
d = tf.add_n([a,b,b])
#相减
e = tf.subtract(a,b)
#乘法
f = tf.multiply(a,b)
#除法
g = tf.divide(a,b)
#余数
h = tf.mod(a,b)
print(sess.run(d))
print(sess.run(e))
print(sess.run(f))
print(sess.run(g))
print(sess.run(h))
#转换类型
a_float = tf.cast(a, dtype=tf.float32)
b_float = tf.cast(b, dtype=tf.float32)

i = tf.sin(a_float)
j = tf.exp(tf.divide(1.0,a_float))
k = tf.add(i,tf.log(i))

print(sess.run(i))
print(sess.run(j))
print(sess.run(k))

# 进行构造一个tensor, 它的值等于, sigmoid函数如下定义 1 / 1+ ex
sigmoid = tf.divide(1.0,tf.add(1.0, tf.exp(-b_float)))
print(sess.run(sigmoid))

# 可以通过reshape改变形状 tensorflow支持矩阵操作,broadcast机制
mat_a = tf.constant([1,2,3,4])
mat_a = tf.reshape(mat_a,(2,2))
mat_b = tf.constant([1,3,5,7,9,11])
mat_b = tf.reshape(mat_b,(2,3))
vec_a = tf.constant([1,2])
#将矩阵a乘以矩阵b，生成a * b。
mat_c = tf.matmul(mat_a,mat_b)
#两个矩阵中对应元素各自相乘
mat_d = tf.multiply(mat_a,vec_a)

print(sess.run(mat_a))
print(sess.run(mat_b))
print(sess.run(mat_c))
print(sess.run(mat_d))
