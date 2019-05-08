import tensorflow as tf

x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y+y+2

#with 创建session会在with函数结束后自动关闭session ，不需要sess.close()
#在with快内部，session会被设置为默认的session
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)